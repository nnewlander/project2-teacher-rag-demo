from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Tuple

from app.schemas.document import DocumentChunk, InternalDocument


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 600
    chunk_overlap: int = 80
    # 父块摘要会附在子块里，便于生成阶段补上下文（不改下游代码的最小兼容做法）
    parent_summary_chars: int = 400


class Chunker:
    """
    父子块切分（基础版，兼容现有结构）。

    TODO:
    - 更好地识别标题层级（markdown/编号标题）
    - 代码块保护（``` / 缩进代码）与按块切分
    - parent/child 分离存储：检索只用 child，生成按需回溯 parent
    """

    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg

    def chunk_documents(self, docs: Iterable[InternalDocument]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for d in docs:
            chunks.extend(self._chunk_one(d))
        return chunks

    def _chunk_one(self, doc: InternalDocument) -> List[DocumentChunk]:
        text = (doc.text or "").strip()
        if not text:
            return []

        # code_example：尽量保留“主题/代码/说明”整体，不做句子级切分，避免破坏代码块。
        # 最小实现：创建 1 个 parent + 1 个 child（child 用于检索，parent 用于回溯/生成），文本保持原样。
        if getattr(doc, "source", None) == "code_example":
            parent_id = f"{doc.doc_id}::p0"
            parent_text = self._format_parent_text(doc, text)
            parent = DocumentChunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                text=parent_text,
                start=0,
                end=len(text),
                parent_id=None,
                metadata={"source": doc.source, "title": doc.title, "chunk_level": "parent"},
            )
            child = DocumentChunk(
                chunk_id=f"{doc.doc_id}::c0",
                doc_id=doc.doc_id,
                text=self._format_child_text(text, self._truncate(parent_text, int(self.cfg.parent_summary_chars))),
                start=0,
                end=len(text),
                parent_id=parent_id,
                metadata={"source": doc.source, "title": doc.title, "chunk_level": "child", "parent_id": parent_id},
            )
            return [parent, child]

        out: List[DocumentChunk] = []
        size = max(200, int(self.cfg.chunk_size))
        overlap = max(0, int(self.cfg.chunk_overlap))
        if overlap >= size:
            overlap = max(0, size // 4)

        # 1) 轻量段落切分：按空行切段；再识别“像标题”的短行作为段落边界增强
        paragraphs = self._split_paragraphs(text)

        parent_idx = 0
        child_idx = 0
        cursor = 0  # 用于 start/end 的近似定位（不追求精确字符偏移）

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            parent_id = f"{doc.doc_id}::p{parent_idx}"
            parent_idx += 1

            parent_text = self._format_parent_text(doc, para)
            parent_start = cursor
            cursor += len(para) + 1
            parent_end = parent_start + len(para)

            out.append(
                DocumentChunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    text=parent_text,
                    start=parent_start,
                    end=parent_end,
                    parent_id=None,
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "chunk_level": "parent",
                    },
                )
            )

            # 2) 子块：在父段落内按句子/标点分段，再按 chunk_size 组装
            sentences = self._split_sentences(para)
            if not sentences:
                continue

            parent_summary = self._truncate(parent_text, int(self.cfg.parent_summary_chars))
            assembled, assembled_start = "", 0
            sent_starts = self._sentence_offsets(para, sentences)

            for s, s_start in zip(sentences, sent_starts):
                s = s.strip()
                if not s:
                    continue
                if not assembled:
                    assembled = s
                    assembled_start = s_start
                elif len(assembled) + 1 + len(s) <= size:
                    assembled = assembled + "\n" + s
                else:
                    child_text = self._format_child_text(assembled, parent_summary)
                    out.append(
                        DocumentChunk(
                            chunk_id=f"{doc.doc_id}::c{child_idx}",
                            doc_id=doc.doc_id,
                            text=child_text,
                            start=parent_start + assembled_start,
                            end=parent_start + assembled_start + len(assembled),
                            parent_id=parent_id,
                            metadata={
                                "source": doc.source,
                                "title": doc.title,
                                "chunk_level": "child",
                                "parent_id": parent_id,
                            },
                        )
                    )
                    child_idx += 1

                    # overlap：按字符近似回退（在句子边界内尽量回退）
                    if overlap > 0 and len(assembled) > overlap:
                        assembled = assembled[-overlap:]
                        assembled_start = max(0, assembled_start + len(assembled) - overlap)
                    else:
                        assembled = s
                        assembled_start = s_start

                    # 放入当前句子
                    if assembled != s and len(assembled) + 1 + len(s) <= size:
                        assembled = assembled + "\n" + s

            if assembled:
                child_text = self._format_child_text(assembled, parent_summary)
                out.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}::c{child_idx}",
                        doc_id=doc.doc_id,
                        text=child_text,
                        start=parent_start + assembled_start,
                        end=parent_start + assembled_start + len(assembled),
                        parent_id=parent_id,
                        metadata={
                            "source": doc.source,
                            "title": doc.title,
                            "chunk_level": "child",
                            "parent_id": parent_id,
                        },
                    )
                )
                child_idx += 1

        return out

    def _split_paragraphs(self, text: str) -> List[str]:
        # 先按空行切段
        parts = re.split(r"\n\s*\n+", text)
        out: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # 如果段落内包含很多“短行像标题”，再按标题行切一次
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            buf: List[str] = []
            for ln in lines:
                is_title_like = (len(ln) <= 30) and bool(re.match(r"^([#*]+|\d+[\.\、]|[一二三四五六七八九十]+[、\.])", ln))
                if is_title_like and buf:
                    out.append("\n".join(buf).strip())
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                out.append("\n".join(buf).strip())
        return out

    def _split_sentences(self, text: str) -> List[str]:
        """
        轻量句子切分：
        - 以中文/英文句末标点切分
        - 保留代码块/路径等：不做复杂规则，只按标点+换行切
        """
        text = text.replace("\r\n", "\n")
        # 先按换行切，再对每行按句末标点切
        segments: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 句末标点切分（把标点留在句子里）
            pieces = re.split(r"(?<=[。！？?!；;])\s*", line)
            for pc in pieces:
                pc = pc.strip()
                if pc:
                    segments.append(pc)
        return segments

    def _sentence_offsets(self, paragraph: str, sentences: List[str]) -> List[int]:
        # 近似定位每个句子在段落里的起始位置（用于 start/end）
        offsets: List[int] = []
        pos = 0
        for s in sentences:
            i = paragraph.find(s, pos)
            if i == -1:
                i = pos
            offsets.append(i)
            pos = i + len(s)
        return offsets

    def _format_parent_text(self, doc: InternalDocument, paragraph: str) -> str:
        # parent：用于生成补上下文（尽量保留较完整段落）
        if doc.title:
            return f"【标题】{doc.title}\n{paragraph}"
        return paragraph

    def _format_child_text(self, child: str, parent_summary: str) -> str:
        # child：用于检索。为了不改下游生成逻辑，把父块摘要拼在末尾（轻量、可解释、兼容）
        return f"{child}\n\n【父块摘要】{parent_summary}"

    def _truncate(self, text: str, n: int) -> str:
        t = (text or "").strip().replace("\n", " ")
        if len(t) <= n:
            return t
        return t[:n] + "…"

