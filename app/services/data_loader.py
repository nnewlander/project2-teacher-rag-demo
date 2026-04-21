from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.schemas.document import InternalDocument, RawRecord
from app.utils.jsonl import read_jsonl
from app.utils.text_utils import dedup_keep_order, normalize_text


class DataLoader:
    """
    读取多源 jsonl，并映射为系统内部统一文档结构。

    MVP：接入 documents / support_tickets / faq / code_example 四类。
    TODO:
    - 接入 code_examples / external_refs
    - 支持 OCR 原图（png_ocr）管线：当前只使用 raw_text
    """

    def load_raw_records(self, source: str, path: Path, limit: Optional[int] = None) -> List[RawRecord]:
        rows = list(read_jsonl(path, limit=limit))
        out: List[RawRecord] = []
        for r in rows:
            source_id = (
                r.get("record_id")
                or r.get("ticket_id")
                or r.get("faq_id")
                or r.get("example_id")
                or r.get("id")
                or "unknown"
            )
            out.append(RawRecord(source=source, source_id=str(source_id), payload=r))
        return out

    def to_internal_documents(self, records: Iterable[RawRecord]) -> List[InternalDocument]:
        docs: List[InternalDocument] = []
        for rr in records:
            if rr.source == "document":
                docs.append(self._map_document(rr))
            elif rr.source == "support_ticket":
                docs.append(self._map_support_ticket(rr))
            elif rr.source == "faq":
                docs.append(self._map_faq(rr))
            elif rr.source == "code_example":
                docs.append(self._map_code_example(rr))
            else:
                # 忽略未知来源
                continue
        return docs

    def _map_document(self, rr: RawRecord) -> InternalDocument:
        p = rr.payload
        doc_id = str(p.get("record_id", rr.source_id))
        title = normalize_text(str(p.get("title") or p.get("file_name") or ""))
        text = normalize_text(str(p.get("raw_text") or ""))
        tags = [str(x) for x in (p.get("knowledge_tags") or []) if x]
        tags = dedup_keep_order(tags)
        metadata = {
            "source_type": p.get("source_type"),
            "content_origin": p.get("content_origin"),
            "file_type": p.get("file_type"),
            "department": p.get("department"),
            "product_module": p.get("product_module"),
            "course_stage": p.get("course_stage"),
            "status": p.get("status"),
            "updated_at": p.get("updated_at"),
        }
        return InternalDocument(doc_id=doc_id, source="document", title=title, text=text, tags=tags, metadata=metadata)

    def _map_support_ticket(self, rr: RawRecord) -> InternalDocument:
        p = rr.payload
        doc_id = str(p.get("ticket_id", rr.source_id))
        title = normalize_text(str(p.get("issue_type") or "support_ticket"))

        question = normalize_text(str(p.get("raw_question") or ""))
        resolution = normalize_text(str(p.get("raw_resolution_note") or ""))
        turns = p.get("conversation_turns") or []
        turns_text = "\n".join(
            [normalize_text(f"{t.get('speaker', '')}: {t.get('text', '')}") for t in turns if isinstance(t, dict)]
        ).strip()

        parts = [x for x in [question, resolution, turns_text] if x]
        text = "\n\n".join(parts)

        tags = [str(x) for x in (p.get("knowledge_tags") or []) if x]
        tags = dedup_keep_order(tags)
        metadata = {
            "channel": p.get("channel"),
            "teacher_role": p.get("teacher_role"),
            "campus_region": p.get("campus_region"),
            "product_module": p.get("product_module"),
            "issue_type": p.get("issue_type"),
            "priority": p.get("priority"),
            "course_stage": p.get("course_stage"),
            "grade_band": p.get("grade_band"),
            "status": p.get("status"),
            "updated_at": p.get("updated_at"),
        }
        return InternalDocument(doc_id=doc_id, source="support_ticket", title=title, text=text, tags=tags, metadata=metadata)

    def _map_faq(self, rr: RawRecord) -> InternalDocument:
        p = rr.payload
        doc_id = str(p.get("faq_id", rr.source_id))
        q = normalize_text(str(p.get("standard_question") or p.get("question") or ""))
        a = normalize_text(str(p.get("answer_raw") or ""))
        title = q
        text = "\n\n".join([x for x in [q, a] if x])
        tags = [normalize_text(str(p.get("category") or ""))] if p.get("category") else []
        metadata = {
            "faq_id": doc_id,
            "question": q,
            "answer": a,
            "category": p.get("category"),
            "status": p.get("status"),
            "updated_at": p.get("updated_at"),
            "hit_count_30d": p.get("hit_count_30d"),
            "source_doc_id": p.get("source_doc_id"),
        }
        return InternalDocument(doc_id=doc_id, source="faq", title=title, text=text, tags=tags, metadata=metadata)

    def _map_code_example(self, rr: RawRecord) -> InternalDocument:
        """
        代码示例数据源：raw_code_examples_10pct.jsonl

        目标：
        - 统一映射到 InternalDocument
        - text 尽量保留「主题/代码/说明/常见错误」
        - metadata 保留 source_type=code_example（以及 topic/language/course_stage 等）
        """
        p = rr.payload
        doc_id = str(p.get("example_id", rr.source_id))
        topic = normalize_text(str(p.get("topic") or "代码示例"))
        course_stage = normalize_text(str(p.get("course_stage") or ""))
        language = normalize_text(str(p.get("language") or ""))

        title_parts = [x for x in [topic, course_stage, language] if x]
        title = " · ".join(title_parts) if title_parts else topic

        code = str(p.get("code") or "").rstrip()
        explanation = normalize_text(str(p.get("explanation_raw") or ""))
        expected = normalize_text(str(p.get("expected_output") or ""))
        common_errors = [normalize_text(str(x)) for x in (p.get("common_errors") or []) if x]
        common_errors = dedup_keep_order([x for x in common_errors if x])

        # 用 Markdown 结构化，便于 chunker/检索保留代码块整体语义
        parts: List[str] = []
        parts.append(f"【主题】{topic}" + (f"（课程阶段：{course_stage}）" if course_stage else ""))
        if explanation:
            parts.append(f"【说明】{explanation}")
        if code:
            fence_lang = language.lower() if language else ""
            parts.append(f"【代码】\n```{fence_lang}\n{code}\n```")
        if expected:
            parts.append(f"【预期输出】{expected}")
        if common_errors:
            parts.append("【常见错误】" + "、".join(common_errors))

        text = "\n\n".join([x for x in parts if x]).strip()

        tags = []
        if course_stage:
            tags.append(course_stage)
        if topic:
            tags.append(topic)
        tags = dedup_keep_order(tags)

        metadata = {
            # 关键：下游检索/证据链用 meta['source'] 区分来源类型
            "source_type": "code_example",
            "topic": p.get("topic"),
            "course_stage": p.get("course_stage"),
            "language": p.get("language"),
            "file_name": p.get("file_name"),
            "source_repo": p.get("source"),
            "common_errors": common_errors,
            "noise_flags": p.get("noise_flags"),
        }
        return InternalDocument(doc_id=doc_id, source="code_example", title=title, text=text, tags=tags, metadata=metadata)

