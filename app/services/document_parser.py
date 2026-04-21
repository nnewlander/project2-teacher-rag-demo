from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Optional


class _HTMLTextExtractor(HTMLParser):
    """
    最小 HTML 文本抽取（不引入第三方依赖）。
    - 忽略 script/style
    - 仅拼接可见文本
    """

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        t = (tag or "").lower()
        if t in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        t = (tag or "").lower()
        if t in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return
        if data and data.strip():
            self._buf.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._buf)


def _compact_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _guess_title_from_text(text: str, fallback: str) -> str:
    # 取前几行里第一个“像标题”的行
    for line in (text or "").splitlines()[:10]:
        t = line.strip().lstrip("#").strip()
        if 4 <= len(t) <= 60:
            return t
    return fallback


@dataclass(frozen=True)
class ParsedRawDocument:
    """
    统一解析输出（用于离线导出 jsonl，供 DataLoader/Cleaner/Chunker 复用）。
    """

    source_id: str
    source_type: str
    title: str
    raw_text: str
    metadata: Dict[str, Any]
    parse_method: str
    ocr_used: bool

    def to_jsonl_row(self) -> Dict[str, Any]:
        # 兼容 DataLoader._map_document 的字段命名（record_id/title/raw_text/file_type/...）
        return {
            "source_id": self.source_id,
            "record_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
            "parse_method": self.parse_method,
            "ocr_used": self.ocr_used,
            # 常见文档字段（可选）
            "file_name": self.metadata.get("file_name"),
            "file_type": self.metadata.get("file_type"),
            "updated_at": self.metadata.get("updated_at"),
        }


class DocumentParser:
    """
    多格式原始资料解析（离线）。

    支持：
    - .txt / .md / .html
    - .docx（纯 Python：读取 word/document.xml）
    - .pdf（文本提取 + OCR fallback 接口预留）

    设计目标：
    - 不改主链路，只生成可直接接入 DataLoader 的“document 源 jsonl”
    - OCR 不接入真实引擎，但保留触发规则与接口（未来可接 Tesseract/云 OCR）
    """

    def __init__(
        self,
        *,
        default_source_type: str = "course_doc",
        pdf_min_text_chars_for_non_ocr: int = 120,
    ) -> None:
        self.default_source_type = default_source_type
        self.pdf_min_text_chars_for_non_ocr = int(pdf_min_text_chars_for_non_ocr)

    def parse_file(
        self,
        path: Path,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedRawDocument:
        p = Path(path)
        ext = p.suffix.lower()
        sid = source_id or self._make_source_id(p)
        stype = source_type or self.default_source_type

        meta: Dict[str, Any] = {
            "file_path": str(p.as_posix()),
            "file_name": p.name,
            "file_type": ext.lstrip("."),
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        if ext in (".txt", ".md"):
            raw = p.read_text(encoding="utf-8", errors="ignore")
            text = _compact_whitespace(raw)
            title = _guess_title_from_text(text, fallback=p.stem)
            return ParsedRawDocument(
                source_id=sid,
                source_type=stype,
                title=title,
                raw_text=text,
                metadata=meta,
                parse_method=f"{ext.lstrip('.')}_plain",
                ocr_used=False,
            )

        if ext in (".html", ".htm"):
            raw = p.read_text(encoding="utf-8", errors="ignore")
            extractor = _HTMLTextExtractor()
            extractor.feed(raw)
            text = _compact_whitespace(extractor.text())
            title = _guess_title_from_text(text, fallback=p.stem)
            return ParsedRawDocument(
                source_id=sid,
                source_type=stype,
                title=title,
                raw_text=text,
                metadata=meta,
                parse_method="html_strip",
                ocr_used=False,
            )

        if ext == ".docx":
            text = _compact_whitespace(self._extract_docx_text(p))
            title = _guess_title_from_text(text, fallback=p.stem)
            return ParsedRawDocument(
                source_id=sid,
                source_type=stype,
                title=title,
                raw_text=text,
                metadata=meta,
                parse_method="docx_xml",
                ocr_used=False,
            )

        if ext == ".pdf":
            text = _compact_whitespace(self._extract_pdf_text(p))
            parse_method = "pdf_text"
            ocr_used = False

            # OCR fallback 触发规则（预留）：
            # - 文本提取结果极少（疑似扫描件/图片 PDF）
            # - 或仅有少量页眉页脚/噪声（此处用长度阈值做最小判断）
            if len(text) < self.pdf_min_text_chars_for_non_ocr:
                ocr_text = _compact_whitespace(self.ocr_extract(p))
                if ocr_text:
                    text = ocr_text
                    ocr_used = True
                    parse_method = "pdf_ocr_fallback"
                else:
                    # 未接入 OCR 时：保留接口与触发信息到 metadata，方便后续接入
                    meta["ocr_fallback_triggered"] = True
                    meta["ocr_fallback_reason"] = f"text_chars<{self.pdf_min_text_chars_for_non_ocr}"

            title = _guess_title_from_text(text, fallback=p.stem)
            return ParsedRawDocument(
                source_id=sid,
                source_type=stype,
                title=title,
                raw_text=text,
                metadata=meta,
                parse_method=parse_method,
                ocr_used=ocr_used,
            )

        raise ValueError(f"Unsupported file type: {ext} ({p})")

    def ocr_extract(self, pdf_path: Path) -> str:
        """
        OCR 接口预留（当前不接入真实 OCR 引擎）。

        未来可在这里接入：
        - Tesseract（本地）
        - PaddleOCR / RapidOCR
        - 云 OCR（如通用文字识别）

        返回：
        - OCR 文本（字符串）；若未实现/失败则返回空字符串
        """
        _ = pdf_path
        return ""

    def _make_source_id(self, path: Path) -> str:
        # 稳定且可读：相对路径风格（避免绝对路径泄露）
        s = str(path.as_posix())
        return re.sub(r"[^A-Za-z0-9_\-./]+", "_", s).strip("_")

    def _extract_docx_text(self, path: Path) -> str:
        # 不引入 python-docx：直接读 OOXML（word/document.xml）
        try:
            with zipfile.ZipFile(path, "r") as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Failed to parse docx: {path}") from e

        # 粗略抽取：把标签去掉，保留文本节点
        xml = re.sub(r"</w:p>", "\n", xml)  # 段落换行
        xml = re.sub(r"<[^>]+>", "", xml)
        xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return xml

    def _extract_pdf_text(self, path: Path) -> str:
        """
        PDF 文本提取（依赖轻量库 pypdf）。
        - 若环境缺少 pypdf，会抛出明确错误提示（便于离线链路安装依赖）
        """
        try:
            from pypdf import PdfReader  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("缺少依赖 pypdf（用于 PDF 文本提取），请先 pip install -r requirements.txt") from e

        reader = PdfReader(str(path))
        out: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                out.append(t)
        return "\n\n".join(out)

