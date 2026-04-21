from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional

from app.config import Settings
from app.schemas.query import RetrievedContext


@dataclass(frozen=True)
class LLMResult:
    answer: str
    model: str


class LLMClient:
    """
    最小可替换 LLM 客户端（LangChain 轻封装）。

    当前支持：
    - openai_compatible：OpenAI 兼容 API（base_url/api_key/model_name 可配置）

    TODO:
    - 扩展 provider：Ollama、本地推理服务等
    - 增加输出格式约束（JSON/citations）与安全过滤
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._logger = logging.getLogger("kbqa.llm")

    def is_enabled(self) -> bool:
        if (self.settings.llm_provider or "").strip().lower() == "openai_compatible":
            return bool((self.settings.llm_model_name or "").strip())
        return False

    def generate(self, query: str, contexts: List[RetrievedContext]) -> Optional[LLMResult]:
        if not self.is_enabled():
            return None

        provider = (self.settings.llm_provider or "").strip().lower()
        if provider != "openai_compatible":
            return None

        # LangChain：prompt + llm + parser
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        system_prompt = (
            "你是“教师智能知识库问答系统（RAG）”的答案生成器。\n"
            "\n"
            "硬性规则（必须遵守）：\n"
            "1) 只能基于【检索证据】中明确出现的信息回答；证据没有提到的内容，一律不要编造。\n"
            "2) 如果证据不足以支撑结论，必须明确说“证据不足”，并给出需要补充的具体信息（例如：教师端/学生端、模块名、课程阶段、完整报错、代码前后 5 行、截图文字等）。\n"
            "3) 输出要像真实产品：先给结论，再给步骤/解释，最后给依据来源。\n"
            "4) 不要输出与证据无关的泛化长文；不要提及你是模型；不要编造引用。\n"
            "\n"
            "格式要求（请用 Markdown 输出）：\n"
            "- **结论**：1~3 句话\n"
            "- **建议步骤**：可执行的条目（平台路径/排查顺序/课堂讲法）\n"
            "- **证据不足时**：明确列出“缺什么信息”\n"
            "- **依据来源**：列出你用到的证据条目编号（如 E1/E3），并可摘录 1 句关键原文\n"
        )

        evidence = self._format_contexts(contexts, max_items=8, max_chars_each=900)

        user_prompt = (
            "【问题】\n{query}\n\n"
            "【检索证据】\n{evidence}\n\n"
            "请严格按“格式要求”生成答案。注意：如果证据无法支持确定结论，必须输出“证据不足”。\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        llm = ChatOpenAI(
            api_key=self.settings.llm_api_key or None,
            base_url=self.settings.llm_base_url or None,
            model=self.settings.llm_model_name,
            temperature=float(self.settings.llm_temperature),
            timeout=int(self.settings.llm_timeout_s),
        )

        chain = prompt | llm | StrOutputParser()
        try:
            text = chain.invoke({"query": query, "evidence": evidence})
        except Exception as e:
            # LLM 下游不可用/鉴权失败/502 等情况时，不要让接口直接失败。
            # 返回 None，让 QAService 安全走 fallback。
            self._logger.warning("LLM generate failed: %r", e)
            return None

        return LLMResult(answer=str(text).strip(), model=self.settings.llm_model_name)

    def _format_contexts(self, contexts: List[RetrievedContext], max_items: int, max_chars_each: int) -> str:
        if not contexts:
            return "（无）"
        lines: List[str] = []
        for i, c in enumerate(contexts[:max_items], start=1):
            t = (c.text or "").strip()
            if len(t) > max_chars_each:
                t = t[:max_chars_each] + "…"
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            source_type = str(meta.get("source") or c.source)
            title = str(meta.get("title") or "")
            parent_id = str(meta.get("parent_id") or meta.get("parentId") or "")
            doc_id = str(meta.get("doc_id") or "")
            hdr = (
                f"[E{i}] type={source_type} id={c.source_id} score={c.score:.3f}"
                + (f" title={title}" if title else "")
                + (f" parent_id={parent_id}" if parent_id else "")
                + (f" doc_id={doc_id}" if doc_id else "")
            )
            lines.append(f"{hdr}\n{t}")
        return "\n\n".join(lines)

