from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from app.config import Settings


@dataclass(frozen=True)
class HyDEResult:
    hyde_text: str
    debug: Dict[str, Any]


class HyDEGenerator:
    """
    HyDE（Hypothetical Document Embeddings）最小可运行版：
    - 若已配置 LLM：生成一段“假设性答案/检索描述”，用于辅助召回
    - 若未配置 LLM：安全降级为不生成（hyde_text 为空），不报错

    注意：这里只做单轮、短提示词的文本生成，不做复杂多轮推理。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._logger = logging.getLogger("kbqa.hyde")

    def generate(self, query: str) -> HyDEResult:
        # 与 LLMClient.is_enabled 保持同一判断口径
        provider = (self.settings.llm_provider or "").strip().lower()
        enabled = provider == "openai_compatible" and bool((self.settings.llm_model_name or "").strip())
        if not enabled:
            return HyDEResult(
                hyde_text="",
                debug={
                    "enabled": False,
                    "reason": "llm_disabled",
                    "provider": provider or "disabled",
                    "model": (self.settings.llm_model_name or "").strip() or None,
                },
            )

        # LangChain：尽量复用项目现有依赖，不引入新包
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        system_prompt = (
            "你是中文教育场景的知识库检索助手。"
            "你的任务不是直接回答用户，而是生成一段“用于检索的假设性描述”。"
            "要求：\n"
            "- 只输出一段文本（不要列表/不要引用编号/不要客套话）\n"
            "- 200~400 字左右\n"
            "- 尽量把问题中的隐含关键词、同义词、可能的模块名/功能名写出来\n"
            "- 不要编造具体不可验证的数字或政策条款\n"
        )

        user_prompt = "用户问题：{query}\n\n请生成用于检索的假设性描述："
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])

        llm = ChatOpenAI(
            api_key=self.settings.llm_api_key or None,
            base_url=self.settings.llm_base_url or None,
            model=self.settings.llm_model_name,
            temperature=0.2,
            timeout=int(self.settings.llm_timeout_s),
        )

        chain = prompt | llm | StrOutputParser()
        try:
            text = str(chain.invoke({"query": query}) or "").strip()
        except Exception as e:
            # HyDE 仅用于辅助召回；失败时不应影响整体 /ask。
            self._logger.warning("HyDE generate failed: %r", e)
            return HyDEResult(
                hyde_text="",
                debug={
                    "enabled": True,
                    "provider": provider,
                    "model": self.settings.llm_model_name,
                    "reason": "llm_error",
                    "error": str(e),
                },
            )

        # 最小安全截断：避免过长影响检索效率
        if len(text) > 800:
            text = text[:800].strip()

        return HyDEResult(
            hyde_text=text,
            debug={
                "enabled": True,
                "provider": provider,
                "model": self.settings.llm_model_name,
                "hyde_chars": len(text),
            },
        )

