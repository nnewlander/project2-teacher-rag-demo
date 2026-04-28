from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.schemas.answer import AnswerResponse, Citation, RouteTraceItem
from app.schemas.document import DocumentChunk, InternalDocument
from app.schemas.query import RetrievedContext
from app.schemas.search import SearchHit, SearchResponse
from app.services.cache import create_answer_cache
from app.services.chunker import Chunker, ChunkingConfig
from app.services.cleaner import Cleaner
from app.services.data_loader import DataLoader
from app.services.evidence_filter import filter_evidence
from app.services.faq_retriever import FaqRetriever
from app.services.hybrid_retriever import HybridRetriever
from app.services.hyde_generator import HyDEGenerator
from app.services.llm_client import LLMClient
from app.services.query_processor import QueryProcessor
from app.services.router import Router
from app.services.reranker import create_reranker_from_settings
from app.services.subquery_builder import SubqueryBuilder
from app.services.vector_retriever import VectorRetriever


@dataclass
class KnowledgeBaseArtifacts:
    docs: List[InternalDocument]
    chunks_count: int
    faq_count: int


class QAService:
    _ERROR_TYPE_SYNONYMS: Dict[str, List[str]] = {
        "NameError": ["变量未定义", "名称未定义", "变量名错误", "函数名未定义"],
        "TypeError": ["类型错误", "类型不匹配", "参数类型错误"],
        "SyntaxError": ["语法错误", "冒号缺失", "缩进错误"],
        "IndexError": ["下标越界", "索引越界"],
        "KeyError": ["字典键不存在"],
        "ValueError": ["值错误", "参数值错误"],
        "ModuleNotFoundError": ["模块未找到", "库未安装"],
    }
    _ERROR_TYPE_ORDER: List[str] = [
        "ModuleNotFoundError",
        "SyntaxError",
        "NameError",
        "TypeError",
        "IndexError",
        "KeyError",
        "ValueError",
    ]
    _ERROR_TYPE_RE = re.compile(r"\b(ModuleNotFoundError|SyntaxError|NameError|TypeError|IndexError|KeyError|ValueError)\b", re.IGNORECASE)

    """
    串起 MVP 全流程：
    query -> router -> retriever -> context -> answer

    说明：
    - answer 目前是 mock 生成（拼装 + 引用片段），后续接 LLM（LangChain 轻封装）即可。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = DataLoader()
        self.cleaner = Cleaner()
        self.chunker = Chunker(ChunkingConfig(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap))

        self.faq = FaqRetriever()
        self.vec = VectorRetriever()
        self._light_reranker = create_reranker_from_settings(settings)
        self.hybrid = HybridRetriever(self.faq, self.vec, reranker=self._light_reranker)
        self.router = Router()
        self.query_processor = QueryProcessor()
        self.hyde = HyDEGenerator(settings)
        self.subquery_builder = SubqueryBuilder()
        self.llm = LLMClient(settings)
        self._ready = False
        self._artifacts: Optional[KnowledgeBaseArtifacts] = None
        self._init_limit_used: Optional[int] = None
        # 父块回溯表：parent_chunk_id -> DocumentChunk
        self._parent_chunks: Dict[str, DocumentChunk] = {}
        # 便于回溯相邻 child chunk：chunk_id -> DocumentChunk
        self._chunks_by_id: Dict[str, DocumentChunk] = {}

        # 轻量缓存（FAQ 直答 + 高频 query 结果）
        self._cache = create_answer_cache(settings)
        self._query_counts: Dict[str, int] = {}

        self._logger = logging.getLogger("kbqa.qa")
        # 轻量 warmup（FAQ/BM25）状态
        self._last_warmup_error: Optional[str] = None
        self._last_warmup_cost_ms: Optional[float] = None

    def _pick_ask_init_limit(self) -> Optional[int]:
        ask_limit = int(getattr(self.settings, "ask_init_limit", 0) or 0)
        return ask_limit if ask_limit > 0 else None

    def _ensure_initialized(self, limit: Optional[int] = None) -> None:
        if self._ready:
            return
        t0 = time.perf_counter()
        self._logger.info("start init qa service limit=%s", limit)
        try:
            self.init_kb(limit=limit)
        except Exception:
            cost = (time.perf_counter() - t0) * 1000
            self._logger.exception("finish init qa service failed cost_ms=%.2f", cost)
            raise
        cost = (time.perf_counter() - t0) * 1000
        self._logger.info("finish init qa service cost_ms=%.2f", cost)

    def ready_status(self) -> Dict[str, Any]:
        faq_ready = bool(getattr(self.faq, "_faq_docs", []))
        bm25_ready = bool(getattr(self.faq, "_bm25", None) is not None)
        vector_ready = bool(getattr(self.vec, "_index", None) is not None)
        model_loaded = bool(getattr(self.vec, "_model", None) is not None)
        faq_doc_count = int(len(getattr(self.faq, "_faq_docs", []) or []))
        bm25_doc_count = int(len(getattr(self.faq, "_corpus_tokens", []) or []))
        lightweight_ready = bool(faq_ready and bm25_ready)
        fallback_ready = True
        lightweight_search_ready = bool(faq_ready and bm25_ready)
        full_rag_ready = bool(vector_ready and model_loaded)
        serving_mode = "full_rag" if full_rag_ready else ("lightweight_search" if lightweight_search_ready else "fallback_only")
        if full_rag_ready:
            status = "ready"
        elif lightweight_search_ready:
            status = "partial_ready"
        elif fallback_ready:
            status = "fallback_only"
        else:
            status = "not_ready"
        return {
            "status": status,
            "faq_ready": faq_ready,
            "bm25_ready": bm25_ready,
            "vector_ready": vector_ready,
            "model_loaded": model_loaded,
            "lightweight_ready": lightweight_ready,
            "fallback_ready": fallback_ready,
            "lightweight_search_ready": lightweight_search_ready,
            "full_rag_ready": full_rag_ready,
            "serving_mode": serving_mode,
            "search_mode": str(getattr(self.settings, "search_mode", "lightweight") or "lightweight"),
            "vector_enabled_for_search": str(getattr(self.settings, "search_mode", "lightweight") or "lightweight").lower() == "hybrid",
            "embedding_model_cached": bool(getattr(self.vec, "_model", None) is not None),
            "faq_doc_count": faq_doc_count,
            "bm25_doc_count": bm25_doc_count,
            "last_warmup_error": self._last_warmup_error,
            "last_warmup_cost_ms": self._last_warmup_cost_ms,
        }

    def warmup_lightweight(self, *, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        轻量 warmup：只初始化 FAQ/BM25，不加载 embedding model，不构建向量索引。
        """
        t0 = time.perf_counter()
        self._last_warmup_error = None
        self._last_warmup_cost_ms = None
        try:
            faq_records = self.loader.load_raw_records("faq", Path(self.settings.raw_faq_path), limit=limit)
            docs = self.loader.to_internal_documents(faq_records)
            docs = self.cleaner.clean_documents(docs)

            # 最小种子 FAQ：确保代码报错类在无数据/弱数据时也可被 BM25 命中
            docs.extend(self._seed_error_faq_documents())
            docs = self.cleaner.clean_documents(docs)

            faq_docs = [d for d in docs if d.source == "faq"]
            self.faq.build(faq_docs)
            cost_ms = (time.perf_counter() - t0) * 1000
            self._last_warmup_cost_ms = round(cost_ms, 2)
            self._logger.info("warmup lightweight ok cost_ms=%.2f faq_docs=%d", cost_ms, len(faq_docs))
            return {"ok": True, "cost_ms": self._last_warmup_cost_ms, **self.ready_status()}
        except Exception as e:
            cost_ms = (time.perf_counter() - t0) * 1000
            self._last_warmup_cost_ms = round(cost_ms, 2)
            self._last_warmup_error = repr(e)
            self._logger.warning("warmup lightweight failed cost_ms=%.2f err=%r", cost_ms, e)
            return {"ok": False, "cost_ms": self._last_warmup_cost_ms, "error": self._last_warmup_error, **self.ready_status()}

    def _seed_error_faq_documents(self) -> List[InternalDocument]:
        seeds: List[InternalDocument] = []
        seeds.append(
            InternalDocument(
                doc_id="seed-faq-nameerror",
                source="faq",
                title="NameError 常见原因说明",
                text="NameError 通常表示变量或函数名在使用前未定义，也可能是大小写、拼写或作用域问题。",
                metadata={
                    "question": "课堂演示遇到 NameError，应该怎么给学生解释？",
                    "answer": "先定位报错行，再检查变量/函数是否先定义后使用；检查拼写/大小写；检查作用域（函数内外、for/if 代码块）。",
                    "category": "代码报错",
                },
            )
        )
        seeds.append(
            InternalDocument(
                doc_id="seed-faq-typeerror",
                source="faq",
                title="TypeError 常见原因说明",
                text="TypeError 常见于参数类型不匹配、对 None 做运算、字符串与数值混用。",
                metadata={
                    "question": "课堂上遇到 TypeError，怎么排查？",
                    "answer": "先看报错提示的操作符/函数，再确认入参类型；逐步 print/type 检查；避免 None 参与运算；必要时做显式类型转换。",
                    "category": "代码报错",
                },
            )
        )
        seeds.append(
            InternalDocument(
                doc_id="seed-faq-syntaxerror",
                source="faq",
                title="SyntaxError 常见原因说明",
                text="SyntaxError 表示语法错误，常见于冒号缺失、括号不配对、缩进错误。",
                metadata={
                    "question": "SyntaxError 报错怎么讲解更清楚？",
                    "answer": "让学生从报错行往上回看结构：if/for/def 是否漏冒号；括号是否配对；缩进是否一致；先保证代码能运行再优化写法。",
                    "category": "代码报错",
                },
            )
        )
        return seeds

    def init_kb(self, limit: Optional[int] = None) -> KnowledgeBaseArtifacts:
        # 1) load raw -> internal docs
        docs_records = self.loader.load_raw_records("document", Path(self.settings.raw_documents_path), limit=limit)
        tkt_records = self.loader.load_raw_records("support_ticket", Path(self.settings.raw_support_tickets_path), limit=limit)
        faq_records = self.loader.load_raw_records("faq", Path(self.settings.raw_faq_path), limit=limit)
        code_records = self.loader.load_raw_records("code_example", Path(self.settings.raw_code_examples_path), limit=limit)

        docs = self.loader.to_internal_documents([*docs_records, *tkt_records, *faq_records, *code_records])
        docs = self.cleaner.clean_documents(docs)

        # 2) build faq index
        faq_docs = [d for d in docs if d.source == "faq"]
        self.faq.build(faq_docs)

        # 3) build vector index (chunk non-faq docs; faq 也可入库，这里先入以增加覆盖)
        chunks = self.chunker.chunk_documents(docs)
        self.vec.build(chunks)
        self._parent_chunks = {
            c.chunk_id: c
            for c in chunks
            if isinstance(c.metadata, dict) and str(c.metadata.get("chunk_level") or "") == "parent"
        }
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

        artifacts = KnowledgeBaseArtifacts(docs=docs, chunks_count=len(chunks), faq_count=len(faq_docs))
        self._artifacts = artifacts
        self._init_limit_used = int(limit) if limit is not None else None
        self._ready = True
        return artifacts

    def ask(self, query: str, top_k: Optional[int] = None) -> AnswerResponse:
        kb_init_debug = {
            "phase": "reuse",
            "requested_limit": None,
            "used_limit": self._init_limit_used,
            "reason": "already_ready",
        }
        ask_init_limit = self._pick_ask_init_limit()
        if not self._ready:
            self._ensure_initialized(limit=ask_init_limit)
            kb_init_debug = {
                "phase": "ask_init",
                "requested_limit": ask_init_limit,
                "used_limit": self._init_limit_used,
                "reason": "not_ready",
            }
        elif (self._init_limit_used or 0) > 0:
            # /health 可能在 strict 环境下只做了轻量抽样初始化；首次 /ask 允许继续使用 ask_init_limit 做项目级验证。
            current_limit = int(self._init_limit_used or 0)
            target_limit = ask_init_limit
            if target_limit is None:
                self.init_kb(limit=None)
                kb_init_debug = {
                    "phase": "ask_reinit",
                    "requested_limit": None,
                    "used_limit": self._init_limit_used,
                    "reason": "upgrade_partial_to_full",
                }
            elif current_limit != target_limit:
                self.init_kb(limit=target_limit)
                kb_init_debug = {
                    "phase": "ask_reinit",
                    "requested_limit": target_limit,
                    "used_limit": self._init_limit_used,
                    "reason": "upgrade_partial_to_ask_limit",
                }
            else:
                kb_init_debug = {
                    "phase": "reuse_partial",
                    "requested_limit": target_limit,
                    "used_limit": self._init_limit_used,
                    "reason": "existing_partial_matches_ask_limit",
                }

        t_total0 = time.perf_counter()
        qp = self.query_processor.process(query)
        q = qp.cleaned_query
        if not q:
            return AnswerResponse(
                query=query,
                mode="rag_mock",
                answer="请提供一个更具体的问题。",
                contexts=[],
                citations=[],
                matched_sources=[],
                filtered_out_count=0,
                kept_context_count=0,
                route_trace=[
                    RouteTraceItem(
                        step="query.process",
                        detail={"cleaned_query": qp.cleaned_query, "query_type": qp.query_type, **(qp.debug or {})},
                    ),
                    RouteTraceItem(step="router.decide", detail={"route": "hybrid", "reason": "empty_query"}),
                ],
                debug={"reason": "empty_query", "query_type": qp.query_type, "kb_init": kb_init_debug},
            )

        # 统计 query 频次（用于“高频才缓存”）
        self._query_counts[q] = self._query_counts.get(q, 0) + 1

        # 读取缓存（仅对非 need_clarify 的最终结果缓存；key 包含 top_k）
        cache_enabled = bool(getattr(self.settings, "cache_enabled", True))
        cache_key = f"ask::{q}::k={top_k if top_k is not None else 'default'}"
        if cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # 轻量可观测性：命中缓存也给出 timing
                cached.debug = {**(cached.debug or {}), "cache": {"hit": True}}
                self._logger.info("ask cache_hit route=%s", getattr(cached, "route", None))
                return cached

        decision = self.router.decide(q, query_type=qp.query_type)
        debug = {"route": decision.route, "route_reason": decision.reason, "kb_init": kb_init_debug}
        route_trace: List[RouteTraceItem] = [
            RouteTraceItem(
                step="query.process",
                detail={"cleaned_query": qp.cleaned_query, "query_type": qp.query_type, **(qp.debug or {})},
            ),
            RouteTraceItem(step="router.decide", detail={"route": decision.route, "reason": decision.reason, "eval_label": getattr(decision, "eval_label", None)}),
        ]

        # need_clarify：信息不足时先澄清，不进入检索生成
        if decision.route == "need_clarify":
            clarifications = self._build_clarifications(q)
            resp = AnswerResponse(
                query=q,
                mode="rag_mock",
                route="need_clarify",
                answer="为了更准确地回答，我需要你补充一些信息：",
                clarifications=clarifications,
                contexts=[],
                citations=[],
                matched_sources=[],
                filtered_out_count=0,
                kept_context_count=0,
                route_trace=route_trace + [RouteTraceItem(step="need_clarify", detail={"questions": clarifications})],
                debug={**debug, "timing_ms": {"total": round((time.perf_counter() - t_total0) * 1000, 2)}, "cache": {"hit": False}},
            )
            self._logger.info("ask route=need_clarify")
            return resp

        # 1) faq first path
        if decision.route == "faq_first":
            t_faq0 = time.perf_counter()
            hits = self.faq.search(q, top_k=self.settings.faq_top_k)
            t_faq = (time.perf_counter() - t_faq0) * 1000
            if hits and hits[0].score >= float(self.settings.faq_min_score):
                top = hits[0]
                debug["faq_top_score"] = top.score
                ctxs = self.faq.as_contexts(hits[:3])
                ef_faq = filter_evidence(query=q, contexts=ctxs, settings=self.settings)
                ctxs = ef_faq.kept
                citations = self._to_citations(ctxs)
                route_trace.append(RouteTraceItem(step="faq.hit", detail={"faq_id": top.faq_id, "score": top.score}))
                route_trace.append(
                    RouteTraceItem(
                        step="evidence.filter",
                        detail={
                            "enabled": bool(getattr(self.settings, "evidence_filter_enabled", True)),
                            "filtered_out_count": ef_faq.filtered_out_count,
                            "kept_context_count": ef_faq.kept_context_count,
                            "by_reason": ef_faq.by_reason,
                            "relaxed_kept_top1": ef_faq.relaxed_kept_top1,
                        },
                    )
                )
                resp = AnswerResponse(
                    query=q,
                    mode="faq",
                    route="bm25_faq",
                    answer=top.answer or "（该 FAQ 暂无标准答案）",
                    contexts=ctxs,
                    citations=citations,
                    matched_sources=citations,
                    route_trace=route_trace,
                    faq_id=top.faq_id,
                    filtered_out_count=ef_faq.filtered_out_count,
                    kept_context_count=ef_faq.kept_context_count,
                    debug={
                        **debug,
                        "evidence_filter": {
                            "filtered_out_count": ef_faq.filtered_out_count,
                            "kept_context_count": ef_faq.kept_context_count,
                            "by_reason": ef_faq.by_reason,
                            "relaxed_kept_top1": ef_faq.relaxed_kept_top1,
                        },
                        "timing_ms": {"faq": round(t_faq, 2), "total": round((time.perf_counter() - t_total0) * 1000, 2)},
                        "cache": {"hit": False},
                    },
                )
                # 缓存：FAQ 直答优先缓存（但仍遵循频次阈值，避免污染）
                if cache_enabled and self._query_counts.get(q, 0) >= int(getattr(self.settings, "cache_min_hits_to_store", 2)):
                    self._cache.set(cache_key, resp)
                self._logger.info("ask route=bm25_faq faq_id=%s faq_ms=%.2f", top.faq_id, t_faq)
                return resp

        # 2) subquery / HyDE / hybrid fallback
        hk = int(top_k) if top_k is not None else int(self.settings.hybrid_top_k)
        t_h0 = time.perf_counter()

        # subquery：多意图 query 拆分为 2~3 个子查询分别检索，再合并去重+轻量重排
        if decision.route == "subquery":
            t_sq0 = time.perf_counter()
            sq = self.subquery_builder.build(q)
            subqs = (sq.subqueries or [q])[:3]
            t_sq = (time.perf_counter() - t_sq0) * 1000
            route_trace.append(
                RouteTraceItem(
                    step="subquery.split",
                    detail={
                        **(sq.debug or {}),
                        "subqueries": subqs,
                        "split_ms": round(t_sq, 2),
                    },
                )
            )

            merged_contexts = []
            per_retrieval = []
            t_sret0 = time.perf_counter()
            for i, subq in enumerate(subqs, start=1):
                t0 = time.perf_counter()
                r = self.hybrid.retrieve(
                    subq,
                    faq_top_k=int(self.settings.faq_top_k),
                    vec_top_k=int(self.settings.vector_top_k),
                    hybrid_top_k=max(2, hk),
                )
                per_retrieval.append({"i": i, "chars": len(subq), "ms": round((time.perf_counter() - t0) * 1000, 2)})
                merged_contexts.extend(r.contexts)

            # 去重：按 (source, source_id)
            uniq = {}
            for c in merged_contexts:
                uniq[(c.source, c.source_id)] = c
            merged = list(uniq.values())
            reranked = self._light_reranker.rerank(q, merged) if merged else merged

            route_trace.append(
                RouteTraceItem(
                    step="subquery.retrieve",
                    detail={
                        "subqueries_n": len(subqs),
                        "per_subquery": per_retrieval,
                        "before_dedup": len(merged_contexts),
                        "after_dedup": len(merged),
                        "after_rerank": len(reranked),
                        "retrieve_ms": round((time.perf_counter() - t_sret0) * 1000, 2),
                    },
                )
            )

            # 构造一个兼容 HybridResult 的结果，复用后续 backtrack/expand_parent/llm 流程
            result = self.hybrid.retrieve(q, faq_top_k=0, vec_top_k=0, hybrid_top_k=1)
            result = result.__class__(contexts=reranked[:hk], debug={**(result.debug or {}), "subquery": {"enabled": True}})
            t_h = (time.perf_counter() - t_h0) * 1000
            debug.update(result.debug)
        else:
            retrieve_query = q
            if decision.route == "hyde":
                t_hy0 = time.perf_counter()
                hy = self.hyde.generate(q)
                t_hy = (time.perf_counter() - t_hy0) * 1000
                route_trace.append(
                    RouteTraceItem(
                        step="hyde.generate",
                        detail={
                            **(hy.debug or {}),
                            "hyde_ms": round(t_hy, 2),
                            "hyde_used": bool(hy.hyde_text.strip()),
                        },
                    )
                )
                if hy.hyde_text.strip():
                    retrieve_query = hy.hyde_text.strip()

            route_trace.append(
                RouteTraceItem(
                    step="hyde.retrieve" if decision.route == "hyde" else "retriever.query",
                    detail={
                        "query_used": "hyde_text" if (decision.route == "hyde" and retrieve_query != q) else "original_query",
                        "query_chars": len(retrieve_query),
                    },
                )
            )

            result = self.hybrid.retrieve(
                retrieve_query,
                faq_top_k=int(self.settings.faq_top_k),
                vec_top_k=int(self.settings.vector_top_k),
                hybrid_top_k=hk,
            )
            t_h = (time.perf_counter() - t_h0) * 1000
            debug.update(result.debug)

            # HyDE 场景：用原始 query 再做一次轻量重排（避免“HyDE 文本”偏离用户真实关注点）
            if decision.route == "hyde" and result.contexts:
                t_hr0 = time.perf_counter()
                reranked = self._light_reranker.rerank(q, result.contexts)
                t_hr = (time.perf_counter() - t_hr0) * 1000
                result = result.__class__(contexts=reranked[:hk], debug=result.debug)
                route_trace.append(RouteTraceItem(step="hyde.rerank", detail={"type": "KeywordOverlapReranker", "rerank_ms": round(t_hr, 2)}))
        route_trace.append(
            RouteTraceItem(
                step="retriever.hybrid",
                detail={
                    "faq_top_k": int(self.settings.faq_top_k),
                    "vector_top_k": int(self.settings.vector_top_k),
                    "hybrid_top_k": hk,
                    "rerank": result.debug.get("rerank"),
                },
            )
        )

        # backtrack：首轮结果质量弱时，补召回（parent + 相邻 child）
        did_backtrack = False
        if bool(getattr(self.settings, "backtrack_enabled", True)) and self._is_weak_retrieval(result.contexts):
            did_backtrack = True
            t_bt0 = time.perf_counter()
            expanded = self._backtrack_expand_contexts(q, result.contexts)
            t_bt = (time.perf_counter() - t_bt0) * 1000
            route_trace.append(
                RouteTraceItem(
                    step="backtrack",
                    detail={
                        "triggered": True,
                        "min_top_score": float(getattr(self.settings, "backtrack_min_top_score", 0.0)),
                        "before": len(result.contexts),
                        "after": len(expanded),
                        "backtrack_ms": round(t_bt, 2),
                    },
                )
            )
            # 轻量重排 + 截断，保持返回结构兼容
            reranked = self._light_reranker.rerank(q, expanded)
            result = result.__class__(contexts=reranked[:hk], debug=result.debug)
            citations = self._to_citations(result.contexts)

        ef = filter_evidence(query=q, contexts=result.contexts, settings=self.settings)
        result = result.__class__(contexts=ef.kept, debug=result.debug)
        citations = self._to_citations(result.contexts)
        # 兼容旧逻辑：debug 仍保留一份（但推荐前端读 AnswerResponse.citations）
        debug["citations"] = [
            {"source": c.source, "id": c.source_id, "score": c.score, "meta": c.metadata} for c in result.contexts
        ]
        ef_debug = {
            "filtered_out_count": ef.filtered_out_count,
            "kept_context_count": ef.kept_context_count,
            "by_reason": ef.by_reason,
            "relaxed_kept_top1": ef.relaxed_kept_top1,
        }
        debug["evidence_filter"] = ef_debug
        route_trace.append(
            RouteTraceItem(
                step="evidence.filter",
                detail={
                    "enabled": bool(getattr(self.settings, "evidence_filter_enabled", True)),
                    **ef_debug,
                },
            )
        )

        # 生成阶段：child 命中后按 parent_id 回溯 parent 原文上下文（去重 + 长度控制）
        t_ctx0 = time.perf_counter()
        gen_contexts = self._build_generation_contexts(result.contexts, max_parents=6, max_total_chars=4500)
        t_ctx = (time.perf_counter() - t_ctx0) * 1000
        route_trace.append(
            RouteTraceItem(
                step="context.expand_parent",
                detail={
                    "retrieved_contexts": len(result.contexts),
                    "generation_contexts": len(gen_contexts),
                    "expand_ms": round(t_ctx, 2),
                },
            )
        )

        t_llm0 = time.perf_counter()
        llm_result = self.llm.generate(q, gen_contexts)
        t_llm = (time.perf_counter() - t_llm0) * 1000
        if llm_result is not None and llm_result.answer:
            debug["llm_provider"] = self.settings.llm_provider
            debug["llm_model"] = llm_result.model
            route_trace.append(RouteTraceItem(step="llm.generate", detail={"provider": self.settings.llm_provider, "model": llm_result.model, "llm_ms": round(t_llm, 2)}))
            resp = AnswerResponse(
                query=q,
                mode="rag_mock",
                route=("subquery" if decision.route == "subquery" else ("hyde" if decision.route == "hyde" else "rag_standard")),
                answer=llm_result.answer,
                contexts=result.contexts,
                citations=citations,
                matched_sources=citations,
                route_trace=route_trace,
                filtered_out_count=ef.filtered_out_count,
                kept_context_count=ef.kept_context_count,
                debug={
                    **debug,
                    "timing_ms": {
                        "hybrid_total": round(t_h, 2),
                        "faq": debug.get("timing_ms", {}).get("faq") if isinstance(debug.get("timing_ms"), dict) else None,
                        "vector": debug.get("timing_ms", {}).get("vector") if isinstance(debug.get("timing_ms"), dict) else None,
                        "rerank": debug.get("timing_ms", {}).get("rerank") if isinstance(debug.get("timing_ms"), dict) else None,
                        "llm": round(t_llm, 2),
                        "total": round((time.perf_counter() - t_total0) * 1000, 2),
                    },
                    "cache": {"hit": False},
                },
            )
            # 缓存：高频 query 结果缓存
            if cache_enabled and self._query_counts.get(q, 0) >= int(getattr(self.settings, "cache_min_hits_to_store", 2)):
                self._cache.set(cache_key, resp)
            self._logger.info(
                "ask route=rag_standard hybrid_ms=%.2f llm_ms=%.2f backtrack=%s",
                t_h,
                t_llm,
                did_backtrack,
            )
            return resp

        # 安全 fallback：未配置 LLM 时仍可运行
        debug["llm_provider"] = "disabled"
        answer = self._fallback_answer(q, result.contexts)
        route_trace.append(RouteTraceItem(step="llm.fallback", detail={"provider": "disabled", "llm_ms": round(t_llm, 2)}))
        resp = AnswerResponse(
            query=q,
            mode="rag_mock",
            route=("subquery" if decision.route == "subquery" else ("hyde" if decision.route == "hyde" else "rag_standard")),
            answer=answer,
            contexts=result.contexts,
            citations=citations,
            matched_sources=citations,
            route_trace=route_trace,
            filtered_out_count=ef.filtered_out_count,
            kept_context_count=ef.kept_context_count,
            debug={
                **debug,
                "timing_ms": {
                    "hybrid_total": round(t_h, 2),
                    "llm": round(t_llm, 2),
                    "total": round((time.perf_counter() - t_total0) * 1000, 2),
                },
                "cache": {"hit": False},
            },
        )
        if cache_enabled and self._query_counts.get(q, 0) >= int(getattr(self.settings, "cache_min_hits_to_store", 2)):
            self._cache.set(cache_key, resp)
        self._logger.info("ask route=rag_standard fallback backtrack=%s", did_backtrack)
        return resp

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, object]] = None, request_id: Optional[str] = None) -> SearchResponse:
        """
        仅检索证据，不调用 LLM，不走 /ask 生成链路。
        """
        _ = filters or {}
        _ = request_id
        t_total0 = time.perf_counter()
        deadline = t_total0 + 5.0
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        errors: List[str] = []
        hybrid_skipped = False
        fallback_inserted = False
        fast_path_used = False
        real_retrieval_used = False
        fallback_only = False
        fallback_reason: Optional[str] = None
        real_hit_count = 0
        real_top1_source_id: Optional[str] = None
        real_top1_relevance_reason: Optional[str] = None
        top_hit_reason = "no_hits"
        rerank_debug: Dict[str, Any] = {"enabled": False}
        fallback_insert_position: Optional[int] = None
        final_top1_source_id: Optional[str] = None
        final_top1_is_fallback: Optional[bool] = None

        original_query = query

        # SEARCH_MODE：默认 lightweight（不允许加载向量/embedding）
        settings_mode = str(getattr(self.settings, "search_mode", "lightweight") or "lightweight").lower()
        include_vector = bool((_ or {}).get("include_vector")) if isinstance(_, dict) else False
        search_mode = "hybrid" if (settings_mode == "hybrid" or include_vector) else "lightweight"
        vector_used = False
        model_loaded_before = bool(getattr(self.vec, "_model", None) is not None)

        t0 = time.perf_counter()
        qp = self.query_processor.process(query)
        timings["query_process"] = round((time.perf_counter() - t0) * 1000, 2)
        cleaned = qp.cleaned_query
        if not cleaned:
            return SearchResponse(
                hits=[],
                query=query,
                route_trace=["query_clean"],
                debug={
                    "top_k": max(1, int(top_k)),
                    "retriever": "local/mock",
                    "reason": "empty_query",
                    "timing_ms": {**timings, "total": round((time.perf_counter() - t_total0) * 1000, 2)},
                },
            )

        t0 = time.perf_counter()
        detected_error_type = self._detect_error_type(cleaned)
        timings["detect_error_type"] = round((time.perf_counter() - t0) * 1000, 2)
        expanded_terms = self._expand_error_terms(cleaned, detected_error_type)
        normalized_query = self._build_search_query(cleaned, expanded_terms)

        t0 = time.perf_counter()
        decision = self.router.decide(normalized_query, query_type=qp.query_type)
        timings["route"] = round((time.perf_counter() - t0) * 1000, 2)
        route_trace = ["query_clean", "retrieve", "rerank"]
        hk = max(1, int(top_k))
        route_tag = "rag_standard"
        contexts: List[RetrievedContext] = []
        ef_filtered_out = 0
        retriever_name = f"bm25+{self.settings.vector_store_backend}"
        vector_optional = bool(detected_error_type)
        resource_state = self.ready_status()
        faq_bm25_ready = bool(resource_state.get("faq_ready") and resource_state.get("bm25_ready"))
        lightweight_ready = faq_bm25_ready

        if detected_error_type:
            fast_path_used = True
            warnings.append("stage skipped: vector optional for error_type")

        # lightweight 模式下 /search 不允许触发 full init（会构建向量索引并可能加载 embedding）
        if search_mode == "hybrid" and (not self._ready) and (not detected_error_type):
            ask_init_limit = self._pick_ask_init_limit()
            try:
                self._ensure_initialized(limit=ask_init_limit)
            except Exception as e:
                return SearchResponse(
                    hits=[],
                    query=query,
                    route_trace=route_trace,
                    debug={
                        "top_k": hk,
                        "retriever": "init_failed",
                        "warnings": warnings + ["fallback used: init_failed"],
                        "errors": [f"init_failed:{e!r}"],
                        "detected_error_type": detected_error_type,
                        "timing_ms": self._finalize_timeout_timings(timings, t_total0),
                    },
                )

        # error_type 快速路径：
        # - 若 FAQ/BM25 ready：优先真实 FAQ/BM25（phrase match + boost），不足再 fallback
        # - 若 FAQ/BM25 not ready：直接 fallback（不阻塞）
        if detected_error_type and not faq_bm25_ready:
            warnings.append("fallback used: faq/bm25 not_ready")
            fallback_only = True
            fallback_reason = "faq_bm25_not_ready"
            timings["faq_search"] = 0.0
            timings["phrase_match"] = 0.0
            timings["hybrid_retrieve"] = 0.0
            timings["rerank_boost"] = 0.0
            t_fb0 = time.perf_counter()
            fallback_hit = self._build_error_fallback_hit(detected_error_type, route="error_fast_path")
            fallback_inserted = True
            timings["fallback_build"] = round((time.perf_counter() - t_fb0) * 1000, 2)
            timings = self._finalize_timeout_timings(timings, t_total0)
            return SearchResponse(
                hits=[fallback_hit],
                query=query,
                route_trace=route_trace,
                debug={
                    "top_k": hk,
                    "retriever": "fallback_only",
                    "filtered_out_count": 0,
                    "request_query_type": qp.query_type,
                    "original_query": original_query,
                    "normalized_query": normalized_query,
                    "detected_error_type": detected_error_type,
                    "expanded_terms": expanded_terms,
                    "search_mode": search_mode,
                    "resource_state": resource_state,
                    "real_retrieval_used": False,
                    "real_hit_count": 0,
                    "real_top1_source_id": None,
                    "real_top1_relevance_reason": None,
                    "fallback_only": True,
                    "fallback_reason": fallback_reason,
                    "rerank_boost_applied": {"enabled": False, "reason": "fallback_only"},
                    "top_hit_reason": "fallback_not_ready",
                    "fast_path_used": True,
                    "vector_optional": True,
                    "vector_used": False,
                    "model_loaded_this_request": False,
                    "hybrid_skipped": True,
                    "fallback_inserted": fallback_inserted,
                    "fallback_insert_position": 0,
                    "final_top1_source_id": fallback_hit.source_id,
                    "final_top1_is_fallback": True,
                    "warnings": warnings,
                    "errors": errors,
                    "timing_ms": timings,
                    "llm_called": False,
                },
            )

        t0 = time.perf_counter()
        faq_hits = self.faq.search(normalized_query, top_k=max(int(self.settings.faq_top_k), hk)) if faq_bm25_ready else []
        timings["faq_search"] = round((time.perf_counter() - t0) * 1000, 2)
        if faq_hits:
            contexts = self.faq.as_contexts(faq_hits[:hk])
            route_tag = "bm25_faq"
            real_retrieval_used = True
            real_hit_count = len(faq_hits[:hk])

        # error_type：补充 phrase match（不依赖中文分词），保证英文异常名可命中 seed/真实 FAQ
        t0 = time.perf_counter()
        if detected_error_type and faq_bm25_ready:
            pm_ctxs = self._phrase_match_faq(detected_error_type, expanded_terms, top_k=hk)
            if pm_ctxs:
                # phrase match 结果优先（再拼接 BM25 命中，去重）
                uniq: Dict[str, RetrievedContext] = {}
                for c in [*pm_ctxs, *contexts]:
                    uniq[str(c.source_id)] = c
                contexts = list(uniq.values())[:hk]
                route_tag = "bm25_faq"
                real_retrieval_used = True
                real_hit_count = max(real_hit_count, len(pm_ctxs))
        timings["phrase_match"] = round((time.perf_counter() - t0) * 1000, 2)

        if detected_error_type and len(contexts) >= hk:
            fast_path_used = True
            hybrid_skipped = True
            warnings.append("stage skipped: hybrid by error_type fast path")

        if not hybrid_skipped:
            # 错误类型 query 的优先快路径：FAQ 有结果时不强制进入向量检索
            if detected_error_type and contexts:
                fast_path_used = True
                hybrid_skipped = True
                warnings.append("stage skipped: hybrid with faq hits")
            else:
                if search_mode == "lightweight":
                    hybrid_skipped = True
                    warnings.append("stage skipped: hybrid disabled by search_mode=lightweight")
                else:
                    retrieve_query = normalized_query
                    if decision.route == "hyde":
                        hy = self.hyde.generate(normalized_query)
                        if hy.hyde_text.strip():
                            retrieve_query = hy.hyde_text.strip()
                    t0 = time.perf_counter()
                    try:
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("search total timeout before hybrid")
                        vector_used = True
                        result = self.hybrid.retrieve(
                            retrieve_query,
                            faq_top_k=int(self.settings.faq_top_k),
                            vec_top_k=int(self.settings.vector_top_k),
                            hybrid_top_k=hk,
                        )
                        hybrid_ms = (time.perf_counter() - t0) * 1000
                        timings["hybrid_retrieve"] = round(hybrid_ms, 2)
                        if hybrid_ms > 3000:
                            hybrid_skipped = True
                            warnings.append("stage timeout: hybrid_retrieve over 3s")
                            warnings.append("fallback used: keep existing hits")
                        else:
                            contexts = (contexts + (result.contexts if result else []))[
                                : max(hk, int(self.settings.hybrid_top_k))
                            ]
                            route_tag = "rag_standard"
                            rerank_type = (((result.debug if result else {}) or {}).get("rerank") or {}).get("type")
                            if rerank_type:
                                retriever_name = f"{retriever_name}/{rerank_type}"
                        if time.perf_counter() >= deadline:
                            warnings.append("stage timeout: total timeout guard after hybrid")
                    except Exception as e:
                        timings["hybrid_retrieve"] = round((time.perf_counter() - t0) * 1000, 2)
                        warnings.append("hybrid_exception")
                        errors.append(f"hybrid_exception:{e!r}")
                        self._logger.warning("search hybrid exception: %r", e)
        else:
            timings["hybrid_retrieve"] = 0.0

        if time.perf_counter() >= deadline:
            warnings.append("stage timeout: timeout_guard")
            warnings.append("skipped stage: filter/rerank partial")
        t0 = time.perf_counter()
        ef = filter_evidence(query=normalized_query, contexts=contexts[:hk], settings=self.settings)
        timings["filter_evidence"] = round((time.perf_counter() - t0) * 1000, 2)
        ef_filtered_out = ef.filtered_out_count
        hits = self._to_search_hits(ef.kept[:hk], route=route_tag)

        t0 = time.perf_counter()
        reranked_hits, rerank_debug, top_hit_reason = self._rerank_search_hits(
            hits, detected_error_type=detected_error_type, expanded_terms=expanded_terms
        )
        timings["rerank_boost"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        # 记录真实 top1（fallback 之前）
        if reranked_hits:
            real_top1_source_id = reranked_hits[0].source_id
            real_top1_relevance_reason = self._relevance_reason_for_hit(
                reranked_hits[0], detected_error_type=detected_error_type, expanded_terms=expanded_terms
            )

        final_hits, inserted, fallback_insert_position, fallback_reason2 = self._ensure_error_fallback_hits(
            reranked_hits,
            detected_error_type=detected_error_type,
            expanded_terms=expanded_terms,
            top_k=hk,
            route=route_tag,
        )
        fallback_inserted = inserted
        if fallback_reason2:
            fallback_reason = fallback_reason2
        if inserted:
            warnings.append("fallback used: inserted_error_evidence")
            if fallback_insert_position == 0:
                fallback_only = (real_hit_count == 0)
        timings["fallback_build"] = round((time.perf_counter() - t0) * 1000, 2)
        timings = self._finalize_timeout_timings(timings, t_total0)

        if final_hits:
            final_top1_source_id = final_hits[0].source_id
            final_top1_is_fallback = bool(final_hits[0].metadata.get("fallback")) or (final_hits[0].source_type == "fallback_error_guide")
        model_loaded_after = bool(getattr(self.vec, "_model", None) is not None)
        model_loaded_this_request = (not model_loaded_before) and model_loaded_after

        self._logger.info(
            "search timing_ms=%s error_type=%s fast_path=%s hybrid_skipped=%s fallback=%s",
            timings,
            detected_error_type,
            fast_path_used,
            hybrid_skipped,
            fallback_inserted,
        )

        return SearchResponse(
            hits=final_hits[:hk],
            query=query,
            route_trace=route_trace,
            debug={
                "top_k": hk,
                "retriever": retriever_name,
                "filtered_out_count": ef_filtered_out,
                "request_query_type": qp.query_type,
                "original_query": original_query,
                "normalized_query": normalized_query,
                "detected_error_type": detected_error_type,
                "expanded_terms": expanded_terms,
                "search_mode": search_mode,
                "rerank_boost_applied": rerank_debug,
                "top_hit_reason": top_hit_reason,
                "fast_path_used": fast_path_used,
                "vector_optional": vector_optional,
                "hybrid_skipped": hybrid_skipped,
                "fallback_inserted": fallback_inserted,
                "resource_state": resource_state,
                "real_retrieval_used": real_retrieval_used,
                "real_hit_count": real_hit_count,
                "real_top1_source_id": real_top1_source_id,
                "real_top1_relevance_reason": real_top1_relevance_reason,
                "fallback_only": fallback_only,
                "fallback_reason": fallback_reason,
                "fallback_insert_position": fallback_insert_position,
                "final_top1_source_id": final_top1_source_id,
                "final_top1_is_fallback": final_top1_is_fallback,
                "vector_used": vector_used,
                "model_loaded_this_request": model_loaded_this_request,
                "warnings": warnings,
                "errors": errors,
                "timing_ms": timings,
                "llm_called": False,
            },
        )

    def _relevance_reason_for_hit(self, hit: SearchHit, *, detected_error_type: Optional[str], expanded_terms: List[str]) -> str:
        blob = f"{hit.title} {hit.snippet}".lower()
        if detected_error_type and detected_error_type.lower() in blob:
            return "match_error_type"
        for t in expanded_terms:
            if t and str(t).lower() in blob:
                return f"match_term:{t}"
        return "no_term_match"

    def _finalize_timeout_timings(self, timings: Dict[str, float], t_total0: float) -> Dict[str, float]:
        merged = dict(timings)
        merged["timeout_guard"] = round(max(0.0, 5000.0 - ((time.perf_counter() - t_total0) * 1000)), 2)
        merged["total"] = round((time.perf_counter() - t_total0) * 1000, 2)
        # 统一补齐关键字段，便于前端稳定读取
        for k in ("detect_error_type", "fallback_build", "faq_search", "phrase_match", "rerank_boost", "hybrid_retrieve"):
            merged.setdefault(k, 0.0)
        return merged

    def _is_weak_retrieval(self, contexts: List[RetrievedContext]) -> bool:
        if not contexts:
            return True
        top_score = float(contexts[0].score)
        threshold = float(getattr(self.settings, "backtrack_min_top_score", 0.0))
        return top_score < threshold

    def _backtrack_expand_contexts(self, query: str, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        """
        最小 backtrack：
        - 对 top 若干 child chunk，补充：
          - parent chunk 原文（用于补召回）
          - 相邻 child chunk（c{i-1}, c{i+1}）
        - 去重：按 (source, source_id)
        """
        window = max(0, int(getattr(self.settings, "backtrack_neighbor_window", 1)))
        max_extra = max(1, int(getattr(self.settings, "backtrack_max_extra_contexts", 6)))

        uniq: Dict[tuple, RetrievedContext] = {(c.source, c.source_id): c for c in contexts}
        extra_added = 0

        def add_ctx(rc: RetrievedContext) -> None:
            nonlocal extra_added
            key = (rc.source, rc.source_id)
            if key in uniq:
                return
            uniq[key] = rc
            extra_added += 1

        for c in contexts:
            if extra_added >= max_extra:
                break
            if c.source != "chunk":
                continue
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            if str(meta.get("chunk_level") or "") != "child":
                continue

            parent_id = meta.get("parent_id")
            if parent_id and str(parent_id) in self._parent_chunks and extra_added < max_extra:
                p = self._parent_chunks[str(parent_id)]
                add_ctx(
                    RetrievedContext(
                        source="chunk",
                        source_id=p.chunk_id,
                        score=float(c.score) * 0.98,
                        text=(p.text or ""),
                        metadata={**{k: v for k, v in (p.metadata or {}).items()}, "doc_id": p.doc_id, "chunk_level": "parent"},
                    )
                )

            # 相邻 child：依据 chunk_id 形如 doc::c123
            cid = c.source_id
            m = None
            try:
                m = __import__("re").match(r"^(?P<doc>.+)::c(?P<i>\d+)$", str(cid))
            except Exception:
                m = None
            if not m:
                continue
            doc = m.group("doc")
            i = int(m.group("i"))
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                nid = f"{doc}::c{i + delta}"
                if nid in self._chunks_by_id and extra_added < max_extra:
                    nb = self._chunks_by_id[nid]
                    add_ctx(
                        RetrievedContext(
                            source="chunk",
                            source_id=nb.chunk_id,
                            score=float(c.score) * 0.95,
                            text=(nb.text or ""),
                            metadata={**{k: v for k, v in (nb.metadata or {}).items()}, "doc_id": nb.doc_id},
                        )
                    )

        return list(uniq.values())

    def _build_generation_contexts(
        self, retrieved_contexts: List[RetrievedContext], *, max_parents: int, max_total_chars: int
    ) -> List[RetrievedContext]:
        """
        正式父子块 RAG：
        - 检索阶段命中 child chunk（retrieved_contexts）
        - 生成阶段根据 child.metadata.parent_id 回溯 parent chunk 原文
        - 去重：同一 parent 只拼一次
        - 长度控制：限制总字符数，避免上下文膨胀

        兼容说明：
        - FAQ contexts 原样保留
        - 找不到 parent 时回退到原 child text
        """

        out: List[RetrievedContext] = []
        used_parent_ids = set()
        total_chars = 0
        parents_added = 0

        for c in retrieved_contexts:
            if c.source == "faq":
                t = (c.text or "").strip()
                if not t:
                    continue
                if total_chars + len(t) > max_total_chars:
                    continue
                out.append(c)
                total_chars += len(t)
                continue

            meta = c.metadata if isinstance(c.metadata, dict) else {}
            parent_id = meta.get("parent_id") or meta.get("parentId")
            parent_id = str(parent_id) if parent_id else None

            # 仅对 child chunk 做 parent 回溯；如果已经是 parent 或没有 parent_id，则直接用原文
            is_child = str(meta.get("chunk_level") or "") == "child"
            if is_child and parent_id and parent_id in self._parent_chunks and parent_id not in used_parent_ids:
                if parents_added >= max_parents:
                    continue
                parent = self._parent_chunks[parent_id]
                p_text = (parent.text or "").strip()
                if not p_text:
                    continue
                if total_chars + len(p_text) > max_total_chars:
                    continue
                used_parent_ids.add(parent_id)
                parents_added += 1
                out.append(
                    RetrievedContext(
                        source="chunk",
                        source_id=parent.chunk_id,
                        score=float(c.score),
                        text=p_text,
                        metadata={
                            **{k: v for k, v in (parent.metadata or {}).items()},
                            "doc_id": parent.doc_id,
                            "chunk_level": "parent",
                        },
                    )
                )
                total_chars += len(p_text)
                continue

            # fallback：直接使用 child（避免生成无上下文）
            t = (c.text or "").strip()
            if not t:
                continue
            if total_chars + len(t) > max_total_chars:
                continue
            out.append(c)
            total_chars += len(t)

        return out

    def _to_citations(self, contexts: List[RetrievedContext]) -> List[Citation]:
        out: List[Citation] = []
        for c in contexts:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            source_type = str(meta.get("source") or c.source)
            title = str(meta.get("title") or "")
            parent_id = meta.get("parent_id") or meta.get("parentId") or None
            snippet = (c.text or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            out.append(
                Citation(
                    source_id=c.source_id,
                    source_type=source_type,
                    title=title,
                    score=float(c.score),
                    parent_id=str(parent_id) if parent_id else None,
                    snippet=snippet,
                    metadata={k: v for k, v in meta.items()},
                )
            )
        return out

    def _to_search_hits(self, contexts: List[RetrievedContext], *, route: str) -> List[SearchHit]:
        out: List[SearchHit] = []
        for c in contexts:
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            source_type = str(meta.get("source") or c.source)
            title = str(meta.get("title") or c.source_id)
            snippet = (c.text or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            out.append(
                SearchHit(
                    source_id=c.source_id,
                    title=title,
                    snippet=snippet,
                    score=float(c.score),
                    source_type=source_type,
                    metadata={**{k: v for k, v in meta.items()}, "route": route},
                )
            )
        return out

    def _detect_error_type(self, text: str) -> Optional[str]:
        if not text:
            return None
        found = self._ERROR_TYPE_RE.findall(text)
        if not found:
            return None
        normalized = {x.lower(): x for x in self._ERROR_TYPE_ORDER}
        for m in found:
            key = str(m).lower()
            if key in normalized:
                return normalized[key]
        return None

    def _expand_error_terms(self, query: str, detected_error_type: Optional[str]) -> List[str]:
        terms: List[str] = []
        if detected_error_type:
            terms.append(detected_error_type)
            terms.extend(self._ERROR_TYPE_SYNONYMS.get(detected_error_type, []))
        # 保底补充常见代码报错类术语，利于 faq/vector 双路召回
        for t in ["报错排查", "课堂演示", "代码调试"]:
            if t not in query:
                terms.append(t)
        # 去重并保持顺序
        out: List[str] = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _build_search_query(self, cleaned_query: str, expanded_terms: List[str]) -> str:
        if not expanded_terms:
            return cleaned_query
        joined = " ".join(expanded_terms)
        return f"{cleaned_query} {joined}".strip()

    def _phrase_match_faq(self, error_type: str, expanded_terms: List[str], *, top_k: int) -> List[RetrievedContext]:
        """
        对 FAQ 做子串匹配（phrase match），避免中文分词导致英文异常名召回失败。
        """
        docs = getattr(self.faq, "_faq_docs", []) or []
        if not docs:
            return []
        terms = [error_type, *expanded_terms]
        lowered = [str(t).lower() for t in terms if t]
        matched: List[Tuple[float, RetrievedContext]] = []
        for d in docs:
            meta = d.metadata if isinstance(d.metadata, dict) else {}
            q = str(meta.get("question") or d.title or "")
            a = str(meta.get("answer") or "")
            blob = f"{q}\n{a}\n{d.text}".lower()
            m = sum(1 for t in lowered if t and t in blob)
            if m <= 0:
                continue
            score = 100.0 + 5.0 * m
            text = f"FAQ 问：{q}\nFAQ 答：{a}"
            matched.append(
                (
                    score,
                    RetrievedContext(
                        source="faq",
                        source_id=str(d.doc_id),
                        score=float(score),
                        text=text,
                        metadata={"category": str(meta.get("category") or "")},
                    ),
                )
            )
        matched.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in matched[: max(1, int(top_k))]]

    def _rerank_search_hits(
        self, hits: List[SearchHit], *, detected_error_type: Optional[str], expanded_terms: List[str]
    ) -> tuple[List[SearchHit], Dict[str, object], str]:
        if not hits:
            return hits, {"enabled": False, "reason": "no_hits"}, "no_hits"

        term_set = {t.lower() for t in expanded_terms if t}
        if detected_error_type:
            term_set.add(detected_error_type.lower())

        boosted: List[tuple[float, SearchHit, List[str]]] = []
        for h in hits:
            base = float(h.score)
            reasons: List[str] = []
            text_blob = " ".join(
                [
                    str(h.title or ""),
                    str(h.snippet or ""),
                    " ".join([f"{k}:{v}" for k, v in (h.metadata or {}).items()]),
                ]
            ).lower()
            bonus = 0.0

            if detected_error_type and detected_error_type.lower() in text_blob:
                bonus += 1.5
                reasons.append("match_error_type")

            matched_terms = [t for t in term_set if t and t in text_blob]
            if matched_terms:
                bonus += min(1.0, 0.2 * len(matched_terms))
                reasons.append(f"match_terms:{len(matched_terms)}")

            category = str((h.metadata or {}).get("category") or "")
            if category == "代码报错":
                bonus += 1.2
                reasons.append("category_code_error")
                if h.source_type == "faq":
                    bonus += 0.8
                    reasons.append("faq_code_error")

            if any(x in text_blob for x in ["变量未定义", "函数未定义", "nameerror"]):
                bonus += 1.1
                reasons.append("nameerror_keyword")

            boosted.append((base + bonus, h, reasons))

        boosted.sort(key=lambda x: x[0], reverse=True)
        reranked_hits = [x[1] for x in boosted]
        top_reasons = boosted[0][2] if boosted else []
        debug = {
            "enabled": True,
            "detected_error_type": detected_error_type,
            "boosted_hits": [
                {"source_id": h.source_id, "boosted_score": round(score, 4), "reasons": reasons}
                for score, h, reasons in boosted[:5]
            ],
        }
        return reranked_hits, debug, ",".join(top_reasons) if top_reasons else "base_score"

    def _ensure_error_fallback_hits(
        self,
        hits: List[SearchHit],
        *,
        detected_error_type: Optional[str],
        expanded_terms: List[str],
        top_k: int,
        route: str,
    ) -> Tuple[List[SearchHit], bool, Optional[int], Optional[str]]:
        """
        返回：
        - final_hits
        - fallback_inserted
        - fallback_insert_position
        - fallback_reason
        """
        if not detected_error_type:
            return hits, False, None, None

        # 先按“是否包含 error_type/扩展词”把真实命中提到前面，避免 fallback 覆盖真实相关 FAQ
        terms = [detected_error_type, *expanded_terms]
        lowered = [str(t).lower() for t in terms if t]

        def is_relevant(h: SearchHit) -> bool:
            blob = f"{h.title} {h.snippet}".lower()
            return any(t and t in blob for t in lowered)

        relevant = [h for h in hits if is_relevant(h)]
        irrelevant = [h for h in hits if not is_relevant(h)]
        reordered = [*relevant, *irrelevant]

        # 决定是否插入 fallback
        if relevant:
            # 已有真实相关命中：fallback 只做补充，放到末尾（不抢 top1）
            if len(reordered) >= max(1, top_k):
                return reordered[: max(1, top_k)], False, None, None
            fallback = self._build_error_fallback_hit(detected_error_type, route=route)
            pos = min(len(reordered), max(1, top_k) - 1)
            out = [*reordered]
            out.insert(pos, fallback)
            return out[: max(1, top_k)], True, pos, "supplement_after_relevant_real_hit"

        # 没有任何相关真实命中：fallback 可作为 top1
        fallback = self._build_error_fallback_hit(detected_error_type, route=route)
        out = [fallback, *reordered]
        return out[: max(1, top_k)], True, 0, "real_hits_irrelevant_or_empty"

    def _build_error_fallback_hit(self, error_type: str, *, route: str) -> SearchHit:
        if error_type == "NameError":
            snippet = "NameError 通常表示变量或函数名在使用前未定义，也可能是大小写、拼写或作用域问题。课堂上可以让学生先定位报错行，再检查变量是否先定义后使用。"
        elif error_type == "TypeError":
            snippet = "TypeError 常见于参数类型不匹配。建议先确认函数入参类型，再检查字符串与数值混用、None 参与运算等场景。"
        elif error_type == "SyntaxError":
            snippet = "SyntaxError 表示语法错误，常见于冒号缺失、括号不配对、缩进错误。可让学生从报错行向上两三行回看结构。"
        else:
            snippet = f"{error_type} 排查建议：先阅读完整报错，再核对触发代码上下文与数据输入。"
        return SearchHit(
            source_id=f"FALLBACK-{error_type}",
            title=f"{error_type} 常见原因与课堂解释",
            snippet=snippet,
            score=0.99,
            source_type="fallback_error_guide",
            metadata={
                "category": "代码报错",
                "route": "fallback_error_guide",
                "fallback": True,
                "detected_error_type": error_type,
            },
        )

    def _fallback_answer(self, query: str, contexts: List[RetrievedContext]) -> str:
        """
        当未配置 LLM 时的安全兜底答案（仍保留“检索->回答”链路的可解释性）。

        TODO:
        - 增加安全与风格约束（适配教师场景）
        """
        if not contexts:
            return (
                "**结论**\n"
                "目前知识库中没有检索到足够相关的证据来支撑确定回答。\n\n"
                "**建议你补充的信息（任选其一即可）**\n"
                "- 你使用的是教师端还是学生端？具体在哪个模块/页面（如 作业批改台 / 练习包下发 / 教师工作台 / 在线编程环境）？\n"
                "- 如果是报错类问题：请粘贴完整 Traceback，或至少给出错误类型 + 触发的那一行代码（前后 5 行）。\n"
                "- 如果是课程/讲法类问题：请说明课程阶段（Scratch / Python基础 / 数据处理入门 等）与课堂目标。\n"
            )

        # 生成更像“产品输出”的兜底：先给结论，再给可执行步骤，最后附依据来源（片段节选）
        top_snippets: List[str] = []
        basis: List[str] = []
        for i, c in enumerate(contexts[:3], start=1):
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            src = str(meta.get("source") or c.source)
            title = str(meta.get("title") or "")
            sid = str(c.source_id or "")
            snip = (c.text or "").strip().replace("\n", " ")
            if len(snip) > 220:
                snip = snip[:220] + "…"
            top_snippets.append(f"- 片段{i}：{snip}")
            basis.append(f"- E{i}: type={src} id={sid}" + (f" title={title}" if title else ""))

        return (
            "**结论**\n"
            "我已基于当前检索到的证据给出一个可执行的回答。注意：当前未连接可用 LLM（或调用失败），以下为安全兜底模板输出。\n\n"
            f"**问题**\n{query}\n\n"
            "**建议步骤**\n"
            "1) 先判断问题类型：平台入口/操作、代码报错、课程设计/讲法、Python语法。\n"
            "2) 平台入口/路径类：优先按模块名在教师端导航中查找；若疑似改版，关注“入口调整/版本差异/新版路径”。\n"
            "3) 代码报错类：补充完整报错（Traceback）与前后 5 行代码，先复现再定位触发条件。\n"
            "4) 课程设计/讲法类：明确课程阶段与课堂目标（导入/过渡/互动/讲评），再选择最小可执行案例组织讲解。\n\n"
            "**依据来源（节选）**\n"
            + "\n".join(basis)
            + "\n\n"
            + "\n".join(top_snippets)
            + "\n"
        )

    def _build_clarifications(self, query: str) -> List[str]:
        """
        最小澄清问题集：不做分类器，仅根据常见缺失信息给出可执行的补充项。
        """
        qs: List[str] = []
        if any(x in query for x in ["报错", "错误", "异常", "失败", "闪退"]):
            qs.extend(
                [
                    "请粘贴完整报错信息（Traceback）或说明错误类型（如 NameError/TypeError）。",
                    "请贴出相关代码（至少前后 5 行）或说明发生在哪一行/哪一步。",
                    "发生在教师端还是学生端？使用的是哪个模块（如 作业批改台/教师工作台/在线编程环境）？",
                ]
            )
        else:
            qs.extend(
                [
                    "你指的“这个/那个”具体是哪个功能或页面？（例如 作业发布入口 / 登录 / 课件投屏 等）",
                    "是哪个课程阶段/模块？（例如 Scratch图形化 / Python基础 / 作业批改台）",
                    "你希望得到：操作步骤、原因解释，还是课堂讲法（面向学生的讲解）？",
                ]
            )
        # 去重保持顺序
        out: List[str] = []
        seen = set()
        for x in qs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

