from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional

from app.config import Settings
from app.schemas.answer import AnswerResponse, Citation, RouteTraceItem
from app.schemas.document import DocumentChunk, InternalDocument
from app.schemas.query import RetrievedContext
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

    def _pick_ask_init_limit(self) -> Optional[int]:
        ask_limit = int(getattr(self.settings, "ask_init_limit", 0) or 0)
        return ask_limit if ask_limit > 0 else None

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
            self.init_kb(limit=ask_init_limit)
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

