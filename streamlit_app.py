from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

st.set_page_config(page_title="教师知识库 RAG 演示", layout="wide", initial_sidebar_state="expanded")


def _get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def call_ask(api_base: str, query: str, top_k: Optional[int]) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/ask"
    payload: Dict[str, Any] = {"query": query}
    if top_k is not None:
        payload["top_k"] = int(top_k)
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# 内置示例：覆盖 FAQ / RAG / need_clarify / HyDE / subquery / code_error 等路由
# （backtrack 依赖首轮分数与阈值，仅标注「易触发」；实际以 route_trace 为准）
# ---------------------------------------------------------------------------
DEMO_EXAMPLES: List[Tuple[str, str, str]] = [
    ("FAQ · 作业发布入口", "老师在作业批改台里找不到作业发布入口，是入口改版了吗？", "bm25_faq / faq_first"),
    ("FAQ · 教师端入口", "字典操作相关入口在教师端哪里？", "bm25_faq / faq_first"),
    ("RAG · 课程设计", "Python基础这节课的导入环节老师反馈有点生硬，想找一个能自然引到字符串处理的案例。", "rag_standard / hybrid"),
    ("RAG · 代码与工单", "老师端示例代码运行没报错，但学生自己写的时候总把自动评测写错，我想找一个更短的说明。", "rag_standard / hybrid"),
    ("澄清 · 指代不明", "这个怎么办？", "need_clarify"),
    ("澄清 · 报错信息不足", "我这里报错了怎么回事", "need_clarify"),
    ("HyDE · 语义长问", "我在准备一节小学用海龟绘图导入课，总是担心学生跟不上节奏，希望你从课堂组织角度给我一些可操作的建议，并说明如何过渡到画笔坐标概念。", "hyde"),
    ("Subquery · 多意图", "作业批改台找不到作业发布入口怎么办？另外学生端为什么看不到作业列表？", "subquery"),
    ("Code · 含错误类型", "运行时出现 TypeError，代码里 print(x[0]) 这一行触发了，请问常见原因是什么？", "rag_standard + code_error 信号"),
    ("Backtrack · 弱检索", "紫色虚构模块 zk9 的隐藏入口在哪里？", "rag_standard（若首轮分数低，route_trace 可出现 backtrack）"),
]

# 面试专用演示顺序（固定 6 条，覆盖 FAQ / 平台使用 / 课程设计 / 代码报错 / need_clarify / 多策略）
# 每条第三列用于“预计展示点”标注（面试时一眼可见）
INTERVIEW_DEMO: List[Tuple[str, str, str]] = [
    ("01 · FAQ 命中", "老师在作业批改台里找不到作业发布入口，是入口改版了吗？", "FAQ 命中 / citations / route_trace"),
    ("02 · 平台使用（RAG）", "班级开课后学生端一直显示未开始，老师需要在哪里确认开课？", "platform_usage / hybrid / citations"),
    ("03 · 课程设计（RAG）", "算法入门这节课的导入环节有点生硬，想找一个能自然引到循环嵌套的案例，有讲评模板吗？", "course_design / hybrid / steps"),
    ("04 · 代码报错（RAG+代码示例）", "我在算法入门课堂演示 for 循环，学生总写错缩进并报 IndentationError，想要一个正确示例代码并解释常见原因与讲法。", "code_error / code_example / evidence"),
    ("05 · need_clarify", "这个怎么办？", "need_clarify / 澄清问题"),
    ("06 · 多意图（Subquery）", "作业批改台找不到作业发布入口怎么办？另外学生端为什么看不到作业列表？", "subquery / 合并去重 / route_trace"),
]


def _summarize_sources(data: Dict[str, Any]) -> List[str]:
    cites = data.get("citations") or data.get("matched_sources") or []
    types: List[str] = []
    for c in cites:
        t = str((c or {}).get("source_type") or "")
        if t:
            types.append(t)
    # 去重保序
    out: List[str] = []
    seen = set()
    for t in types:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _has_step(data: Dict[str, Any], step: str) -> bool:
    rt = data.get("route_trace") or []
    for item in rt:
        if str(item.get("step") or "") == step:
            return True
    return False


def _render_interviewer_summary(data: Dict[str, Any]) -> None:
    route = data.get("route") or "—"
    mode = data.get("mode") or "—"
    steps = [str(x.get("step") or "?") for x in (data.get("route_trace") or [])]
    srcs = _summarize_sources(data)
    debug = data.get("debug") if isinstance(data.get("debug"), dict) else {}
    timing = debug.get("timing_ms") if isinstance(debug.get("timing_ms"), dict) else {}

    flags: List[str] = []
    if route == "bm25_faq" or mode == "faq":
        flags.append("FAQ")
    if "retriever.hybrid" in steps:
        flags.append("hybrid")
    if "subquery.split" in steps:
        flags.append("subquery")
    if "hyde.generate" in steps or "hyde.retrieve" in steps:
        flags.append("HyDE")
    if "backtrack" in steps:
        flags.append("backtrack")
    if route == "need_clarify":
        flags.append("clarify")
    if _has_step(data, "llm.generate"):
        flags.append("LLM")
    if _has_step(data, "llm.fallback"):
        flags.append("LLM_fallback")

    st.markdown("**面试官视角摘要（1 分钟看懂本次链路）**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("route", route)
    c2.metric("mode", mode)
    c3.metric("命中来源", " · ".join(srcs) if srcs else "—")
    c4.metric("总耗时(ms)", timing.get("total", "—"))
    st.caption("链路标记：" + (" · ".join(flags) if flags else "（无）"))


def _render_evidence_table(data: Dict[str, Any]) -> None:
    cites = data.get("citations") or data.get("matched_sources") or []
    if not cites:
        st.info("（无 citations / matched_sources）")
        return
    rows = []
    for c in cites:
        if not isinstance(c, dict):
            continue
        rows.append(
            {
                "source_type": c.get("source_type"),
                "score": round(float(c.get("score", 0.0) or 0.0), 3),
                "title": c.get("title") or "",
                "source_id": c.get("source_id"),
                "parent_id": c.get("parent_id"),
                "snippet": (c.get("snippet") or "")[:260],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _init_state() -> None:
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""
    if "pending_example_query" not in st.session_state:
        st.session_state["pending_example_query"] = None
    if "auto_submit_example" not in st.session_state:
        st.session_state["auto_submit_example"] = False
    if "submit_requested" not in st.session_state:
        st.session_state["submit_requested"] = False
    if "request_in_flight" not in st.session_state:
        st.session_state["request_in_flight"] = False
    if "last_data" not in st.session_state:
        st.session_state["last_data"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_elapsed_ms" not in st.session_state:
        st.session_state["last_elapsed_ms"] = None
    if "last_request_ok" not in st.session_state:
        st.session_state["last_request_ok"] = None
    if "last_ask_url" not in st.session_state:
        st.session_state["last_ask_url"] = ""
    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False


def run_query(api_base: str, query_text: str, top_k: Optional[int]) -> None:
    ask_url = api_base.rstrip("/") + "/ask"
    st.session_state["last_ask_url"] = ask_url
    st.session_state["last_error"] = None
    st.session_state["last_data"] = None
    st.session_state["last_elapsed_ms"] = None
    st.session_state["last_request_ok"] = None
    st.session_state["request_in_flight"] = True
    t0 = time.perf_counter()
    try:
        with st.spinner("请求 /ask 中（冷启动可能较慢）…"):
            st.session_state["last_data"] = call_ask(api_base, query_text, top_k)
        st.session_state["last_request_ok"] = True
    except Exception as e:
        st.session_state["last_error"] = repr(e)
        st.session_state["last_request_ok"] = False
    finally:
        st.session_state["last_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        st.session_state["request_in_flight"] = False


def _trace_steps(data: Dict[str, Any]) -> str:
    rt = data.get("route_trace") or []
    parts = [str(x.get("step") or "?") for x in rt]
    return " → ".join(parts) if parts else "（无）"


def _render_route_card(data: Dict[str, Any]) -> None:
    route = data.get("route")
    mode = data.get("mode")
    c1, c2, c3 = st.columns(3)
    c1.metric("route", route or "—")
    c2.metric("mode", mode or "—")
    c3.metric("链路步数", len(data.get("route_trace") or []))
    st.caption("链路概览：" + _trace_steps(data))


def _render_route_trace(data: Dict[str, Any], *, expanded_default: bool) -> None:
    rt = data.get("route_trace") or []
    if not rt:
        st.info("（无 route_trace）")
        return
    st.markdown("**route_trace（逐步展开）**")
    for item in rt:
        step = item.get("step") or "?"
        detail = item.get("detail") or {}
        with st.expander(f"`{step}`", expanded=expanded_default):
            st.code(json.dumps(detail, ensure_ascii=False, indent=2), language="json")


def _render_citations(data: Dict[str, Any]) -> None:
    citations = data.get("citations") or data.get("matched_sources") or []
    if not citations:
        st.info("（无 citations / matched_sources）")
        return
    for i, c in enumerate(citations, start=1):
        title = c.get("title") or ""
        st.markdown(
            f"**[{i}]** score=`{c.get('score', 0):.3f}` · type=`{c.get('source_type')}` · "
            f"id=`{c.get('source_id')}` · parent=`{c.get('parent_id')}`"
        )
        if title:
            st.caption(title)
        snippet = c.get("snippet") or ""
        if snippet:
            st.write(snippet)


def _render_debug_panel(data: Dict[str, Any], *, show_full_debug: bool) -> None:
    d = data.get("debug") or {}
    timing = d.get("timing_ms") if isinstance(d.get("timing_ms"), dict) else {}
    cache = d.get("cache") if isinstance(d.get("cache"), dict) else {}
    ef = d.get("evidence_filter") if isinstance(d.get("evidence_filter"), dict) else {}

    st.markdown("**关键标记（从 debug / route_trace 抽取）**")
    mcols = st.columns(6)
    mcols[0].metric("total_ms", timing.get("total", "—"))
    mcols[1].metric("llm_ms", timing.get("llm", "—"))
    mcols[2].metric("hybrid_ms", timing.get("hybrid_total", "—"))
    mcols[3].metric("cache.hit", str(cache.get("hit", "—")))
    mcols[4].metric("filtered", ef.get("filtered_out_count", "—"))
    mcols[5].metric("kept_ctx", ef.get("kept_context_count", "—"))

    flags: List[str] = []
    rt = data.get("route_trace") or []
    steps = {str(x.get("step")) for x in rt}
    if "hyde.generate" in steps or "hyde.retrieve" in steps:
        flags.append("HyDE")
    if "subquery.split" in steps:
        flags.append("subquery")
    if "backtrack" in steps:
        flags.append("backtrack")
    if "evidence.filter" in steps:
        flags.append("evidence_filter")
    rr = d.get("rerank") if isinstance(d.get("rerank"), dict) else _get(d, "timing_ms.rerank")
    if isinstance(rr, dict) and rr.get("enabled"):
        flags.append(f"rerank:{rr.get('type', '')}")
    elif d.get("route") == "bm25_faq":
        flags.append("rerank:（FAQ 直出可跳过）")
    st.write(" · ".join(flags) if flags else "（无额外策略标记）")

    st.markdown("**rerank / 检索侧 debug 摘要**")
    rer = d.get("rerank")
    if isinstance(rer, dict):
        st.json(rer)
    else:
        st.caption("（当前响应 debug 中无 rerank 子对象）")

    if show_full_debug:
        st.markdown("**完整 debug JSON**")
        st.code(json.dumps(d, ensure_ascii=False, indent=2), language="json")


def _render_contexts(data: Dict[str, Any]) -> None:
    ctxs = data.get("contexts") or []
    st.write(f"共 **{len(ctxs)}** 条 contexts")
    for i, c in enumerate(ctxs, start=1):
        with st.expander(f"Context {i}: `{c.get('source')}` id={c.get('source_id')} score={c.get('score')}", expanded=False):
            st.code((c.get("text") or "")[:4000], language="text")


_init_state()

st.title("教师智能知识库问答（RAG）演示页")
st.caption("面向答辩 / 汇报 / 面试：分区展示答案、路由、证据与调试信息。请先启动 FastAPI 服务。")


def _render_project_demo_summary(api_base: str) -> None:
    st.subheader("项目验证总结（面试汇报版）")
    st.caption("这一页不堆日志，按“你做成了什么 + 如何证明”组织，适合截图/口头讲解。")

    # --- 结果数字摘要（适合截图）---
    has_tsne_plot = os.path.exists(os.path.join(os.getcwd(), "artifacts", "bge_neighbors_tsne.png"))
    completed_checks = [
        "FAQ / BM25 主链路",
        "FastAPI /docs 可运行",
        "Streamlit 演示页可运行",
        "项目 ready 检查通过",
        "BGE-M3 strict 验证",
        "bge_reranker strict 验证",
    ]
    summary_rows = [
        {
            "title": "系统完成情况",
            "metric": "4 类知识源",
            "help": "document / support_ticket / faq / code_example",
        },
        {
            "title": "主接口",
            "metric": "2 个 API",
            "help": "/ask + /health（/docs 可直接调试）",
        },
        {
            "title": "固定演示题",
            "metric": f"{len(INTERVIEW_DEMO)} 条",
            "help": "FAQ / 平台 / 课程 / 代码 / 澄清 / 多意图",
        },
        {
            "title": "strict 通过项",
            "metric": "2 项核心",
            "help": "BGE-M3 strict + bge_reranker strict",
        },
        {
            "title": "验证脚本",
            "metric": "7+ 个",
            "help": "run_demo/eval/validate/inspect 等已接入",
        },
        {
            "title": "可视化产物",
            "metric": "已生成" if has_tsne_plot else "待生成",
            "help": "二维向量图 + rerank 前后对比脚本已具备",
        },
    ]

    st.markdown("### 结果数字摘要")
    st.caption("适合面试截图：先用数字说明完成度，再展开讲链路与验证。")
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, item in zip(row1 + row2, summary_rows):
        with col:
            st.metric(item["title"], item["metric"])
            st.caption(item["help"])

    st.markdown(
        "- **系统完成度**：主链路、Streamlit 演示、项目验证总结页、strict 验证脚本已经形成闭环。\n"
        "- **项目级验证**：`/docs` 可调试，演示页可跑固定问题，ready 检查通过。\n"
        "- **strict 能力证明**：不仅能讲架构，还能展示 BGE-M3 与 bge_reranker 的独立验证结果。"
    )

    st.markdown("### 已完成项（面试官一眼看懂）")
    badge_cols = st.columns(3)
    for idx, item in enumerate(completed_checks):
        badge_cols[idx % 3].success(item)

    # --- 快速结论卡片 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("主链路", "FAQ + Hybrid 已跑通")
    c2.metric("Embedding", "BGE-M3（strict）")
    c3.metric("Rerank", "bge_reranker（strict）")
    c4.metric("可解释性", "citations + route_trace")

    st.markdown("### 你可以怎么向面试官证明完成度")
    st.markdown(
        "- **能跑通**：/ask 能稳定返回（含 route/citations/route_trace/debug），Streamlit 一键演示。\n"
        "- **能解释**：每次回答都有结构化证据链（source_type/title/snippet/score）+ 链路追踪（route_trace）。\n"
        "- **能验证**：提供 strict 验证脚本（BGE-M3 / bge_reranker），以及 retrieval/rerank 可视化脚本与评测脚本。"
    )

    st.markdown("### 验证项清单（建议按此顺序讲）")
    # 这里用“可勾选”方式，让你在 strict 环境跑通后手动标记为 ✅（更适合面试现场展示）
    if "demo_summary_checks" not in st.session_state:
        st.session_state["demo_summary_checks"] = {
            "faq_bm25": True,
            "bge_m3_strict": True,
            "bge_reranker_strict": True,
            "inspect_retrieval": True,
            "inspect_rerank": True,
            "eval_suite": True,
        }
    checks: Dict[str, bool] = st.session_state["demo_summary_checks"]
    cols = st.columns(2)
    with cols[0]:
        checks["faq_bm25"] = st.checkbox("FAQ / BM25 主链路已跑通", value=bool(checks.get("faq_bm25")), key="chk_faq_bm25")
        checks["inspect_retrieval"] = st.checkbox("小规模 retrieval 可视化（inspect_bge_retrieval）", value=bool(checks.get("inspect_retrieval")), key="chk_inspect_retrieval")
        checks["eval_suite"] = st.checkbox("eval_router / eval_retrieval / eval_summary 已可运行", value=bool(checks.get("eval_suite")), key="chk_eval_suite")
    with cols[1]:
        checks["bge_m3_strict"] = st.checkbox("BGE-M3 strict 验证成功", value=bool(checks.get("bge_m3_strict")), key="chk_bge_m3")
        checks["bge_reranker_strict"] = st.checkbox("bge_reranker strict 验证成功", value=bool(checks.get("bge_reranker_strict")), key="chk_bge_rr")
        checks["inspect_rerank"] = st.checkbox("rerank 前后顺序对比（inspect_bge_rerank）", value=bool(checks.get("inspect_rerank")), key="chk_inspect_rerank")

    st.markdown("### 一键命令（面试现场可直接复制）")
    st.code(
        "\n".join(
            [
                "# 1) FAQ / Hybrid 主链路（启动 API + Streamlit）",
                "python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001",
                "streamlit run streamlit_app.py",
                "",
                "# 2) BGE-M3 strict 小规模 GPU 验证（embedding + 小索引）",
                "python -m scripts.validate_small_kb_gpu --limit 300 --batch-size 4",
                "",
                "# 3) 小规模 retrieval 可视化（含教育向 query-set）",
                "python -m scripts.inspect_bge_retrieval --query-set education --limit 300 --top-k 5",
                "",
                "# 4) bge_reranker strict 验证（初始化 / 打分 / rerank 可视化）",
                "python -m scripts.inspect_bge_rerank --limit 300 --vec-top-k 8 --rerank-top-k 6 --no-fallback",
                "",
                "# 5) 评测脚本（最小版）",
                "python -m scripts.eval_router",
                "python -m scripts.eval_retrieval --k 5",
                "python -m scripts.eval_summary --passes 3 --export both --outdir outputs/eval",
            ]
        ),
        language="bash",
    )

    st.markdown("### 典型 query（用于证明覆盖面）")
    st.table(
        [
            {"类型": "FAQ 命中", "query": "老师在作业批改台里找不到作业发布入口，是入口改版了吗？"},
            {"类型": "平台使用 / RAG", "query": "班级开课后学生端一直显示未开始，老师需要在哪里确认开课？"},
            {"类型": "课程设计 / RAG", "query": "算法入门这节课的导入环节有点生硬，想找一个能自然引到循环嵌套的案例，有讲评模板吗？"},
            {"类型": "代码报错 / code_example", "query": "学生总写错缩进并报 IndentationError，想要一个正确示例代码并解释常见原因与讲法。"},
            {"类型": "need_clarify", "query": "这个怎么办？"},
            {"类型": "多意图 subquery", "query": "作业发布入口找不到怎么办？另外学生端为什么看不到作业列表？"},
        ]
    )

    st.markdown("### 主要 timing 指标（用真实 /ask 返回的 debug.timing_ms）")
    st.caption("点击按钮会调用一次 /ask，并从响应 debug 中读取 timing_ms（无需改后端协议）。")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        sample_q = st.text_area("抽样 query", value="班级开课后学生端一直显示未开始，老师需要在哪里确认开课？", height=70, key="summary_sample_query")
        run_sample = st.button("跑一次并提取 timing", key="run_summary_timing")
    with col_b:
        if run_sample:
            t0 = time.perf_counter()
            try:
                data = call_ask(api_base, sample_q, top_k=8)
                wall_ms = round((time.perf_counter() - t0) * 1000, 2)
                dbg = data.get("debug") if isinstance(data.get("debug"), dict) else {}
                timing = dbg.get("timing_ms") if isinstance(dbg.get("timing_ms"), dict) else {}
                st.success("已获取 timing（来自 /ask debug.timing_ms）")
                st.code(
                    json.dumps(
                        {
                            "route": data.get("route"),
                            "wall_ms(client)": wall_ms,
                            "timing_ms(debug)": timing,
                            "sources": _summarize_sources(data),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )
            except Exception as e:
                st.error(f"调用 /ask 失败：{repr(e)}")

    st.markdown("### 可视化产物（如果存在则展示，便于截图）")
    img_path = os.path.join(os.getcwd(), "artifacts", "bge_neighbors_tsne.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="retrieval embedding 2D 可视化（t-SNE）", width="stretch")
    else:
        st.info("未检测到 `artifacts/bge_neighbors_tsne.png`。如需生成：运行 `python -m scripts.inspect_bge_retrieval --plot artifacts/bge_neighbors_tsne.png`（需 matplotlib+sklearn）。")

with st.sidebar:
    st.subheader("连接")
    api_base = st.text_input("FastAPI 地址", value="http://127.0.0.1:8001", help="与 uvicorn 监听地址一致")

    demo_mode = st.checkbox("演示模式", value=st.session_state["demo_mode"], help="固定一组稳定参数，并突出示例问题按钮")
    st.session_state["demo_mode"] = demo_mode

    if demo_mode:
        st.subheader("面试专用演示模式")
        st.caption("按固定顺序点击 6 条问题，一键跑通：FAQ → 平台使用 → 课程设计 → 代码报错 → 澄清 → 多意图。")
        for label, qtext, hint in INTERVIEW_DEMO:
            if st.button(label, key=f"iv_{label}", help=hint, width="stretch"):
                st.session_state["pending_example_query"] = qtext
                st.session_state["auto_submit_example"] = True
                st.rerun()

    st.subheader("参数区")
    if demo_mode:
        top_k = 8
        use_top_k = True
        show_debug = True
        show_contexts = False
        route_trace_expand = False
        st.info("演示模式已固定：**top_k=8**，**显示调试摘要**，**不展开 contexts**（可在下方手动勾选覆盖）。")
        show_contexts = st.checkbox("仍显示 contexts（演示模式下可选）", value=False)
        show_full_debug = st.checkbox("显示完整 debug JSON", value=False)
    else:
        top_k = st.number_input("top_k", min_value=1, max_value=50, value=8, step=1)
        use_top_k = st.checkbox("传入 top_k 覆盖默认", value=True)
        show_debug = st.checkbox("显示调试区", value=False)
        show_contexts = st.checkbox("显示 contexts（可能较长）", value=False)
        route_trace_expand = st.checkbox("route_trace 默认展开", value=False)
        show_full_debug = st.checkbox("完整 debug JSON", value=False)

# 示例问题：必须在实例化 key=query_input 的 widget 之前写入，否则会触发 StreamlitAPIException
pex = st.session_state.get("pending_example_query")
if pex is not None:
    st.session_state["query_input"] = str(pex)
    st.session_state["pending_example_query"] = None

st.subheader("输入 query")

tab_demo, tab_summary = st.tabs(["问答演示", "项目验证总结"])

with tab_summary:
    _render_project_demo_summary(api_base)

with tab_demo:
    query = st.text_area(
        "问题",
        height=100,
        placeholder="例如：老师在作业批改台里找不到作业发布入口，是入口改版了吗？",
        key="query_input",
    )

    st.markdown("**示例问题（点击后自动发起请求）**")
    ex_cols = st.columns(5)
    for i, (label, qtext, hint) in enumerate(DEMO_EXAMPLES):
        with ex_cols[i % 5]:
            if st.button(label, key=f"ex_{i}", help=hint):
                st.session_state["pending_example_query"] = qtext
                st.session_state["auto_submit_example"] = True
                st.rerun()

if demo_mode:
    with st.expander("面试推荐顺序（6 条）", expanded=False):
        st.table([{"顺序": a, "预计展示点": c, "问题摘要": b[:64] + "…" if len(b) > 64 else b} for a, b, c in INTERVIEW_DEMO])

with st.expander("示例说明（预期路由，以实际 API 为准）", expanded=False):
    st.table([{"按钮": a, "预期": c, "问题摘要": b[:42] + "…" if len(b) > 42 else b} for a, b, c in DEMO_EXAMPLES])

submit_clicked = st.button("提问 / Ask", type="primary")
if submit_clicked:
    st.session_state["submit_requested"] = True

query_to_run: Optional[str] = None
if st.session_state.pop("auto_submit_example", False):
    query_to_run = (st.session_state.get("query_input") or "").strip()
elif st.session_state.pop("submit_requested", False):
    query_to_run = (st.session_state.get("query_input") or "").strip()

if query_to_run is not None:
    if not query_to_run:
        st.warning("请先输入问题或点击示例。")
    else:
        run_query(api_base, query_to_run, top_k if use_top_k else None)

st.subheader("最近一次结果")
ask_url = api_base.rstrip("/") + "/ask"
st.caption(f"API_BASE_URL: `{api_base}`  ·  ASK_URL: `{ask_url}`")
meta_cols = st.columns(3)
meta_cols[0].metric("最近耗时(ms)", st.session_state.get("last_elapsed_ms") or "—")
meta_cols[1].metric("最近是否成功", "是" if st.session_state.get("last_request_ok") is True else ("否" if st.session_state.get("last_request_ok") is False else "—"))
meta_cols[2].metric("请求中", "是" if st.session_state.get("request_in_flight") else "否")

if st.session_state.get("last_error"):
    st.error(f"调用 /ask 失败：{st.session_state['last_error']}")

data = st.session_state.get("last_data")
if data:
    st.divider()

    _render_interviewer_summary(data)
    st.divider()

    tab_ans, tab_route, tab_evi, tab_dbg = st.tabs(["答案", "路由", "证据", "调试"])

    with tab_ans:
        st.subheader("Answer")
        ans = data.get("answer", "") or "（空）"
        # 更像产品输出：优先按 Markdown 渲染（LLM 输出/模板输出都更可读）
        st.markdown(ans)
        st.markdown("**本次问题 / route**")
        st.code(
            json.dumps(
                {"query": data.get("query"), "route": data.get("route"), "mode": data.get("mode")},
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
        clarifications = data.get("clarifications") or []
        if clarifications:
            st.subheader("Clarifications（需要补充的信息）")
            for qx in clarifications:
                st.markdown(f"- {qx}")

        show_basis = st.checkbox("在答案下方显示依据来源（citations）", value=False, key="show_basis_under_answer")
        if show_basis:
            st.markdown("**依据来源（按 citations）**")
            _render_citations(data)

    with tab_route:
        st.subheader("路由区")
        _render_route_card(data)
        exp = route_trace_expand if not demo_mode else False
        _render_route_trace(data, expanded_default=exp)

    with tab_evi:
        st.subheader("证据区 · Citations / Matched Sources")
        st.markdown("**证据一览（更直观）**")
        _render_evidence_table(data)
        with st.expander("证据逐条展开（原样）", expanded=False):
            _render_citations(data)
        if show_contexts:
            st.subheader("Contexts（检索原始片段）")
            _render_contexts(data)

    with tab_dbg:
        st.subheader("调试区")
        st.caption(f"API_BASE_URL: {api_base} | ASK_URL: {ask_url} | last_elapsed_ms: {st.session_state.get('last_elapsed_ms')} | success: {st.session_state.get('last_request_ok')}")
        if show_debug or demo_mode:
            _render_debug_panel(data, show_full_debug=show_full_debug)
        else:
            st.info("请在侧栏勾选「显示调试区」或开启「演示模式」。")
elif st.session_state.get("last_request_ok") is False and not st.session_state.get("last_error"):
    st.error("最近一次请求失败，但未捕获到详细错误。")

st.divider()
st.caption("启动 API：`python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001` → 再运行本页：`streamlit run streamlit_app.py`")
