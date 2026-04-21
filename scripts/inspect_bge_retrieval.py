"""
BGE-M3 小规模检索可视化（不走 FAQ / LLM / QAService / Router）。

与 validate_small_kb_gpu 相同的数据链路：
  DataLoader -> Cleaner -> Chunker -> VectorRetriever.build
然后对内置示例 query 做 top-k 向量检索并打印命中详情；可选输出 t-SNE 二维散点图。

用法（项目根目录，与 strict GPU 验证一致的环境变量即可）：
  python -m scripts.inspect_bge_retrieval --limit 300 --batch-size 4 --top-k 5
  python -m scripts.inspect_bge_retrieval --query-set education   # 默认已是 education
  python -m scripts.inspect_bge_retrieval --query-set default     # 旧版通用 query（5 条）
  python -m scripts.inspect_bge_retrieval --limit 300 --plot bge_neighbors_tsne.png
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# 旧版通用 query（与早期脚本一致，用于对照；偏 IT/商业，与当前教育语料重合度低）
LEGACY_DEFAULT_QUERIES: Tuple[str, ...] = (
    "如何重置账户密码或找回登录权限？",
    "API 返回 401 Unauthorized 应该如何排查？",
    "How do I request a refund or cancel a subscription?",
    "工单状态一直显示处理中，多久会有回复？",
    "数据库连接超时或连接池耗尽该怎么处理？",
)

# 贴近当前知识库（document：教学案例/数据处理/Turtle；support_ticket：platform_usage / course_design 等）
EDUCATION_SAMPLE_QUERIES: Tuple[str, ...] = (
    # 1 平台使用：与工单里「学生端未开始 + 课堂管理后台确认」话术同域
    "班级开课后学生端一直显示未开始，老师问是不是还需要在课堂管理后台再点一次确认？",
    # 2 平台使用：与「作业批改台 / 发布入口」类工单高度重合
    "老师在作业批改台里找不到上节课的作业发布入口，是入口改版了吗？",
    # 3 教学建议：course_design / 算法课导入、循环嵌套等教研表述
    "算法入门这节课的导入环节有点生硬，想找一个能自然引到循环嵌套的案例，有讲评模板吗？",
    # 4 教学建议：文档中「课程资源中心、备课、检索」等高频词
    "备课时想找可直接复用的课堂示例，应该在课程资源中心怎么检索？",
    # 5 代码报错：Turtle 绘图、课堂演示报错，贴近 DOC 中 Turtle 案例标题与正文
    "学生运行 Turtle 绘图教学案例时终端报错，老师怎么带着学生从报错信息定位到具体行？",
    # 6 Python 语法：与文档「常见教学问题：听懂概念但写代码漏细节」同域
    "学生听得懂列表和字符串的基本用法，但写代码时总漏掉边界条件，课堂上怎么讲更清楚？",
    # 7 FAQ 候选：工单正文里「归档到平台使用 FAQ、关键词别名」等固定表述
    "工单里写「归档到平台使用 FAQ，并建议增加关键词别名」通常是什么意思，对后续检索有什么帮助？",
    # 8 代码报错：IndentationError / 演示代码，贴近 Python 教学场景而非数据库/运维
    "课堂演示的 Python 代码一运行就提示 IndentationError，最常见的原因和改法是什么？",
    # 9 平台使用：与「练习包下发、入口」类工单同域（替代泛化 API/401 问法）
    "老师在练习包下发里找不到上节课的练习入口，是新版入口调整了吗？要怎么确认路径？",
    # 10 FAQ/支持流程：技术支持话术「报错截图、代码前后几行」在工单中反复出现
    "技术支持让补充报错完整截图或代码前后几行，主要是为了判断什么问题类型？",
)

QUERY_SETS: Dict[str, Tuple[str, ...]] = {
    "default": LEGACY_DEFAULT_QUERIES,
    "education": EDUCATION_SAMPLE_QUERIES,
}


def _summarize(text: str, max_chars: int = 220) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _maybe_plot_tsne(
    vec: Any,
    queries: Sequence[str],
    out_path: Path,
    max_points: int,
    random_state: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    index = vec._index  # noqa: SLF001 — 调试脚本刻意读取 FAISS 索引
    chunks = vec._chunks  # noqa: SLF001
    if index is None or not chunks:
        print("[plot] 跳过：索引为空。")
        return

    n = len(chunks)
    take = min(n, max(50, max_points))
    if n > take:
        rng = np.random.default_rng(random_state)
        idx_sel = np.sort(rng.choice(n, size=take, replace=False))
    else:
        idx_sel = np.arange(n)

    chunk_mat = np.stack([index.reconstruct(int(i)) for i in idx_sel], axis=0).astype("float32")

    vec._ensure_model()  # noqa: SLF001
    q_mat = vec._encode_texts(list(queries)).astype("float32")  # noqa: SLF001

    X = np.vstack([chunk_mat, q_mat])
    labels: List[str] = []
    colors: List[str] = []
    for i in idx_sel:
        src = str((chunks[int(i)].metadata or {}).get("source") or "?")
        labels.append(f"chunk:{src}")
        colors.append(
            {
                "document": "#1f77b4",
                "support_ticket": "#ff7f0e",
                "faq": "#2ca02c",
            }.get(src, "#7f7f7f")
        )
    for j, _q in enumerate(queries):
        labels.append(f"query:{j}")
        colors.append("#d62728")

    perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
    tsne_kwargs = dict(
        n_components=2,
        init="pca",
        perplexity=float(perplexity),
        random_state=random_state,
    )
    try:
        tsne = TSNE(learning_rate="auto", **tsne_kwargs)
    except TypeError:
        tsne = TSNE(learning_rate=200.0, **tsne_kwargs)
    Y = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_chunk = Y[: len(idx_sel)]
    y_query = Y[len(idx_sel) :]
    ax.scatter(y_chunk[:, 0], y_chunk[:, 1], c=colors[: len(idx_sel)], s=22, alpha=0.75, label="chunks")
    ax.scatter(y_query[:, 0], y_query[:, 1], c="#d62728", s=120, marker="*", edgecolors="k", linewidths=0.4, label="queries")
    ax.set_title("t-SNE of BGE-M3 dense vectors (sampled chunks + queries)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] 已写入: {out_path.resolve()}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Inspect BGE-M3 retrieval on a small KB (no LLM / QAService).")
    ap.add_argument("--limit", type=int, default=300, help="抽样 raw_records 上限")
    ap.add_argument("--batch-size", type=int, default=4, help="embedding batch size")
    ap.add_argument("--top-k", type=int, default=5, help="每条 query 打印 top-k 命中")
    ap.add_argument("--documents-only", action="store_true", help="仅 document，不加载 support_ticket")
    ap.add_argument(
        "--plot",
        type=str,
        default="",
        help="若指定路径，则尝试输出 t-SNE 散点图（需 matplotlib + scikit-learn）",
    )
    ap.add_argument("--plot-max-points", type=int, default=220, help="散点图中最多抽样多少条 chunk（含全部 query）")
    ap.add_argument("--plot-seed", type=int, default=42, help="抽样与 t-SNE 随机种子")
    ap.add_argument(
        "--query-set",
        type=str,
        choices=sorted(QUERY_SETS.keys()),
        default="education",
        help="示例 query 集合：education=教育语料向（10 条，默认）；default=旧版通用（5 条）",
    )
    args = ap.parse_args()

    queries = QUERY_SETS[str(args.query_set)]

    os.environ["KBQA_EMBEDDING_BATCH_SIZE"] = str(int(args.batch_size))

    from app.config import Settings, get_settings

    get_settings.cache_clear()
    s = Settings()

    from app.services.data_loader import DataLoader
    from app.services.cleaner import Cleaner
    from app.services.chunker import Chunker, ChunkingConfig
    from app.services.vector_retriever import VectorRetriever

    loader = DataLoader()
    docs_records = loader.load_raw_records("document", s.raw_documents_path, limit=int(args.limit))
    tkt_records: list = []
    if not args.documents_only:
        tkt_records = loader.load_raw_records("support_ticket", s.raw_support_tickets_path, limit=int(args.limit))
    docs = loader.to_internal_documents([*docs_records, *tkt_records])
    docs = Cleaner().clean_documents(docs)
    chunker = Chunker(ChunkingConfig(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap))
    chunks = chunker.chunk_documents(docs)

    vec = VectorRetriever()
    vec.build(chunks)

    active = getattr(vec, "_active_embedder", None)
    requested = getattr(vec, "_requested_backend", None)

    print("=== inspect_bge_retrieval ===")
    print("requested embedding_backend:", requested)
    print("active embedder:", active)
    print("embedding_model_name:", s.embedding_model_name)
    print("chunks in index:", len(chunks))
    print("top_k:", int(args.top_k))
    print("query_set:", args.query_set, "n_queries:", len(queries))
    print()

    k = max(1, int(args.top_k))
    for qi, q in enumerate(queries):
        print("=" * 72)
        print(f"[query {qi + 1}/{len(queries)}]")
        print(textwrap.fill(q, width=88))
        hits = vec.search(q, top_k=k)
        if not hits:
            print("  (无命中：索引为空或检索失败)")
            continue
        for rank, h in enumerate(hits, start=1):
            meta = h.metadata or {}
            source_type = str(meta.get("source") or "?")
            title = str(meta.get("title") or "")
            parent_id = str(meta.get("parent_id") or "").strip() or "N/A"
            chunk_level = str(meta.get("chunk_level") or "")
            print(f"  --- rank {rank}  score={h.score:.4f}  chunk_id={h.chunk_id}  doc_id={h.doc_id}")
            print(f"       source_type={source_type}  chunk_level={chunk_level}")
            print(f"       title={title[:120]}{'...' if len(title) > 120 else ''}")
            print(f"       parent_id(meta)={parent_id}")
            print(f"       text_summary={_summarize(h.text)}")
        print()

    if args.plot.strip():
        try:
            _maybe_plot_tsne(
                vec,
                queries,
                Path(args.plot.strip()),
                max_points=int(args.plot_max_points),
                random_state=int(args.plot_seed),
            )
        except ImportError as e:
            print(f"[plot] 跳过：缺少依赖 ({e})。可 pip install matplotlib scikit-learn 后重试。")

    print("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
