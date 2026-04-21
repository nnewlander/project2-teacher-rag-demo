from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# 必须使用 pydantic-settings 的 BaseSettings，否则 env_prefix / env_file 不会生效（普通 BaseModel 不读环境变量）。
from pydantic_settings import BaseSettings, SettingsConfigDict

# 与 .env 解析一致：相对仓库根目录，避免 uvicorn 工作目录不在项目根时漏读
_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_DIR.parent
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """
    MVP 配置集中管理。

    TODO:
    - 支持多环境（dev/staging/prod）分层配置
    - 统一日志/观测（trace_id, latency, hit_rate）
    """

    # env：KBQA_<FIELD_NAME>（大写）；布尔值支持 true/false/1/0 等；进程环境变量优先于 .env
    model_config = SettingsConfigDict(
        env_prefix="KBQA_",
        env_file=_DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # 数据路径（默认指向当前仓库示例数据）
    data_root: Path = Path("project2_rag_raw_data_10pct") / "data"
    raw_documents_path: Path = data_root / "raw_documents_10pct.jsonl"
    raw_support_tickets_path: Path = data_root / "raw_support_tickets_10pct.jsonl"
    raw_faq_path: Path = data_root / "raw_faq_10pct.jsonl"
    raw_code_examples_path: Path = data_root / "raw_code_examples_10pct.jsonl"

    # Chunk 参数（先做最小切分）
    chunk_size: int = 600
    chunk_overlap: int = 80

    # FAQ 命中阈值（BM25 分数越大越相关；阈值需结合数据调参）
    faq_min_score: float = 6.0

    # 召回数量
    faq_top_k: int = 5
    vector_top_k: int = 6
    hybrid_top_k: int = 8

    # -------------------------------------------------------------------------
    # Backend 抽象（与项目二简历/文档口径对齐：可切换实现，默认保持当前可运行栈）
    # 说明：以下为「配置契约」；除默认组合外，其它 backend 仅预留占位，不在本仓库接入重依赖。
    # -------------------------------------------------------------------------
    #
    # Device / precision（strict 环境 GPU 优先；无 GPU 自动回退 CPU）
    # - device: auto | cpu | cuda
    # - use_fp16: auto | true | false（auto: cuda 默认 true，cpu 默认 false）
    device: str = "auto"
    use_fp16: str = "auto"
    #
    # Embedding：dense 向量编码（sentence_transformers 默认可运行；BGE-M3 需 FlagEmbedding，见 README）
    embedding_backend: str = "sentence_transformers"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 64
    # bge_m3 加载失败时是否回退到 sentence_transformers（False 则抛出原始异常，便于排查）
    embedding_fallback_on_error: bool = True
    embedding_fallback_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #
    # Vector store：向量索引与检索后端（当前为进程内 FAISS；未来可接 Milvus 等）
    vector_store_backend: str = "faiss"
    #
    # Rerank：keyword_overlap_reranker（默认）| bge_reranker（FlagReranker）| cross_encoder（CrossEncoder）
    reranker_backend: str = "keyword_overlap_reranker"
    reranker_model_name: str = ""
    reranker_batch_size: int = 16
    reranker_fallback_on_error: bool = True
    #
    # Cache：问答结果缓存后端（当前为进程内 TTL；未来可接 Redis）
    cache_backend: str = "ttl_memory_cache"
    cache_enabled: bool = True
    cache_ttl_s: int = 300
    cache_max_size: int = 512
    cache_min_hits_to_store: int = 2  # query 出现次数达到阈值后才写入缓存（避免污染）

    # 证据过滤（检索 + rerank 之后、生成之前；规则版）
    evidence_filter_enabled: bool = True
    # 与 KeywordOverlapReranker 同构的轻量相关性下界（overlap + 0.05*raw_score）
    evidence_filter_min_relevance: float = 0.06
    evidence_filter_min_score_faq: float = 3.0
    evidence_filter_min_score_document: float = 2.5
    evidence_filter_min_score_support_ticket: float = 2.0
    evidence_filter_min_score_external_ref: float = 3.5
    evidence_filter_ocr_mode: str = "penalty"  # penalty | drop
    evidence_filter_ocr_score_factor: float = 0.72
    evidence_filter_near_dup_ratio: float = 0.88

    # backtrack（轻量回溯补召回）
    # 触发条件：首轮 hybrid top1 分数 < backtrack_min_top_score
    # TODO: 以后改为更稳健的质量判定（如分数归一化、top-k 分布、空召回等）
    backtrack_enabled: bool = True
    backtrack_min_top_score: float = 6.2
    backtrack_neighbor_window: int = 1
    backtrack_max_extra_contexts: int = 6

    # LLM（RAG 生成）
    # 说明：为避免绑定具体线上模型，这里采用“OpenAI 兼容”接口配置（可接本地/私有/云端任意兼容服务）
    # TODO: 后续可扩展更多 provider（如 qwen/openrouter/ollama 等），保持同一调用接口
    llm_provider: str = "disabled"  # disabled | openai_compatible
    llm_base_url: str = ""  # 例如：http://localhost:8000/v1
    llm_api_key: str = ""  # 如需鉴权则填写；本地服务可留空
    llm_model_name: str = ""  # 例如：gpt-4o-mini / qwen2.5 / deepseek-chat 等（取决于你的兼容服务）
    llm_temperature: float = 0.2
    llm_timeout_s: int = 60

    # 服务基础信息
    app_name: str = "Teacher KBQA RAG (MVP)"

    # strict 环境项目级验证优化：允许 /health 仅做小规模初始化，避免首次健康检查触发全量建索引
    # - 0 / 未设置：保持默认行为（/health 首次访问走完整 init_kb）
    # - >0：/health 首次访问仅抽样初始化 limit 条各源数据
    health_init_limit: int = 0

    # strict 环境项目级 ask 优化：允许第一次 /ask 使用小规模初始化，而非强制全量 init_kb
    # - 0 / 未设置：保持当前行为（若未初始化则完整 init；若之前是 /health 的轻量初始化则补成完整 init）
    # - >0：第一次 /ask 若尚未完整初始化，则只使用该 limit 做项目级验证
    ask_init_limit: int = 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    首次调用时实例化并缓存。若在进程内修改了环境变量，需先执行
    `get_settings.cache_clear()` 再调用，否则会沿用旧配置。
    """
    return Settings()

