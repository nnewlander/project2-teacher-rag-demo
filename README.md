# 教师智能知识库问答系统（RAG MVP）

一个面向**中文教育场景**的知识库问答项目，目标不是做“自由聊天机器人”，而是做一个**可解释、可验证、可演示、可扩展**的 RAG 系统：

- 能接入多源教育知识数据
- 能区分 FAQ、平台使用、代码报错、课程设计等问题类型
- 能走 FAQ 优先 + RAG 兜底的主链路
- 能输出结构化证据链与路由链路
- 能在 strict 环境下验证 BGE-M3 / bge_reranker
- 能通过 Streamlit 进行面试演示与项目汇报

---

## 1. 项目背景

在真实教育场景里，老师和教研/技术支持团队会遇到大量重复但表达方式不统一的问题，例如：

- **平台使用类**：作业发布入口在哪里、学生端为什么显示未开始、课堂状态为什么不同步
- **代码报错类**：课堂演示代码为什么报 `IndentationError` / `TypeError`
- **课程设计类**：如何自然引入循环嵌套、如何组织课堂导入与讲评
- **FAQ 高频问题**：某些入口、操作路径、固定规则类问题可以直接标准回答

这些知识通常散落在：

- 内部文档
- 历史工单
- FAQ
- 代码示例与错误说明

这个项目的目标就是把这些异构知识统一治理后，用 RAG 的方式提供一个更像真实产品的问答系统，而不是只返回一个“模型生成的黑盒答案”。

---

## 2. 项目目标

一句话概括：

> 做一个教师知识库问答 MVP，让系统既能回答问题，又能解释“为什么这样回答”。

核心目标包括：

1. 跑通 FAQ + 混合检索 + 生成的主链路
2. 保持返回结构可解释：`answer + citations + route_trace + debug`
3. 让 Streamlit 页面可用于答辩、面试、项目演示
4. 通过 strict 脚本验证 BGE-M3 与 bge_reranker 的接入能力
5. 给后续 Milvus / Redis / OCR / 更强路由等扩展预留接口

---

## 3. 当前已接入的数据源

当前主链路已接入 4 类知识源：

1. **document**：内部文档 / 教学资料 / 平台说明
2. **support_ticket**：历史工单 / 技术支持记录
3. **faq**：标准问答
4. **code_example**：代码示例 / 常见错误 / 说明文本

默认数据文件：

- `project2_rag_raw_data_10pct/data/raw_documents_10pct.jsonl`
- `project2_rag_raw_data_10pct/data/raw_support_tickets_10pct.jsonl`
- `project2_rag_raw_data_10pct/data/raw_faq_10pct.jsonl`
- `project2_rag_raw_data_10pct/data/raw_code_examples_10pct.jsonl`

这些数据会统一映射为 `InternalDocument`，然后进入清洗、切块、检索与生成链路。

---

## 4. 系统架构总览

项目结构如下：

```text
app/
  main.py
  config.py
  api/
    routes.py
  schemas/
    query.py
    answer.py
    document.py
  services/
    data_loader.py
    cleaner.py
    chunker.py
    faq_retriever.py
    vector_retriever.py
    hybrid_retriever.py
    reranker.py
    router.py
    query_processor.py
    hyde_generator.py
    subquery_builder.py
    evidence_filter.py
    llm_client.py
    qa_service.py
    cache.py
  utils/
    jsonl.py
    text_utils.py

scripts/
  run_demo.py
  run_demo_suite.py
  eval_router.py
  eval_retrieval.py
  eval_summary.py
  validate_bge_stack.py
  validate_small_kb_gpu.py
  inspect_bge_retrieval.py
  inspect_bge_rerank.py
  build_faq_candidates.py
  ingest_data.py
  parse_raw_sources.py
  run_api.py

docs/
  interview_project2_overview.md

streamlit_app.py
README.md
requirements.txt
requirements-bge-strict.txt
```

架构可以概括为 6 层：

1. **配置层**：`app/config.py`
2. **数据接入与治理层**：DataLoader / Cleaner / Chunker
3. **检索层**：FAQ Retriever / Vector Retriever / Hybrid Retriever / Reranker
4. **策略层**：QueryProcessor / Router / HyDE / Subquery / EvidenceFilter
5. **生成层**：LLMClient + fallback answer
6. **编排层**：QAService

---

## 5. 在线请求链路

### 5.1 `/ask` 的主要流程

一次 `/ask` 大致会经过以下步骤：

1. **初始化知识库**（必要时）
2. **QueryProcessor**：清洗 query，并给出轻量 query_type
3. **Router**：判断走哪条策略
4. **FAQ 优先**：如果是 FAQ-like 且命中阈值足够高，直接 FAQ 直答
5. **Hybrid 检索**：FAQ 候选 + 向量候选融合
6. **可选策略增强**：HyDE / Subquery / Backtrack
7. **EvidenceFilter**：过滤弱相关或重复证据
8. **构造生成上下文**：child 命中后回溯 parent chunk
9. **LLM 生成**：基于证据生成最终答案
10. **Fallback**：若未配置 LLM，则返回安全模板答案

最终响应包含：

- `answer`
- `contexts`
- `citations`
- `matched_sources`
- `route_trace`
- `clarifications`
- `faq_id`
- `filtered_out_count`
- `kept_context_count`
- `debug`

### 5.2 `/search`（evidence-only）接口用途

`/search` 专门用于“只返回检索证据”，不负责生成最终答案，适合被外部系统（如项目一 Agent 平台）通过 HTTP 远程调用。

典型用途：

- 项目一的 `RemoteRAGAdapter` 先调用 `/search` 拿证据
- 项目一在自身流程中做最终诊断/干预建议生成
- 项目二保持检索服务职责，不与项目一代码强耦合

`/search` 返回字段固定为：

- `hits`：证据列表（每条含 `source_id/title/snippet/score/source_type/metadata`）
- `query`：原始 query
- `route_trace`：检索链路步骤
- `debug`：检索调试信息（如 `top_k`、retriever 标识等）

### 5.3 `/health` 的当前设计

`/health` 不仅是普通健康检查，还用于项目级初始化验证。

为了避免 strict 环境中首次 `/health` 触发全量 embedding + 建索引，现在支持：

- `KBQA_HEALTH_INIT_LIMIT`

含义：

- 未设置或为 `0`：保持默认行为，首次 `/health` 走完整初始化
- `>0`：首次 `/health` 只做小规模初始化，用于 strict 环境快速验证项目链路

---

## 6. 核心模块说明

### 6.1 DataLoader

负责：

- 读取不同原始 jsonl 文件
- 映射成统一的 `InternalDocument`
- 在 metadata 中保留来源信息，如 `source_type`

当前已支持：

- `document`
- `support_ticket`
- `faq`
- `code_example`

### 6.2 Cleaner

负责基础清洗与统一化处理，减少脏数据对后续检索与生成的影响。

### 6.3 Chunker

负责把文档切成检索用 chunk。

特点：

- 普通文档 / 工单 / FAQ：走通用切块策略
- `code_example`：尽量保留标题、代码、说明文本，不粗暴打散代码结构

### 6.4 FaqRetriever

负责 FAQ 检索。

设计思想：

- 高频标准问题优先走 BM25/关键词命中
- 命中后直接返回 FAQ 答案
- 避免不必要的复杂 RAG 链路与模型开销

### 6.5 VectorRetriever

负责：

- 对 chunk 做 embedding
- 构建 FAISS 索引
- 做 top-k 向量召回

默认后端：

- `sentence_transformers`

strict/升级后端：

- `bge_m3`

### 6.6 HybridRetriever

负责：

- 融合 FAQ 候选与向量候选
- 去重
- 调用轻量 rerank

适用于 FAQ 未直达时的主检索路径。

### 6.7 Reranker

支持的思路：

- 默认：轻量关键词 overlap reranker
- strict/升级：`bge_reranker`

作用：

- 解决召回后候选排序不够精细的问题
- 在不重构主流程的情况下提升 top-k 质量

### 6.8 QueryProcessor + Router

负责把 query 分流到不同策略：

- `faq_first`
- `need_clarify`
- `rag_standard`
- `hyde`
- `subquery`

这样系统就不是“所有问题都走一条路径”，而是根据问题类型选择更合适的处理策略。

### 6.9 HyDE

适用于长语义、抽象表达的问题。

策略：

- 先生成假设性描述
- 再用该描述辅助检索

### 6.10 Subquery

适用于多意图问题。

策略：

- 把 query 拆成多个子问题
- 分别检索
- 合并去重
- 轻量重排

### 6.11 EvidenceFilter

位于检索与生成之间。

作用：

- 过滤弱相关/低可信/近重复证据
- 保持进入生成阶段的证据更稳定
- 同时将过滤结果写入 debug，方便解释与排查

### 6.12 LLMClient

生成层使用 OpenAI-compatible 接口，不额外造新调用体系。

支持：

- 本地兼容服务
- 私有兼容服务
- 云端兼容服务

输出要求：

- 基于检索证据
- 先回答结论
- 再给步骤或解释
- 证据不足时明确说明

### 6.13 QAService

`QAService` 是整个项目的主编排器，负责把上面所有模块串起来。

它不是某个单点功能，而是整个系统的“主控制器”。

---

## 7. FAQ / 检索 / 路由 / 生成之间的关系

这个项目不是“单模型直答”，而是典型的**分层 RAG 架构**：

### FAQ 的作用

- 优先处理高频稳定问题
- 快速、可控、低成本

### 向量检索的作用

- 处理非标准问法
- 处理长 query、口语化 query
- 处理 support_ticket / code_example / document 的混合语义检索

### Rerank 的作用

- 提升候选排序质量
- 让更有用的证据排在前面

### Router 的作用

- 根据问题类型选择路径
- 避免所有问题都使用单一路径
- 提升复杂 query 处理的合理性

### 生成层的作用

- 把命中的证据组织成更自然、更像真实产品的答案
- 但仍受证据约束，不允许完全脱离检索内容自由发挥

---

## 8. Streamlit 演示页能力

入口文件：

- `streamlit_app.py`

当前页面具备以下能力：

### 8.1 问答演示

- 手动输入 query 并发送
- 点击示例问题自动发起请求
- 统一请求主流程
- 展示最近一次结果

### 8.2 面试专用演示模式

固定 6 条顺序问题，覆盖：

1. FAQ 命中
2. 平台使用
3. 课程设计
4. 代码报错
5. need_clarify
6. 多意图 subquery

### 8.3 项目验证总结页

页面内新增了“项目验证总结”Tab，适合：

- 面试汇报
- 截图
- 汇总项目验证成果

### 8.4 可解释性展示

页面可视化展示：

- `answer`
- `route`
- `citations`
- `route_trace`
- `debug`
- `contexts`
- timing 信息

这使得系统不只是一个 API，而是一个完整的可演示项目。

---

## 9. 默认演示环境 vs strict 验证环境

项目刻意区分了两个环境用途。

### 9.1 默认演示环境

依赖：

```bash
pip install -r requirements.txt
```

目标：

- 跑通 `/ask`
- 跑通 Streamlit
- 跑通 FAQ / hybrid / route_trace / citations
- 适合日常演示、答辩、开发联调

### 9.2 strict 验证环境

依赖：

```bash
pip install -r requirements-bge-strict.txt
```

目标：

- 更稳定地验证 `FlagEmbedding`
- 更稳定地验证 `BGE-M3`
- 更稳定地验证 `bge_reranker`
- 避免 `transformers` 版本兼容问题影响重模型验证

### 9.3 项目级轻量初始化配置

strict 环境下可使用：

- `KBQA_HEALTH_INIT_LIMIT`
- `KBQA_ASK_INIT_LIMIT`

含义：

- `/health` 可用小规模初始化验证项目可启动性
- 第一次 `/ask` 可用小规模初始化做项目级问答验证
- 默认环境不受影响

推荐 strict 环境参数：

- `KBQA_HEALTH_INIT_LIMIT=50`
- `KBQA_ASK_INIT_LIMIT=80`

---

## 10. 安装与运行

### 10.1 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

### 10.2 安装默认依赖

```bash
pip install -r requirements.txt
```

### 10.3 启动 FastAPI（建议端口 8001）

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### 10.4 打开 API 文档

- [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 10.5 启动 Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 11. 常用请求示例

### 11.1 健康检查

```bash
curl http://127.0.0.1:8001/health
```

### 11.2 提问（`/ask`）

```bash
curl -X POST "http://127.0.0.1:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"老师在作业批改台里找不到作业发布入口，是入口改版了吗？","top_k":8}'
```

Windows PowerShell：

```powershell
curl -X POST "http://127.0.0.1:8001/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"老师在作业批改台里找不到作业发布入口，是入口改版了吗？\",\"top_k\":8}"
```

### 11.3 检索证据（`/search`）

```bash
curl -X POST "http://127.0.0.1:8001/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"课堂演示遇到 NameError，应该怎么给学生解释？","top_k":3,"filters":{},"request_id":"demo-search-001"}'
```

Python `requests` 示例（供项目一远程调用）：

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8001/search",
    json={
        "query": "课堂演示遇到 NameError，应该怎么给学生解释？",
        "top_k": 3,
        "filters": {},
        "request_id": "project1-remote-rag",
    },
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
hits = data["hits"]
```

如果你已启动本项目服务，也可以运行脚本快速验证：

```bash
python -m scripts.test_search_api --base-url http://127.0.0.1:8001 --top-k 3
```

---

## 12. 推荐测试问题

### 平台使用类

`班级开课后学生端一直显示“未开始”，老师需要在哪里确认开课？如果课堂已经创建过，应该优先检查哪些入口或状态？`

### 代码报错类

`我在课堂演示 Python for 循环时，学生总因为缩进写错报 IndentationError。请给一个正确示例，并解释最常见的错误原因，适合怎么在课堂上讲。`

### 课程设计类

`算法入门这节课的导入环节有点生硬，我想自然过渡到循环嵌套，能不能给一个更适合课堂讲解的案例，并说明讲评时应该怎么分步骤引导学生？`

---

## 13. 脚本清单与用途说明

### 13.1 演示脚本

#### `run_demo`

快速命令行演示 `/ask`。

```bash
python -m scripts.run_demo
```

#### `run_demo_suite`

读取固定问题集并输出验收报告。

```bash
python -m scripts.run_demo_suite
```

如果已启动 API，也可以走 HTTP：

```bash
python -m scripts.run_demo_suite --http-base http://127.0.0.1:8001
```

### 13.2 路由/检索评测脚本

#### 路由评测

```bash
python -m scripts.eval_router
```

#### 检索评测

```bash
python -m scripts.eval_retrieval --k 5
```

#### 汇总评测

```bash
python -m scripts.eval_summary --passes 3 --export both --outdir outputs/eval
```

### 13.3 FAQ 候选沉淀脚本

从工单中抽取高频问题。

```bash
python -m scripts.build_faq_candidates --min-frequency 3
```

### 13.4 原始资料解析脚本

把多格式原始资料解析成统一 document jsonl。

```bash
python -m scripts.parse_raw_sources --input_dir docs/samples --output outputs/parsed_documents.jsonl
```

### 13.5 数据接入/建库脚本

```bash
python -m scripts.ingest_data --limit 300
```

---

## 14. strict 验证脚本

### 14.1 依赖栈验证

验证 `FlagEmbedding + transformers + BGE` 依赖是否兼容：

```bash
python -m scripts.validate_bge_stack
```

### 14.2 BGE-M3 小规模 GPU 验证

只验证 embedding + 小索引构建，不走完整问答主链路：

```bash
python -m scripts.validate_small_kb_gpu --limit 300 --batch-size 4
```

### 14.3 小规模检索可视化

验证 BGE-M3 检索效果，并可输出 t-SNE 图：

```bash
python -m scripts.inspect_bge_retrieval --query-set education --limit 300 --top-k 5
```

可选输出图：

```bash
python -m scripts.inspect_bge_retrieval --query-set education --limit 300 --top-k 5 --plot artifacts/bge_neighbors_tsne.png
```

### 14.4 bge_reranker 严格验证

验证：

1. 初始化
2. query-passage 打分
3. rerank 前后顺序变化

```bash
python -m scripts.inspect_bge_rerank --limit 300 --vec-top-k 8 --rerank-top-k 6 --no-fallback
```

---

## 15. Streamlit 面试演示推荐顺序

推荐按以下顺序演示：

1. **FAQ 命中**
2. **平台使用（RAG）**
3. **课程设计（RAG）**
4. **代码报错（RAG + code_example）**
5. **need_clarify**
6. **多意图 subquery**

演示时重点让面试官看：

- `answer`
- `citations`
- `route_trace`
- `debug.timing_ms`

这样能清楚展示：

- 系统不是黑盒
- 每个回答都有证据支持
- 不同 query 会走不同策略链路

---

## 16. 项目亮点

### 16.1 工程亮点

- 配置集中管理
- 数据结构统一
- 多模块边界清晰
- 默认环境可跑、strict 环境可验证
- 前后端链路完整

### 16.2 检索/RAG 亮点

- FAQ 优先 + RAG 兜底
- Hybrid 检索
- Rerank 抽象
- HyDE / Subquery / Backtrack
- 证据过滤
- 证据约束生成

### 16.3 产品亮点

- need_clarify 先澄清
- 结构化证据链输出
- Streamlit 演示页适合答辩
- 项目验证总结页适合截图与汇报

---

## 17. 已完成的验证能力

当前项目已经完成并可演示的验证包括：

- FAQ / BM25 主链路已跑通
- `/ask` 主链路已跑通
- Streamlit 演示页已跑通
- BGE-M3 strict 小规模验证
- bge_reranker strict 验证
- 小规模 retrieval 可视化
- rerank 前后顺序对比验证
- `eval_router / eval_retrieval / eval_summary` 最小评测

这意味着项目不仅“实现了功能”，还具备了“工程验证闭环”。

---

## 18. 面试时可以怎么讲这个项目

建议按以下顺序讲：

1. **业务背景**：教育场景知识分散，问题类型多样
2. **方案设计**：FAQ 优先，RAG 兜底，Router 做策略分流
3. **工程实现**：DataLoader / Cleaner / Chunker / Retriever / QAService / Streamlit
4. **可解释性**：citations + route_trace + debug
5. **验证能力**：BGE-M3 strict、bge_reranker strict、评测脚本
6. **扩展位**：Milvus / Redis / OCR / 更强路由与评测

---

## 19. 后续扩展位

当前项目已经预留了比较明确的扩展方向：

### 19.1 Milvus

替换当前进程内 FAISS，支持更正式的向量库部署。

### 19.2 Redis

替换当前 TTL memory cache，支持跨进程缓存与更稳定的线上热数据加速。

### 19.3 OCR 与多格式解析

接入扫描件 / 图片 / PDF 等更真实的企业资料源。

### 19.4 更强 rerank

接入 cross-encoder 或更强的重排模型，提升复杂 query 的精排质量。

### 19.5 更强路由

从规则路由升级到轻量分类器/策略模型。

### 19.6 更系统评测

增加 gold 标注和更标准的 Recall / MRR / nDCG / answer quality 评测。

---

## 20. 相关文档

- 面试答辩文档：`docs/interview_project2_overview.md`
- 输入 schema 设计：`input_schema_design.md`

---

## 21. 一句话总结

> 这是一个已经具备完整主链路、严格验证能力、可解释输出和面试演示能力的教师知识库问答 RAG 项目。

它不是单纯的“模型调用 demo”，而是一个具备**工程化结构、验证闭环和产品化展示能力**的项目。
