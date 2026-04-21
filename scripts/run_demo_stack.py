"""
面试演示一键启动 / 预热 / 就绪检查（不改 API 协议，不改 QAService 主逻辑）。

能力：
- 检查关键配置与数据文件是否存在（通过 Settings 读取 env/.env）
- 可选启动 FastAPI（uvicorn）与 Streamlit
- 轮询 /health 探活
- 发送 1~2 条 /ask 进行预热（触发 init_kb / embedding / index build）
- （可选）探活 Streamlit 首页
- 输出 READY / FAIL 与推荐打开的页面

用法（推荐）：
  # 1) 一键启动 + 预热 + 检查
  python -m scripts.run_demo_stack --start-api --start-streamlit --prewarm

  # 2) 只检查（不启动进程）
  python -m scripts.run_demo_stack --check-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_WARMUP_QUERIES: Tuple[str, ...] = (
    "老师在作业批改台里找不到作业发布入口，是入口改版了吗？",
    "我在算法入门课堂演示 for 循环，学生总写错缩进并报 IndentationError，想要一个正确示例代码并解释常见原因与讲法。",
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    title: str
    detail: Dict[str, Any]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _fmt_bool(x: bool) -> str:
    return "OK" if x else "FAIL"


def _safe_request(method: str, url: str, *, json_data: Optional[dict] = None, timeout_s: float = 30.0) -> Tuple[bool, Any]:
    try:
        import httpx

        with httpx.Client(timeout=timeout_s, trust_env=False) as client:
            if method.upper() == "GET":
                r = client.get(url)
            else:
                r = client.request(method.upper(), url, json=json_data)
        return True, r
    except Exception as e:
        return False, e


def _print_block(title: str, items: List[CheckResult]) -> bool:
    print("\n" + title)
    all_ok = True
    for it in items:
        all_ok = all_ok and bool(it.ok)
        print(f"- [{_fmt_bool(it.ok)}] {it.title}")
        if it.detail:
            for k, v in it.detail.items():
                print(f"    - {k}: {v}")
    return all_ok


def _check_settings() -> List[CheckResult]:
    from app.config import Settings

    s = Settings()
    out: List[CheckResult] = []

    # 数据文件存在性
    paths = {
        "raw_documents_path": Path(s.raw_documents_path),
        "raw_support_tickets_path": Path(s.raw_support_tickets_path),
        "raw_faq_path": Path(s.raw_faq_path),
        "raw_code_examples_path": Path(getattr(s, "raw_code_examples_path")),
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    out.append(
        CheckResult(
            ok=len(missing) == 0,
            title="数据文件路径可用（document / support_ticket / faq / code_example）",
            detail={k: str(v) for k, v in paths.items()} | ({"missing": ", ".join(missing)} if missing else {}),
        )
    )

    # LLM 配置：不强制，但如果启用就检查齐全
    llm_provider = (s.llm_provider or "").strip()
    llm_enabled = llm_provider.lower() == "openai_compatible" and bool((s.llm_model_name or "").strip())
    if llm_enabled:
        ok = bool((s.llm_base_url or "").strip())
        out.append(
            CheckResult(
                ok=ok,
                title="LLM 已启用（openai_compatible）且 base_url/model_name 配置齐全",
                detail={"llm_provider": s.llm_provider, "llm_base_url": s.llm_base_url, "llm_model_name": s.llm_model_name},
            )
        )
    else:
        out.append(
            CheckResult(
                ok=True,
                title="LLM 未启用（允许：将走 fallback 模板答案）",
                detail={"llm_provider": s.llm_provider, "llm_model_name": s.llm_model_name},
            )
        )

    # Embedding/Rerank 仅做展示，不强制（strict 环境可通过脚本单独验证）
    out.append(
        CheckResult(
            ok=True,
            title="Embedding / Rerank 配置（展示当前 effective）",
            detail={
                "device": getattr(s, "device", None),
                "use_fp16": getattr(s, "use_fp16", None),
                "embedding_backend": getattr(s, "embedding_backend", None),
                "embedding_model_name": getattr(s, "embedding_model_name", None),
                "reranker_backend": getattr(s, "reranker_backend", None),
                "reranker_model_name": getattr(s, "reranker_model_name", None),
            },
        )
    )
    return out


def _start_process(cmd: List[str], *, name: str) -> subprocess.Popen:
    # Windows：尽量在新进程组中启动，避免 Ctrl+C 互相影响
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(cmd, creationflags=creationflags)


def _poll_api_health(api_base: str, timeout_s: float) -> Tuple[bool, Dict[str, Any]]:
    url = api_base.rstrip("/") + "/health"
    t0 = time.time()
    last: Any = None
    while time.time() - t0 < timeout_s:
        ok, r = _safe_request("GET", url, timeout_s=10.0)
        if ok and hasattr(r, "status_code") and int(getattr(r, "status_code")) == 200:
            try:
                j = r.json()
            except Exception:
                j = {"raw": getattr(r, "text", "")[:400]}
            return True, {"url": url, "status_code": r.status_code, "json": j}
        last = r
        time.sleep(0.8)
    return False, {"url": url, "error": repr(last) if last is not None else "timeout"}


def _prewarm_ask(api_base: str, queries: Sequence[str], top_k: int) -> Tuple[bool, Dict[str, Any]]:
    url = api_base.rstrip("/") + "/ask"
    details: Dict[str, Any] = {"url": url, "queries": []}
    ok_all = True
    for q in queries:
        t0 = _now_ms()
        ok, r = _safe_request("POST", url, json_data={"query": q, "top_k": int(top_k)}, timeout_s=180.0)
        if not ok:
            ok_all = False
            details["queries"].append({"query": q[:80], "ok": False, "error": repr(r)})
            continue
        sc = int(getattr(r, "status_code", 0) or 0)
        if sc != 200:
            ok_all = False
            details["queries"].append({"query": q[:80], "ok": False, "status_code": sc, "text": getattr(r, "text", "")[:400]})
            continue
        try:
            j = r.json()
        except Exception:
            j = {}
        timing = {}
        dbg = j.get("debug") if isinstance(j.get("debug"), dict) else {}
        if isinstance(dbg.get("timing_ms"), dict):
            timing = dbg.get("timing_ms")
        details["queries"].append(
            {
                "query": q[:80],
                "ok": True,
                "route": j.get("route"),
                "mode": j.get("mode"),
                "sources": _dedup_list([str(x.get("source_type") or "") for x in (j.get("citations") or []) if isinstance(x, dict)]),
                "wall_ms(client)": round(_now_ms() - t0, 2),
                "timing_ms(debug)": timing,
            }
        )
    return ok_all, details


def _dedup_list(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _check_streamlit(streamlit_base: str) -> Tuple[bool, Dict[str, Any]]:
    # Streamlit 根路径返回 HTML；这里只做可达性检查
    ok, r = _safe_request("GET", streamlit_base, timeout_s=8.0)
    if not ok:
        return False, {"url": streamlit_base, "error": repr(r)}
    sc = int(getattr(r, "status_code", 0) or 0)
    good = sc in (200, 301, 302, 307, 308)
    return good, {"url": streamlit_base, "status_code": sc}


def main() -> int:
    ap = argparse.ArgumentParser(description="Interview demo: start/prewarm/check API + Streamlit.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-port", type=int, default=8001)
    ap.add_argument("--streamlit-port", type=int, default=8501)
    ap.add_argument("--check-only", action="store_true", help="只检查，不启动进程，不预热")
    ap.add_argument("--start-api", action="store_true", help="启动 uvicorn API")
    ap.add_argument("--start-streamlit", action="store_true", help="启动 streamlit")
    ap.add_argument("--prewarm", action="store_true", help="对 1~2 条 query 预热 /ask")
    ap.add_argument("--top-k", type=int, default=8, help="预热 /ask 传入 top_k")
    ap.add_argument("--health-timeout-s", type=float, default=60.0, help="/health 探活超时")
    ap.add_argument("--warmup-queries", type=int, default=2, help="预热 query 条数（默认 2）")
    args = ap.parse_args()

    api_base = f"http://{args.host}:{int(args.api_port)}"
    streamlit_base = f"http://{args.host}:{int(args.streamlit_port)}"

    # 1) 配置检查（读取 env/.env 的 effective settings）
    cfg_results = _check_settings()
    cfg_ok = _print_block("=== [A] 配置与数据检查 ===", cfg_results)

    if args.check_only:
        # check-only 也检查可用性（如果服务已启动）
        items: List[CheckResult] = []
        ok_api, health_detail = _poll_api_health(api_base, timeout_s=5.0)
        items.append(CheckResult(ok=ok_api, title="API 可用（/health）", detail=health_detail))
        ok_st, st_detail = _check_streamlit(streamlit_base)
        items.append(CheckResult(ok=ok_st, title="Streamlit 可达（根路径）", detail=st_detail))
        all_ok = cfg_ok and _print_block("=== [B] 服务可达性（check-only） ===", items)
        print("\nRESULT:", "READY" if all_ok else "FAIL")
        if not all_ok:
            print("提示：如需一键拉起：python -m scripts.run_demo_stack --start-api --start-streamlit --prewarm")
        return 0 if all_ok else 1

    procs: List[Tuple[str, subprocess.Popen]] = []
    try:
        # 2) 启动 API / Streamlit（可选）
        if args.start_api:
            cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(int(args.api_port)), "--reload"]
            p = _start_process(cmd, name="api")
            procs.append(("api", p))
            print(f"\n[start] API: {api_base}  (pid={p.pid})")

        if args.start_streamlit:
            cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(int(args.streamlit_port))]
            p = _start_process(cmd, name="streamlit")
            procs.append(("streamlit", p))
            print(f"[start] Streamlit: {streamlit_base}  (pid={p.pid})")

        # 3) 探活 API
        ok_api, health_detail = _poll_api_health(api_base, timeout_s=float(args.health_timeout_s))
        svc_items = [CheckResult(ok=ok_api, title="API 可用（/health）", detail=health_detail)]

        # 4) （可选）探活 Streamlit
        ok_st, st_detail = _check_streamlit(streamlit_base)
        svc_items.append(CheckResult(ok=ok_st, title="Streamlit 可达（根路径）", detail=st_detail))

        svc_ok = _print_block("=== [B] 服务探活 ===", svc_items)

        # 5) 预热
        warm_ok = True
        warm_detail: Dict[str, Any] = {}
        if args.prewarm and ok_api:
            qs = list(DEFAULT_WARMUP_QUERIES)[: max(1, int(args.warmup_queries))]
            warm_ok, warm_detail = _prewarm_ask(api_base, qs, top_k=int(args.top_k))
            _print_block("=== [C] 预热 /ask（触发 init_kb / embedding / 索引） ===", [CheckResult(ok=warm_ok, title="prewarm /ask", detail=warm_detail)])

        all_ok = cfg_ok and svc_ok and (warm_ok if args.prewarm else True)
        print("\n=== [D] 面试前推荐打开 ===")
        print(f"- API health: {api_base}/health")
        print(f"- Streamlit:  {streamlit_base}")

        print("\nRESULT:", "READY" if all_ok else "FAIL")
        if not all_ok:
            print("\n排查建议：")
            print("- 先确认数据路径存在（见 [A]）")
            print("- 若 /health 不通：检查端口占用、防火墙、uvicorn 是否启动成功")
            print("- 若 /ask 预热失败：看响应 text/错误，或先单独访问 /health 触发 init_kb")
        return 0 if all_ok else 1
    finally:
        # 不自动杀进程：面试演示通常希望服务继续跑着
        pass


if __name__ == "__main__":
    raise SystemExit(main())

