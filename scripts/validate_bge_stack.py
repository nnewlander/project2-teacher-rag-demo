"""
新环境「BGE 严格栈」自检：transformers 版本、FlagEmbedding 导入、BGE-M3 与 FlagReranker 能否初始化。

用法（在已激活的 conda 环境中）：
  python -m scripts.validate_bge_stack
  python -m scripts.validate_bge_stack --imports-only
  python -m scripts.validate_bge_stack --skip-reranker
  python -m scripts.validate_bge_stack --skip-m3

退出码：0 全部通过，1 有失败项。
"""

from __future__ import annotations

import argparse
import sys


def _ver(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "?"


def _parse_major_minor(s: str) -> tuple[int, int]:
    parts = (s or "").split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 FlagEmbedding + BGE-M3 + bge_reranker 依赖栈")
    ap.add_argument("--imports-only", action="store_true", help="仅检查 import 与版本，不加载大模型")
    ap.add_argument("--skip-m3", action="store_true", help="跳过 BGE-M3 初始化与编码")
    ap.add_argument("--skip-reranker", action="store_true", help="跳过 FlagReranker 初始化与打分")
    ap.add_argument("--m3-model", default="BAAI/bge-m3", help="BGE-M3 模型名或本地路径")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3", help="FlagReranker 模型名或本地路径")
    args = ap.parse_args()

    ok = True
    print("=== [a] transformers 版本（应 < 5.0 以规避已知 FlagEmbedding 兼容问题）===")
    try:
        import transformers

        tv = transformers.__version__
        print(f"transformers {tv}")
        maj, mino = _parse_major_minor(tv)
        if maj >= 5:
            print("FAIL: transformers>=5.0，建议重装 requirements-bge-strict.txt 中的上界约束。")
            ok = False
        else:
            print("OK")
    except Exception as e:
        print(f"FAIL: {e!r}")
        ok = False

    print("\n=== FlagEmbedding 可导入性 ===")
    try:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker  # noqa: F401

        print(f"FlagEmbedding {_ver('FlagEmbedding')}")
        print("OK")
    except Exception as e:
        print(f"FAIL: {e!r}")
        ok = False

    if args.imports_only:
        print("\n(--imports-only) 跳过后续模型加载。")
        return 0 if ok else 1

    if not args.skip_m3:
        print(f"\n=== [a 续] BGE-M3 初始化 + 短句编码（{args.m3_model!r}）===")
        try:
            from FlagEmbedding import BGEM3FlagModel

            m3 = BGEM3FlagModel(args.m3_model, use_fp16=False)
            out = m3.encode(["严格验证环境短句测试。"], batch_size=1, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            if isinstance(out, dict):
                dv = out.get("dense_vecs")
                assert dv is not None, "dense_vecs 缺失"
            print("OK")
        except TypeError:
            try:
                from FlagEmbedding import BGEM3FlagModel

                m3 = BGEM3FlagModel(args.m3_model, use_fp16=False)
                m3.encode(["严格验证环境短句测试。"], batch_size=1, max_length=512)
                print("OK (compat encode kwargs)")
            except Exception as e2:
                print(f"FAIL: {e2!r}")
                ok = False
        except Exception as e:
            print(f"FAIL: {e!r}")
            ok = False
    else:
        print("\n(--skip-m3) 跳过 BGE-M3。")

    if not args.skip_reranker:
        print(f"\n=== [b] FlagReranker 初始化 + compute_score（{args.rerank_model!r}）===")
        try:
            from FlagEmbedding import FlagReranker

            rr = FlagReranker(args.rerank_model, use_fp16=False)
            pairs = [["测试 query", "测试 passage 文本片段。"]]
            try:
                s = rr.compute_score(pairs, batch_size=1)
            except TypeError:
                s = rr.compute_score(pairs)
            print(f"score sample: {s!r}")
            print("OK")
        except Exception as e:
            print(f"FAIL: {e!r}")
            ok = False
    else:
        print("\n(--skip-reranker) 跳过 FlagReranker。")

    if not args.skip_m3 and not args.skip_reranker and ok:
        print("\n=== [c] 小结：BGE-M3 与 bge_reranker 在本轮均已尝试加载 ===")

    print("\n" + ("全部通过。" if ok else "存在失败项，请检查依赖版本与网络/缓存模型。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
