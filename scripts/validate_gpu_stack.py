"""
GPU/CPU 自动选择验证：
- torch 版本与 cuda 可用性
- Settings.device / Settings.use_fp16 的解析结果
- BGE-M3 是否能按当前配置初始化（并在可能情况下打印实际 device/fp16）

用法：
  python -m scripts.validate_gpu_stack

可选：
  $env:KBQA_DEVICE='auto'|'cpu'|'cuda'
  $env:KBQA_USE_FP16='auto'|'true'|'false'
  $env:KBQA_EMBEDDING_BACKEND='bge_m3'
  $env:KBQA_EMBEDDING_MODEL_NAME='BAAI/bge-m3'
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=== torch / cuda ===")
    try:
        import torch  # type: ignore

        print("torch", getattr(torch, "__version__", "?"))
        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        print("cuda.is_available()", cuda_ok)
        if cuda_ok:
            try:
                print("cuda.device_count()", torch.cuda.device_count())
                print("cuda.current_device()", torch.cuda.current_device())
                print("cuda.get_device_name(0)", torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception as e:
        print("torch import FAIL:", repr(e))
        return 1

    print("\n=== Settings ===")
    from app.config import Settings

    s = Settings()
    print("KBQA_DEVICE ->", s.device)
    print("KBQA_USE_FP16 ->", s.use_fp16)
    print("embedding_backend ->", s.embedding_backend)
    print("embedding_model_name ->", s.embedding_model_name)
    print("embedding_batch_size ->", s.embedding_batch_size)

    if str(s.embedding_backend).strip().lower() != "bge_m3":
        print("\nSKIP: embedding_backend != bge_m3")
        return 0

    print("\n=== BGE-M3 init ===")
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
    except Exception as e:
        print("FlagEmbedding import FAIL:", repr(e))
        return 1

    # 复用与服务一致的选择逻辑：尽量模拟实际初始化参数（不依赖业务主链路）
    try:
        req_device = (s.device or "auto").strip().lower()
        req_fp16 = (s.use_fp16 or "auto").strip().lower()

        cuda_ok = bool(torch.cuda.is_available())
        if req_device in ("", "auto"):
            device = "cuda" if cuda_ok else "cpu"
        else:
            device = req_device

        if req_fp16 in ("", "auto"):
            fp16 = device == "cuda"
        else:
            fp16 = req_fp16 in ("1", "true", "yes", "y", "on")

        print("picked_device =", device)
        print("picked_use_fp16 =", fp16)

        try:
            m3 = BGEM3FlagModel(s.embedding_model_name, device=device, use_fp16=fp16)
        except TypeError:
            m3 = BGEM3FlagModel(s.embedding_model_name, use_fp16=fp16)

        # 轻量 encode，验证能跑通
        try:
            out = m3.encode(
                ["GPU stack smoke test."],
                batch_size=1,
                max_length=128,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            out = m3.encode(["GPU stack smoke test."], batch_size=1, max_length=128)

        ok = False
        if isinstance(out, dict):
            ok = out.get("dense_vecs") is not None
        else:
            ok = out is not None
        print("encode_ok =", ok)
        return 0 if ok else 1
    except Exception as e:
        print("BGE-M3 init/encode FAIL:", repr(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

