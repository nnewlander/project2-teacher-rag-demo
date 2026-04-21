from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Tuple, TypeVar


if TYPE_CHECKING:
    from app.config import Settings

T = TypeVar("T")


def normalize_cache_backend(raw: str) -> str:
    x = (raw or "").strip().lower().replace("-", "_")
    if x in ("ttl_memory_cache", "memory", "ttl", "in_process", "local"):
        return "ttl_memory_cache"
    if x in ("redis", "redis_cache"):
        return "redis"
    return x


def create_answer_cache(settings: "Settings") -> "TTLCache[Any]":
    """
    根据 `Settings.cache_backend` 构造缓存实现；默认进程内 TTL，与项目二「可替换为 Redis」口径对齐。
    """
    backend = normalize_cache_backend(str(getattr(settings, "cache_backend", "ttl_memory_cache") or ""))
    if backend == "ttl_memory_cache":
        return TTLCache[Any](
            max_size=int(getattr(settings, "cache_max_size", 512)),
            ttl_s=int(getattr(settings, "cache_ttl_s", 300)),
        )
    if backend == "redis":
        raise RuntimeError(
            "cache_backend=redis 尚未接入（需 redis-py 与连接串配置等）。"
            "请保持 cache_backend=ttl_memory_cache，或实现 Redis 适配后再切换。"
        )
    raise RuntimeError(
        f"不支持的 cache_backend={getattr(settings, 'cache_backend', None)!r}（规范化后={backend!r}）。"
    )


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evicts: int = 0


class TTLCache(Generic[T]):
    """
    ttl_memory_cache 默认实现：进程内 TTL + 简单淘汰（不依赖 Redis）。

    与 `Settings.cache_backend=ttl_memory_cache` 对应；未来可并列实现 `RedisAnswerCache` 等。

    TODO:
    - 更严格的 LRU
    - 可选落盘缓存
    """

    def __init__(self, *, max_size: int = 1024, ttl_s: int = 300) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_s = max(1, int(ttl_s))
        self._store: Dict[str, Tuple[float, float, T]] = {}  # key -> (created_at, expires_at, value)
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[T]:
        now = time.time()
        item = self._store.get(key)
        if not item:
            self.stats.misses += 1
            return None
        created_at, expires_at, value = item
        if expires_at <= now:
            self._store.pop(key, None)
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, key: str, value: T) -> None:
        now = time.time()
        if len(self._store) >= self.max_size:
            self._evict_one(now)
        self._store[key] = (now, now + self.ttl_s, value)
        self.stats.sets += 1

    def _evict_one(self, now: float) -> None:
        # 先清理过期，再淘汰一个最早创建的
        expired = [k for k, (_, exp, _) in self._store.items() if exp <= now]
        for k in expired:
            self._store.pop(k, None)
        if len(self._store) < self.max_size:
            return
        oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
        self._store.pop(oldest_key, None)
        self.stats.evicts += 1

