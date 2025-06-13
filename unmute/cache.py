import logging
import time
from typing import Generic, Optional, TypeVar, cast

import redis
from redis.typing import EncodableT  # Import EncodableT for Redis compatibility

from unmute.kyutai_constants import REDIS_SERVER

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=EncodableT)  # Generic type bound to EncodableT


class CacheError(Exception):
    """An error happened while accessing the cache.

    This is so that we get the same exception type regardless of the cache implementation.
    """


class LocalCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 3600):  # Default 1 hour expiration
        self.cache: dict[
            str, tuple[T, float]
        ] = {}  # {key: (value, expiration_timestamp)}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[T]:
        cached = self.cache.get(key)
        if cached is not None:
            value, expiration = cached
            if time.time() < expiration:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        else:
            return None

    def set(self, key: str, value: T):
        expiration = time.time() + self.ttl_seconds
        self.cache[key] = (value, expiration)

    def delete(self, key: str):
        """Delete a key from the cache."""
        if key in self.cache:
            del self.cache[key]

    def cleanup(self):
        """Remove all expired entries"""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.cache.items() if exp < now]
        for k in expired_keys:
            del self.cache[k]


class RedisCache(Generic[T]):
    def __init__(self, redis_url: str, prefix: str, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self.redis_client = redis.Redis.from_url(redis_url, socket_connect_timeout=2)

    def get(self, key: str) -> Optional[T]:
        key = f"{self.prefix}:{key}"

        try:
            redis_value = self.redis_client.get(key)
            if redis_value is not None:
                logger.info(f"Retrieved value from Redis: {key}")
                return cast(T, redis_value)
            else:
                return None
        except redis.RedisError as e:
            raise CacheError(f"Failed to store value in Redis: {e}") from e

    def set(self, key: str, value: T):
        key = f"{self.prefix}:{key}"
        try:
            # Store with the TTL
            self.redis_client.setex(key, self.ttl_seconds, value)
        except redis.RedisError as e:
            raise CacheError(f"Failed to store value in Redis: {e}") from e

    def delete(self, key: str):
        key = f"{self.prefix}:{key}"
        try:
            # No error if the key does not exist.
            self.redis_client.delete(key)
        except redis.RedisError as e:
            raise CacheError(f"Failed to delete value from Redis: {e}") from e

    def cleanup(self):
        pass  # No cleanup needed for Redis


def get_cache(prefix: str, ttl_seconds: int) -> LocalCache[T] | RedisCache[T]:
    """
    Returns the appropriate cache based on the environment variables.
    If KYUTAI_REDIS_URL is set, it returns a RedisCache instance.
    If not, it returns a LocalCache instance.
    """
    if REDIS_SERVER is not None:
        cache = RedisCache[T](REDIS_SERVER, prefix, ttl_seconds=ttl_seconds)
    else:
        logger.info(
            "Redis cache address was not given in environment variables, using local cache."
        )
        cache = LocalCache[T](ttl_seconds=ttl_seconds)

    return cache
