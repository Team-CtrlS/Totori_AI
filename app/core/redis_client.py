import os
import redis.asyncio as redis

_redis_cilent: redis.Redis | None = None

async def connect():
    global _redis_client
    _redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

async def close():
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None

def get_redis() -> redis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis가 연결되지 않았습니다. connect()를 먼저 실행해주세요.")
    return _redis_client