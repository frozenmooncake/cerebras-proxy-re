import hashlib
import time
from typing import Optional

import httpx


async def admit_fixed_window(
    client: httpx.AsyncClient,
    redis_url: str,
    redis_token: str,
    namespace: str,
    identity: str,
    bucket: str,
    limit: int,
) -> Optional[bool]:
    if not redis_url or not redis_token:
        return True
    minute = int(time.time() // 60)
    digest = hashlib.sha256(
        (namespace + "\0" + identity + "\0" + bucket).encode("utf-8")
    ).hexdigest()
    key = f"cpr:upstream:{digest}:{minute}"
    try:
        response = await client.post(
            redis_url.rstrip("/") + "/pipeline",
            json=[["INCR", key], ["EXPIRE", key, 60, "NX"]],
            headers={"Authorization": "Bearer " + redis_token},
            timeout=3.0,
        )
        response.raise_for_status()
        data = response.json()
        value = data[0].get("result") if data and isinstance(data[0], dict) else None
        return int(value) <= limit
    except Exception:
        return None
