from __future__ import annotations
 
import logging
from typing import Optional, Union
 
import aiohttp
from aiohttp import FormData
 
_TIMEOUT = aiohttp.ClientTimeout(total=300, sock_connect=30, sock_read=None)
 
async def _request_json(
    method: str,
    host: str,
    endpoint: str,
    *,
    data: Optional[Union[FormData, bytes]] = None,
) -> tuple[int | None, object]:
    url = f"http://{host}{endpoint}"
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.request(method.upper(), url, data=data) as resp:
                try:
                    payload = await resp.json()
                except Exception:
                    payload = await resp.text()
                return resp.status, payload
    except Exception as exc:
        logging.error("[%s] %s%s – %s", method.upper(), host, endpoint, exc)
        return None, str(exc)
 
 
async def remote_get(host: str, endpoint: str, headers=None, timeout=10):
    url = f"http://{host}{endpoint}" if not endpoint.startswith("http") else endpoint
    logging.info(f"🌐 [remote_get] URL: {url}")
    if headers:
        logging.info(f"🌐 [remote_get] Headers: {headers}")
    else:
        logging.info(f"🌐 [remote_get] No custom headers")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                logging.info(f"🌐 [remote_get] Status: {resp.status}")
                try:
                    data = await resp.json()
                    logging.info(f"🌐 [remote_get] JSON response: {data}")
                except Exception:
                    data = await resp.text()
                    logging.info(f"🌐 [remote_get] Text response: {data}")
                return resp.status, data
    except Exception as e:
        logging.error(f"🌐 [remote_get] Exception: {e}")
        return None, str(e)
 
 
async def remote_post_form(
    host: str,
    endpoint: str,
    form: FormData,
    *,
    method: str = "POST",
):
    return await _request_json(method, host, endpoint, data=form)