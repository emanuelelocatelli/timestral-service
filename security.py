import hashlib
import hmac
import ipaddress
import logging

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings
from database import has_any_api_tokens, is_ip_allowed, validate_api_token

logger = logging.getLogger(__name__)

PROTECTED_API_PREFIXES = ("/predict", "/predict-batch")
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
SESSION_COOKIE = "timestral_session"


def trusted_proxy_networks() -> tuple:
    networks = []
    for entry in get_settings().trusted_proxies.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Ignoring unparseable trusted proxy entry: %s", entry)
    return tuple(networks)


def _is_trusted_proxy(ip: str) -> bool:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(address.version == network.version and address in network for network in trusted_proxy_networks())


def resolve_client_ip(request: Request) -> str:
    """Real caller when the socket peer is the reverse proxy.

    Traefik appends the address it saw, so the rightmost X-Forwarded-For entry
    is the client. The leftmost entry is whatever the caller wrote and is ignored.
    A peer outside trusted_proxies is judged as itself, so a neighbour on the
    docker network cannot pick an identity by forging the header.
    """
    peer = request.client.host if request.client else "unknown"
    if not _is_trusted_proxy(peer):
        return peer
    forwarded = request.headers.get("X-Forwarded-For", "")
    candidates = [part.strip() for part in forwarded.split(",") if part.strip()]
    return candidates[-1] if candidates else peer


def _same(left: str, right: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(left.encode()).hexdigest(),
        hashlib.sha256(right.encode()).hexdigest(),
    )


def create_session_token(email: str) -> str:
    payload = f"{email}:{get_settings().secret_key}"
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_session_token(token: str, email: str) -> bool:
    return hmac.compare_digest(token, create_session_token(email))


def admin_credentials_match(email: str, password: str) -> bool:
    settings = get_settings()
    configured = (settings.admin_password or "").strip()
    if not configured or configured == "change-me":
        return False
    return _same(email.strip(), settings.admin_email.strip()) and _same(password, configured)


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/static"):
            return await call_next(request)
        if any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_API_PREFIXES):
            return await self._check_api_auth(request, call_next)
        if path.startswith("/admin"):
            return await self._check_admin_auth(request, call_next, path)
        return await call_next(request)

    async def _check_api_auth(self, request: Request, call_next):
        client_ip = resolve_client_ip(request)
        if not is_ip_allowed(client_ip):
            logger.warning("API request from blocked IP: %s", client_ip)
            return JSONResponse(status_code=403, content={"detail": f"IP {client_ip} is not in the allowed list"})

        if has_any_api_tokens():
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header. Use: Bearer <token>"})
            token_name = validate_api_token(header[7:].strip())
            if token_name is None:
                logger.warning("Invalid API token from IP: %s", client_ip)
                return JSONResponse(status_code=401, content={"detail": "Invalid or revoked API token"})
            request.state.token_name = token_name
        else:
            request.state.token_name = None

        request.state.client_ip = client_ip
        return await call_next(request)

    async def _check_admin_auth(self, request: Request, call_next, path: str):
        if path in {"/admin/login", "/admin/api/login"}:
            return await call_next(request)
        session_token = request.cookies.get(SESSION_COOKIE)
        email = get_settings().admin_email
        if not session_token or not verify_session_token(session_token, email):
            if path.startswith("/admin/api/"):
                return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
            return RedirectResponse(url="/admin/login", status_code=302)
        return await call_next(request)
