from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from config import get_settings
from database import (
    add_allowed_ip,
    create_api_token,
    delete_allowed_ip,
    delete_api_token,
    get_allowed_ips,
    get_api_tokens,
    toggle_api_token,
)
from security import SESSION_COOKIE, admin_credentials_match, create_session_token

router = APIRouter()


class LoginBody(BaseModel):
    email: str
    password: str


class IpBody(BaseModel):
    ip_address: str = Field(..., min_length=3, max_length=64)
    description: str = ""


class TokenBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class TokenToggleBody(BaseModel):
    is_active: bool


def _page(name: str) -> HTMLResponse:
    with open(f"static/{name}", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


@router.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
def login_page() -> HTMLResponse:
    return _page("login.html")


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page() -> HTMLResponse:
    return _page("admin.html")


@router.post("/admin/api/login")
def login(body: LoginBody):
    if not admin_credentials_match(body.email, body.password):
        return JSONResponse(status_code=401, content={"detail": "Credenziali non valide"})
    response = JSONResponse({"ok": True})
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(get_settings().admin_email),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 12,
        path="/",
    )
    return response


@router.post("/admin/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.get("/admin/api/status")
def status(request: Request):
    started = getattr(request.app.state, "started_at", None)
    described = request.app.state.predictor.describe()
    return {
        "model_loaded": described["model_loaded"],
        "model": described["model_name"],
        "device": described["device"],
        "torch_num_threads": described["torch_num_threads"],
        "version": get_settings().service_version,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ") if started else None,
    }


@router.get("/admin/api/ips")
def list_ips():
    return get_allowed_ips()


@router.post("/admin/api/ips")
def create_ip(body: IpBody):
    try:
        return add_allowed_ip(body.ip_address, body.description)
    except ValueError as error:
        return JSONResponse(status_code=409, content={"detail": str(error)})


@router.delete("/admin/api/ips/{ip_id}")
def remove_ip(ip_id: int):
    if not delete_allowed_ip(ip_id):
        return JSONResponse(status_code=404, content={"detail": "IP non trovato"})
    return {"ok": True}


@router.get("/admin/api/tokens")
def list_tokens():
    return get_api_tokens()


@router.post("/admin/api/tokens")
def create_token(body: TokenBody):
    return create_api_token(body.name.strip())


@router.patch("/admin/api/tokens/{token_id}")
def update_token(token_id: int, body: TokenToggleBody):
    if not toggle_api_token(token_id, body.is_active):
        return JSONResponse(status_code=404, content={"detail": "Token non trovato"})
    return {"ok": True}


@router.delete("/admin/api/tokens/{token_id}")
def remove_token(token_id: int):
    if not delete_api_token(token_id):
        return JSONResponse(status_code=404, content={"detail": "Token non trovato"})
    return {"ok": True}
