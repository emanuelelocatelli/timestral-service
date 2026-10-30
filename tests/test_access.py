from starlette.requests import Request

from config import get_settings
from database import (
    add_allowed_ip,
    ensure_plain_token,
    init_database,
    is_ip_allowed,
    reset_connection,
    validate_api_token,
)
from security import admin_credentials_match, resolve_client_ip


def _request(peer: str, forwarded: str = "") -> Request:
    headers = []
    if forwarded:
        headers.append((b"x-forwarded-for", forwarded.encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/predict",
            "raw_path": b"/predict",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 40000),
            "server": ("timestral", 8003),
        }
    )


def test_empty_whitelist_admits_everyone(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "access.db"))
    get_settings.cache_clear()
    reset_connection()
    init_database()

    assert is_ip_allowed("203.0.113.8") is True


def test_whitelist_and_imported_token(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "access.db"))
    get_settings.cache_clear()
    reset_connection()
    init_database()
    add_allowed_ip("46.225.85.147", "srv1")
    ensure_plain_token("price-candles", "secret-bearer")

    assert is_ip_allowed("46.225.85.147") is True
    assert is_ip_allowed("198.51.100.4") is False
    assert validate_api_token("secret-bearer") == "price-candles"
    assert validate_api_token("other") is None


def test_forwarded_for_is_trusted_only_from_the_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "access.db"))
    monkeypatch.setenv("TRUSTED_PROXIES", "172.18.0.2/32")
    get_settings.cache_clear()

    assert resolve_client_ip(_request("172.18.0.2", "1.2.3.4, 46.225.85.147")) == "46.225.85.147"
    assert resolve_client_ip(_request("10.9.9.9", "46.225.85.147")) == "10.9.9.9"


def test_default_admin_password_cannot_log_in(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me")
    get_settings.cache_clear()
    assert admin_credentials_match("emanuele.locatelli@hotmail.it", "change-me") is False
