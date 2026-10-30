import hashlib
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import get_settings

_local = threading.local()

# Same callers already admitted by the Chronos whitelist on srv2.
# srv1 is the host that runs price_candles.
DEFAULT_ALLOWED_IPS = (
    ("138.68.109.57", "WAO"),
    ("77.42.17.240", "SRV2"),
    ("46.225.85.147", "srv1"),
)


def reset_connection() -> None:
    connection = getattr(_local, "connection", None)
    if connection is not None:
        connection.close()
    _local.connection = None


def _get_connection() -> sqlite3.Connection:
    if getattr(_local, "connection", None) is None:
        settings = get_settings()
        directory = os.path.dirname(settings.database_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        connection = sqlite3.connect(settings.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        _local.connection = connection
    return _local.connection


def init_database() -> None:
    connection = _get_connection()
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS allowed_ips (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address  TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS api_tokens (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            token_hash   TEXT NOT NULL UNIQUE,
            token_prefix TEXT NOT NULL,
            is_active    INTEGER NOT NULL DEFAULT 1,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    connection.commit()


def get_allowed_ips() -> List[Dict[str, Any]]:
    rows = _get_connection().execute("SELECT * FROM allowed_ips ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def add_allowed_ip(ip_address: str, description: str = "") -> Dict[str, Any]:
    connection = _get_connection()
    ip_address = ip_address.strip()
    try:
        cursor = connection.execute(
            "INSERT INTO allowed_ips (ip_address, description) VALUES (?, ?)",
            (ip_address, description),
        )
        connection.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError(f"IP '{ip_address}' already exists") from error
    row = connection.execute("SELECT * FROM allowed_ips WHERE id=?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def delete_allowed_ip(ip_id: int) -> bool:
    cursor = _get_connection().execute("DELETE FROM allowed_ips WHERE id=?", (ip_id,))
    _get_connection().commit()
    return cursor.rowcount > 0


def is_ip_allowed(ip_address: str) -> bool:
    """Empty whitelist admits every caller. A non-empty list admits only those rows."""
    connection = _get_connection()
    count = connection.execute("SELECT COUNT(*) FROM allowed_ips").fetchone()[0]
    if count == 0:
        return True
    row = connection.execute(
        "SELECT 1 FROM allowed_ips WHERE ip_address=? OR ip_address='0.0.0.0'",
        (ip_address,),
    ).fetchone()
    return row is not None


def seed_default_ips() -> None:
    connection = _get_connection()
    count = connection.execute("SELECT COUNT(*) FROM allowed_ips").fetchone()[0]
    if count:
        return
    connection.executemany(
        "INSERT INTO allowed_ips (ip_address, description) VALUES (?, ?)",
        DEFAULT_ALLOWED_IPS,
    )
    connection.commit()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def get_api_tokens() -> List[Dict[str, Any]]:
    rows = _get_connection().execute(
        "SELECT id, name, token_prefix, is_active, created_at FROM api_tokens ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def create_api_token(name: str) -> Dict[str, Any]:
    plain_token = secrets.token_hex(32)
    token_prefix = plain_token[:8] + "..."
    cursor = _get_connection().execute(
        "INSERT INTO api_tokens (name, token_hash, token_prefix) VALUES (?, ?, ?)",
        (name, _hash_token(plain_token), token_prefix),
    )
    _get_connection().commit()
    return {
        "id": cursor.lastrowid,
        "name": name,
        "token": plain_token,
        "token_prefix": token_prefix,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def ensure_plain_token(name: str, plain_token: str) -> None:
    """Keep a bearer that already lives outside the database, such as the env token price_candles uses."""
    plain_token = (plain_token or "").strip()
    if not plain_token:
        return
    digest = _hash_token(plain_token)
    connection = _get_connection()
    existing = connection.execute("SELECT id FROM api_tokens WHERE token_hash=?", (digest,)).fetchone()
    if existing:
        return
    connection.execute(
        "INSERT INTO api_tokens (name, token_hash, token_prefix) VALUES (?, ?, ?)",
        (name, digest, plain_token[:8] + "..."),
    )
    connection.commit()


def toggle_api_token(token_id: int, is_active: bool) -> bool:
    cursor = _get_connection().execute(
        "UPDATE api_tokens SET is_active=? WHERE id=?",
        (int(is_active), token_id),
    )
    _get_connection().commit()
    return cursor.rowcount > 0


def delete_api_token(token_id: int) -> bool:
    cursor = _get_connection().execute("DELETE FROM api_tokens WHERE id=?", (token_id,))
    _get_connection().commit()
    return cursor.rowcount > 0


def validate_api_token(token: str) -> Optional[str]:
    row = _get_connection().execute(
        "SELECT name FROM api_tokens WHERE token_hash=? AND is_active=1",
        (_hash_token(token),),
    ).fetchone()
    return row["name"] if row else None


def has_any_api_tokens() -> bool:
    count = _get_connection().execute("SELECT COUNT(*) FROM api_tokens WHERE is_active=1").fetchone()[0]
    return count > 0
