from functools import wraps

from flask import Blueprint, jsonify, g, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db

auth_bp = Blueprint("auth", __name__)


@auth_bp.before_app_request
def load_user():
    user_id = session.get("user_id")
    user = None
    if user_id:
        user = (
            get_db()
            .execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            )
            .fetchone()
        )
    g.user = user


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)

    return wrapped


@auth_bp.post("/api/signup")
def signup():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()
    exists = db.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    ).fetchone()
    if exists:
        return jsonify({"error": "Username already taken"}), 409

    password_hash = generate_password_hash(password)
    db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash),
    )
    db.commit()
    user_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    session["user_id"] = user_id

    return jsonify({"id": user_id, "username": username})


@auth_bp.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = (
        get_db()
        .execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        .fetchone()
    )
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    session["user_id"] = user["id"]
    return jsonify({"id": user["id"], "username": user["username"]})


@auth_bp.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.get("/api/me")
def me():
    if g.get("user") is None:
        return jsonify({"error": "Not authenticated"}), 401
    user = g.user
    return jsonify({"id": user["id"], "username": user["username"]})


@auth_bp.get("/partials/session")
def session_partial():
    if g.get("user") is None:
        content = (
            '<div id="session-badge" class="badge badge--anon">Guest</div>'
        )
    else:
        content = (
            f'<div id="session-badge" class="badge badge--user">'
            f'{g.user["username"]}</div>'
        )
    return content
