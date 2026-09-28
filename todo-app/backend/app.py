import os
import sqlite3
from datetime import datetime, timezone

from flask import Flask, g, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "todos.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=None)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row):
    return {
        "id": row["id"],
        "text": row["text"],
        "completed": bool(row["completed"]),
        "created_at": row["created_at"],
    }


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.route("/api/todos", methods=["GET"])
def list_todos():
    cur = get_db().execute(
        "SELECT id, text, completed, created_at FROM todos ORDER BY id ASC"
    )
    rows = cur.fetchall()
    return jsonify([row_to_dict(r) for r in rows]), 200


@app.route("/api/todos", methods=["POST"])
def create_todo():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "text is required and must be non-empty"}), 400
    text = text.strip()
    created_at = datetime.now(timezone.utc).isoformat()
    db = get_db()
    cur = db.execute(
        "INSERT INTO todos (text, completed, created_at) VALUES (?, 0, ?)",
        (text, created_at),
    )
    db.commit()
    row = db.execute(
        "SELECT id, text, completed, created_at FROM todos WHERE id = ?",
        (cur.lastrowid,),
    ).fetchone()
    return jsonify(row_to_dict(row)), 201


@app.route("/api/todos/<int:todo_id>", methods=["PUT"])
def update_todo(todo_id):
    data = request.get_json(silent=True) or {}
    fields = []
    values = []

    if "text" in data:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "text must be a non-empty string"}), 400
        fields.append("text = ?")
        values.append(text.strip())

    if "completed" in data:
        completed = data.get("completed")
        if not isinstance(completed, bool):
            return jsonify({"error": "completed must be a boolean"}), 400
        fields.append("completed = ?")
        values.append(1 if completed else 0)

    if not fields:
        return (
            jsonify({"error": "at least one of text/completed must be provided"}),
            400,
        )

    db = get_db()
    existing = db.execute(
        "SELECT id FROM todos WHERE id = ?", (todo_id,)
    ).fetchone()
    if existing is None:
        return jsonify({"error": "todo not found"}), 404

    values.append(todo_id)
    db.execute(f"UPDATE todos SET {', '.join(fields)} WHERE id = ?", values)
    db.commit()
    row = db.execute(
        "SELECT id, text, completed, created_at FROM todos WHERE id = ?",
        (todo_id,),
    ).fetchone()
    return jsonify(row_to_dict(row)), 200


@app.route("/api/todos/<int:todo_id>", methods=["DELETE"])
def delete_todo(todo_id):
    db = get_db()
    existing = db.execute(
        "SELECT id FROM todos WHERE id = ?", (todo_id,)
    ).fetchone()
    if existing is None:
        return jsonify({"error": "todo not found"}), 404
    db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    db.commit()
    return "", 204


# --- Static file hosting (root path) ---
@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/todos", methods=["OPTIONS"])
@app.route("/api/todos/<int:todo_id>", methods=["OPTIONS"])
def cors_options(_todo_id=None):
    return "", 204


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
