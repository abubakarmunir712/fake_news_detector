import os
import logging
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException

from .auth import auth_bp
from .config import Config
from .db import init_app as init_db
from .detect import detect_bp


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def create_app() -> Flask:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    settings = Config.load()

    app = Flask(
        __name__,
        static_folder=os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist")),
        static_url_path="/",
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    database_path = settings.database or os.path.join(app.instance_path, "app.sqlite")
    app.config.update(
        SECRET_KEY=settings.secret_key,
        DATABASE=database_path,
        GEMINI_API_KEY=settings.gemini_api_key,
        TAVILY_API_KEY=settings.tavily_api_key,
        PORT=settings.port,
        TIMEOUT=settings.timeout,
    )

    os.makedirs(app.instance_path, exist_ok=True)

    CORS(
        app,
        resources={r"/api/*": {"origins": settings.cors_origins}},
        supports_credentials=True,
    )
    init_db(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(detect_bp)

    # Error handlers
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return jsonify({"error": e.description}), e.code

    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def root(path: str):
        if path.startswith("api/") or path.startswith("partials/"):
            return jsonify({"error": "Not found"}), 404

        file_path = os.path.join(app.static_folder, path)
        if os.path.isfile(file_path):
            return send_from_directory(app.static_folder, path)

        return send_from_directory(app.static_folder, "index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"})

    return app

