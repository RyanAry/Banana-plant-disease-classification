from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config


__path__ = [str(Path(__file__).resolve().parent / "app")]


def create_app() -> Flask:
	app = Flask(
		__name__,
		template_folder=Config.TEMPLATES_FOLDER,
		static_folder=Config.STATIC_FOLDER,
		static_url_path=Config.STATIC_URL_PATH,
	)
	app.config.from_object(Config)

	Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

	from app.routes.upload import bp as upload_bp

	app.register_blueprint(upload_bp)

	@app.errorhandler(RequestEntityTooLarge)
	def handle_request_entity_too_large(error: RequestEntityTooLarge):
		return (
			render_template(
				"index.html",
				error_message="File is too large. Maximum upload size is 5 MB.",
				uploaded_image=None,
				prediction_result=None,
			),
			413,
		)

	return app


app = create_app()


if __name__ == "__main__":
	app.run(debug=app.config.get("DEBUG", False))
