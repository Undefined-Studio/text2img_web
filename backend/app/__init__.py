from flask import Flask
from .blueprints.text2img import text2img_blueprint


def create_app():
    app = Flask(__name__)

    app.register_blueprint(text2img_blueprint)

    return app





