from flask import Flask
from .text2pic_blueprint import text2pic_blueprint


def create_app():
    app = Flask(__name__)

    app.register_blueprint(text2pic_blueprint)

    return app





