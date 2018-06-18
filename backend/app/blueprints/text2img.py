from flask import Blueprint, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from app.common.generator import Generator

text2img_blueprint = Blueprint(
    'text2pic',
    __name__,
    url_prefix='/api/text2pic'
)

CORS(text2img_blueprint, resources={r'/api/*': {'origins': '*'}})

text2img_api = Api(text2img_blueprint)


class Create(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        text = json_data['data']
        return "Hello, World!"

        # gen = Generator()
        # gen.run(text)


class Result(Resource):
    def get(self):
        return "Hello, World!"


text2img_api.add_resource(Create, '/create')
text2img_api.add_resource(Result, "/result")


