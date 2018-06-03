from flask import Blueprint, jsonify, request
from flask_restful import Resource, Api, reqparse

text2pic_blueprint = Blueprint(
    'text2pic',
    __name__,
    url_prefix='/api/text2pic'
)

text2pic_api = Api(text2pic_blueprint)


class Text2Pic(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        text = json_data['data']
        return "Hello, World!"


text2pic_api.add_resource(Text2Pic, '/create')

