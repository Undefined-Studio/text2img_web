from flask import Blueprint, request
from flask_restful import Resource, Api
from flask_cors import CORS
from app.utils import gen

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
        gen.run(text)


text2img_api.add_resource(Create, '/create')


