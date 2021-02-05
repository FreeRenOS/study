from flask import Flask, Response
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)  # Flask 객체에 Api 객체 등록


@api.route('/pharm')  # 데코레이터 이용
class Pharm(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        result = {"약국정보": "Web API"}
        result = json.dumps(result, ensure_ascii=False).encode('utf-8')
        return Response(result, content_type='application/json; charset=utf-8')


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
