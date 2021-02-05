import base64

from bson import ObjectId
from flask import Flask, render_template, request
from pymongo import MongoClient
from datetime import datetime
import random

app = Flask(__name__)


@app.route('/')
def hello_world():
    name = '이미자'
    return render_template('index.html', name=name)


@app.route('/book_add', methods=['GET'])
def book_add():
    return render_template('book_add.html')


@app.route('/book_add_process', methods=['POST'])
def book_add_process():
    client = MongoClient("mongodb://localhost:27017/")
    database = client["song-db"]
    collection = database["books"]

    title = request.form['title']
    file = request.files['file']
    author = request.form['author']
    price = request.form['price']
    isbn = request.form['isbn']
    encoded_data = base64.b64encode(file.read())

    doc = {'title': title, 'encoded_data': encoded_data, 'author': author,
           'price': price, 'created_date': datetime.now()}

    result = collection.insert_one(doc)

    book_add_result = None
    if result.inserted_id is not None:
        print(result.inserted_id)
        book_add_result = "정상 등록"
    else:
        book_add_result = "등록 실패"

    return render_template('book_add_result.html',
                           book_add_result=book_add_result)


@app.route('/book_id_search', methods=['GET'])
def book_id_search():
    return render_template('book_id_search.html')


@app.route('/book_id_search_process', methods=['POST'])
def book_id_search_process():
    client = MongoClient()
    database = client["song-db"]
    collection = database["books"]
    _id = request.form['id']

    doc = collection.find_one({'_id': ObjectId(_id)})
    title = doc['title']
    decoded_data = doc['encoded_data'].decode('utf-8')
    # decoded_data = base64.b64decode(encoded_data)

    img_src_data = f'data:image/png;base64, {decoded_data}'

    return render_template('book_id_search_result.html',
                           title=title, img_src_data=img_src_data)


@app.route('/newgugu', methods=['GET', 'POST'])
def newgugu():
    if request.method == 'POST':
        dan = int(request.form['dan'])

        gugu_list = []
        for i in range(1, 10):
            gugu = f'{dan} × {i} = {dan * i}'
            gugu_list.append(gugu)

    else:
        gugu_list = []

    return render_template('newgugu.html', gugu_list=gugu_list)


@app.route('/gugudan', methods=['GET'])
def gugudan():
    dan = 5
    gugu_list = [] # list()
    for i in range(1, 10):
        gugu = f'{dan} × {i} = {dan * i}'
        gugu_list.append(gugu)

    return render_template('gugudan.html', gugu_list=gugu_list)


@app.route('/jiji', methods=['GET', 'POST'])
def jiji():
    if request.method == 'POST':
        year = int(request.form['year'])
        jiji_list = ['자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해']
        jiji_index = (year - 4) % 12
        jiji = jiji_list[jiji_index]

    else:
        jiji = None
    return render_template('jiji.html', jiji=jiji)


@app.route('/game', methods=['GET'])
def game():
    my_list = [
        '1-1', '1-2', '2-1', '2-2', '3-1', '3-2', '4-1', '4-2', '5-1', '5-2',
        '6-1', '6-2', '7-1', '7-2', '8-1', '8-2', '9-1', '9-2', '10-1', '10-2'
     ]
    random.shuffle(my_list)
    selected_list = my_list[0:4]
    print(selected_list)
    return render_template('game.html', selected_list=selected_list)


if __name__ == '__main__':
    app.run()
