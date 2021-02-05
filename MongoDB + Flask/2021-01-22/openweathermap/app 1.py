import base64

from bson import ObjectId
from flask import Flask, render_template, request, flash, redirect, url_for, session
from pymongo import MongoClient
from datetime import datetime
import random
import cv2
import requests
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
filename = 'atlas_connect_info.txt'


class MyMongoClient(object):
    def __init__(self):
        with open(filename, encoding='utf-8') as f:
            self.atlas_connection_info = f.read()
        self.client = MongoClient(self.atlas_connection_info)
        self.database = self.client["bookstore"]
        self.collection = self.database["books"]


@app.route('/weather', methods=['GET'])
def weather():
    city = 'daegu'
    appid = 'e5d4ba22d1c0aae4130753ea87c69eec'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={appid}'
    res = requests.get(url)
    #weather_data = json.loads(res.text)
    weather_data = res.json()

    description = weather_data["weather"][0]["description"]
    icon = weather_data["weather"][1]["icon"]
    temp = weather_data["main"]["temp"]-273

    return render_template('weather.html',
                           description=description,
                           icon=icon,
                           temp=temp)


@app.route('/card', methods=['GET'])
def card():
    return render_template('card.html')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test')
def test():

    if 'counter' in session:
        session['counter'] += 1
    else:
        session['counter'] = 0

    flash('ddddddddddddddd', 'error')
    return redirect("/")


@app.route('/atlas_connect_info')
def atlas_connect_info():
    with open(filename, encoding='utf-8') as f:
        atlas_connection_info = f.read()
    return render_template('atlas_connect_info.html',
                           atlas_connection_info=atlas_connection_info)


@app.route('/atlas_connect_info_update', methods=['POST'])
def atlas_connect_info_update():
    atlas_connect_info = request.form['atlas_connect_info']
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(atlas_connect_info)
    return render_template('atlas_connect_info_update_result.html')


@app.route('/book_add', methods=['GET'])
def book_add():
    return render_template('book_add.html')


@app.route('/book_add_process', methods=['POST'])
def book_add_process():
    # client = MongoClient("mongodb://localhost:27017/")
    # database = client["song-db"]
    # collection = database["books"]
    myclient = MyMongoClient()
    title = request.form['title']
    file = request.files['file']
    author = request.form['author']
    price = request.form['price']
    isbn = request.form['isbn']
    encoded_data = base64.b64encode(file.read())

    doc = {'title': title, 'encoded_data': encoded_data, 'author': author,
           'price': price, 'created_date': datetime.now()}
    # flash('Thanks for registering')

    result = myclient.collection.insert_one(doc)
    book_add_result = None
    if result.inserted_id is not None:
        print(result.inserted_id)
        book_add_result = "정상 등록"
    else:
        book_add_result = "등록 실패"

    return render_template('book_add_result.html',
                           book_add_result=book_add_result)


@app.route('/book_search', methods=['GET'])
def book_search():
    return render_template('book_search.html')


@app.route('/book_search_process', methods=['POST'])
def book_search_process():
    item = request.form['item_to_search']
    data = request.form['data_to_search']

    myclient = MyMongoClient()
    if item == 'id':
        query = {'_id': data}
    elif item == 'title':
        query = {'title': data}

    books = myclient.collection.find(query)

    return render_template('book_search_process_result.html', books=books)


@app.route('/book_id_search', methods=['GET'])
def book_id_search():
    return render_template('book_id_search.html')


@app.route('/book_id_search_process', methods=['POST'])
def book_id_search_process():
    search = request.form['search']
    print(search)

    # client = MongoClient()
    # database = client["song-db"]
    # collection = database["books"]
    # myclient = MyMongoClient()
    # _id = request.form['id']
    #
    # doc = myclient.collection.find_one({'_id': ObjectId(_id)})
    # title = doc['title']
    # decoded_data = doc['encoded_data'].decode('utf-8')
    # # decoded_data = base64.b64decode(encoded_data)
    #
    # img_src_data = f'data:image/png;base64, {decoded_data}'

    return render_template('book_id_search_result.html')


@app.route('/book_list', methods=['GET'])
def book_list():
    #_id = request.args.get("_id")
    #print(_id)

    myclient = MyMongoClient()
    total_count = myclient.collection.find().count()

    books = myclient.collection.find()

    # booklist = []
    # for book in books:
    #     decoded_data = book['encoded_data'].decode('utf-8')
    #     img_src_data = f'data:image/png;base64, {decoded_data}'
    #     book['encoded_data'] = img_src_data
    #     booklist.append(book)

    return render_template('book_list.html',
                           books=books, total_count=total_count)


@app.route('/book_delete/<_id>')
def book_delete(_id=None):
    print(_id)
    return redirect("/")


@app.route('/book_details/<_id>')
def book_details(_id=None):
    print(_id)
    return render_template('book_details.html')


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
    gugu_list = []  # list()
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

    my_sum = int(selected_list[0].split('-')[0]) + int(selected_list[1].split('-')[0])
    computer_sum = int(selected_list[2].split('-')[0]) + int(selected_list[3].split('-')[0])
    my = my_sum % 10
    computer = computer_sum % 10

    # 판정
    winner = None
    if my > computer:
        winner = '내가 이겼다'
    elif my == computer:
        winner = '무승부'
    else:
        winner = '컴퓨터가 이겼다'

    return render_template('game.html',
                           selected_list=selected_list, winner=winner)


if __name__ == '__main__':
    app.run()
