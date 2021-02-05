import base64

from bson import ObjectId
from flask import Flask, render_template, request
from pymongo import MongoClient
from datetime import datetime

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

    query = {'_id': ObjectId(_id)}
    doc = collection.find_one({}, query)
    print(doc['title'])

    return render_template('book_id_search_result.html')


if __name__ == '__main__':
    app.run()
