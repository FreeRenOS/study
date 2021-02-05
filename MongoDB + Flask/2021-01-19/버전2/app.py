import base64

from flask import Flask, render_template, request
from pymongo import MongoClient

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
    title = request.form['title']
    file = request.files['file']
    author = request.files['author']
    price = request.files['price']
    isbn = request.files['isbn']
    encoded_data = base64.b64encode(file.read())

    return render_template('book_add_result.html', title=title)


if __name__ == '__main__':
    app.run()
