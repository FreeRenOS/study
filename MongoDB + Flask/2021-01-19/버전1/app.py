from flask import Flask, render_template

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

    return render_template('book_add_result.html')


if __name__ == '__main__':
    app.run()
