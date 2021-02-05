import random

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('lotto_input.html')


@app.route('/lotto_result', methods=['POST'])
def lotto_result():
    mylotto1 = int(request.form['mylotto1'])
    mylotto2 = int(request.form['mylotto2'])
    mylotto3 = int(request.form['mylotto3'])
    mylotto4 = int(request.form['mylotto4'])
    mylotto5 = int(request.form['mylotto5'])
    mylotto6 = int(request.form['mylotto6'])

    # lotot_all = []
    # for l in range(1, 46):
    #     lotot_all.append(l)
    # list comprehension : 리스트 내포
    lotot_all = [x for x in range(1, 46)]
    # list 내포
    # 섞어야 한다
    random.shuffle(lotot_all)
    # print(lotot_all)

    # 슬라이싱
    lotto_list = lotot_all[0:6]

    print(lotto_result)
    return render_template('lotto_result.html', lotto_list=lotto_list)


if __name__ == '__main__':
    app.run()
