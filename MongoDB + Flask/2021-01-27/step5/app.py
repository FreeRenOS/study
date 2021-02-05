import random

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('lotto_input.html')


@app.route('/lotto_result', methods=['POST'])
def lotto_result():
    mylotto = request.form.values()
    mylotto2 = int(request.form['mylotto2'])
    mylotto3 = int(request.form['mylotto3'])
    mylotto4 = int(request.form['mylotto4'])
    mylotto5 = int(request.form['mylotto5'])
    mylotto6 = int(request.form['mylotto6'])
    # my_list = list(mylotto1, mylotto2, mylotto3, mylotto4, mylotto5, mylotto6)
    for m in mylotto:
        print(m)
    # for l in range(1, 46):
    #     lotot_all.append(l)
    # list comprehension : 리스트 내포
    lotot_all = [x for x in range(1, 46)]
    # list 내포
    # 섞어야 한다
    # random.shuffle(lotot_all)
    lotto_list = random.sample(lotot_all, 6)
    print(lotto_list)

    # 슬라이싱
    # lotto_list = lotot_all[0:6]

    print(lotto_result)
    return render_template('lotto_result.html', lotto_list=lotto_list)


if __name__ == '__main__':
    app.run()
