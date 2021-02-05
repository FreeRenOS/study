import random

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('lotto_input.html')


@app.route('/lotto_result', methods=['POST'])
def lotto_result():

    my_lotto = [int(my) for my in request.form.values()]

    # my_list = list(mylotto1, mylotto2, mylotto3, mylotto4, mylotto5, mylotto6)
    # for l in range(1, 46):
    #     lotot_all.append(l)
    # list comprehension : 리스트 내포
    lotot_all = [x for x in range(1, 46)]
    # list 내포
    # 섞어야 한다
    # random.shuffle(lotot_all)
    lotto_list = random.sample(lotot_all, 6)

    # 슬라이싱
    # lotto_list = lotot_all[0:6]

    # 몇 개 당첨 되었는지
    myset = set(my_lotto)
    lottoset = set(lotto_list)

    winset = myset.intersection(lottoset)
    print('당첨된 숫자는 ', winset)
    print('당첨된 숫자 개수는 ', len(winset))

    return render_template('lotto_result.html', my_lotto=my_lotto,
                           lotto_list=lotto_list)


if __name__ == '__main__':
    app.run()
