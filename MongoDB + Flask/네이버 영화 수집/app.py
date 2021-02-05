import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template

app = Flask(__name__)

url = 'https://movie.naver.com/movie/running/current.nhn'


@app.route('/')
def data_gathering():
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # print(soup.prettify())

    ul = soup.find('ul', class_="lst_detail_t1")
    print(ul)

    return render_template('data_gathering.html');


if __name__ == '__main__':
    app.run()
