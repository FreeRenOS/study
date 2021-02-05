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
    img = ul.select('li > div > a > img')
    # print(len(img))
    # for i in img:
    #     print(i.get('src'))
    rating = ul.select('li > dl > dt > span')
    # for rate in rating:
    #     print(rate.text)
    movie_names = ul.select('li > dl > dt > a')
    # for movie_name in movie_names:
    #     print(movie_name.text)
    estimated_people = ul.select('dl > dd.star > dl.info_star > dt')
    # for estimated_p in estimated_people:
    #     print(estimated_p.text)

    estimate_scores = ul.select('dl > dd.star > dl.info_star > dd > div > a > span.num')
    # for estimate_score in estimate_scores:
    #     print(estimate_score.text)

    number_of_participants = \
        ul.select('dl > dd.star > dl.info_star > dd > div > a > span.num2 > em')
    for number_of_participant in number_of_participants:
        print(number_of_participant.text)

    return render_template('data_gathering.html');


if __name__ == '__main__':
    app.run()
