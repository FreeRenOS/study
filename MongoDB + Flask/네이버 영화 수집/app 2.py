import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template

app = Flask(__name__)

url = 'https://movie.naver.com/movie/running/current.nhn'


@app.route('/')
def data_gathering():
    img_list = []
    rating_list = []
    movie_name_list = []
    estimated_people_list = []
    estimate_scores_list = []
    number_of_participants_list = []

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # print(soup.prettify())

    ul = soup.find('ul', class_="lst_detail_t1")
    img = ul.select('li > div > a > img')
    # print(len(img))
    for i in img:
        img_list.append(i.get('src'))

    rating = ul.select('li > dl > dt > span')
    for rate in rating:
        rating_list.append(rate.text)
    movie_names = ul.select('li > dl > dt > a')
    for movie_name in movie_names:
        movie_name_list.append(movie_name.text)

    estimated_people = ul.select('dl > dd.star > dl.info_star > dt')
    for estimated_p in estimated_people:
        estimated_people_list.append(estimated_p.text)

    estimate_scores = ul.select('dl > dd.star > dl.info_star > dd > div > a > span.num')
    for estimate_score in estimate_scores:
        estimate_scores_list.append(estimate_score.text)

    number_of_participants = \
        ul.select('dl > dd.star > dl.info_star > dd > div > a > span.num2 > em')
    for number_of_participant in number_of_participants:
        number_of_participants_list.append(number_of_participant.text)

    print(number_of_participants_list)
    return render_template('data_gathering.html');


if __name__ == '__main__':
    app.run()
