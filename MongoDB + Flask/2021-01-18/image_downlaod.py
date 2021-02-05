import requests
from bs4 import BeautifulSoup
from pathlib import Path

movie_dir = Path('movie_image_download')
movie_dir.mkdir(exist_ok=True)

url = 'http://movie.naver.com/movie/running/current.nhn'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
# movie_category = soup.find('ul', {'class': 'lst_detail_t1'})
movie_category = soup.find('ul', class_='lst_detail_t1')

for image in movie_category.find_all('img'):
    image_name = image.get('src')
    download_image_name = image_name.split('?')[0]
    imgdata = requests.get(download_image_name)

    splited_names = download_image_name.split('/')
    filename = f'{splited_names[3]}-{splited_names[4]}-{splited_names[5]}'
    with open (movie_dir / filename, mode='wb') as f:
        f.write(imgdata.content)
