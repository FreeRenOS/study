import requests
from bs4 import BeautifulSoup


url = "https://kabutan.jp/stock/?code=7203"
html = requests.get(url)
soup = BeautifulSoup(html.content, "html.parser")
stock = soup.find(class_="kabuka").text
print(stock)