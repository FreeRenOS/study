import requests
from bs4 import BeautifulSoup
import urllib

url = 'https://www.ymori.com/books/python2nen/test2.html'
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
for element in soup.find_all("img"):
    src = element.get("src")
    image_url = urllib.parse.urljoin(url, src)
    filename = image_url.split("/")[-1]
    print(image_url, ">>", filename)
