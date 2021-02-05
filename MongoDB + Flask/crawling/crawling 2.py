import requests
from bs4 import BeautifulSoup
import urllib
from pathlib import Path
import time


url = 'https://www.ymori.com/books/python2nen/test2.html'
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

out_folder = Path("download2")
out_folder.mkdir(exist_ok=True)

for element in soup.find_all("img"):
    src = element.get("src")
    image_url = urllib.parse.urljoin(url, src)
    imgdata = requests.get(image_url)

    filename = image_url.split("/")[-1]
    out_path = out_folder.joinpath(filename)

    with open(out_path, mode="wb") as f:
        f.write(imgdata.content)
    time.sleep(1)