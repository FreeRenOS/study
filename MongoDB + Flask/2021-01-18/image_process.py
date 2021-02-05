import base64

from bson import ObjectId
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
photo_file = "C:\\Users\\w\\Pictures\\1.jpg"
photo = open(photo_file, 'rb')
encoded_string = base64.b64encode(photo.read())
print(encoded_string)
book = {"title": "Python 서적", "photo": encoded_string}

# client.song_db.books.insert_one(book)
document = client.song_db.books.find({'_id':ObjectId("6004e7e6ea9aa311bad2ad06")})

for doc in document:
    with open('88.jpg', mode='wb') as f:
        f.write(doc['photo'])
