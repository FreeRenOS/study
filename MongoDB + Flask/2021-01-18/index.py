from bottle import route, run, template


@route('/')
def index():
    return '<h3>안녕하세요</h3>!'


run(host='localhost', port=8080)
