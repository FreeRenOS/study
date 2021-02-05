import MySQLdb


def db_select():
    """ select """

    # 접속한다
    connect = MySQLdb.connect(
        user='root',
        password='1234',
        host='localhost',
        db='bookstore',
        charset='utf8')

    cursor = connect.cursor()
    sql = 'select id, title, author from books'
    cursor.execute(sql)

    rows = cursor.fetchall()
    for row in rows:
        print(row)

    cursor.close()
    connect.close()


def db_insert():
    """ insert """

    # 접속한다
    connect = MySQLdb.connect(
        user='root',
        password='1234',
        host='localhost',
        db='bookstore',
        charset='utf8')

    cursor = connect.cursor()
    title = 'machine learning'
    author = '이순자'
    sql = f"insert into books(title, author) values('{title}', '{author}')"
    cursor.execute(sql)

    connect.commit() # db에 반영하라

    cursor.close()
    connect.close()


db_insert()