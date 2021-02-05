import MySQLdb


def db_sample():
    """ 접속 테스트"""

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


if __name__ == "__main__":
    db_sample()