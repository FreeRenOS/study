import MySQLdb


class Connect:
    def __init__(self):
        self.connect = MySQLdb.connect(
            user = 'root',
            password = '1234',
            host = 'localhost',
            db = 'bookstore',
            charset = 'utf8')

        self.cursor = self.connect.cursor()


while True:
    con = Connect()
    print ('1: 전체 조회  2: 등록  7: 건수 표시   9: 종료 ', end='')
    choice = input()
    choice = int(choice)

    if choice == 1:
       sql = 'select id, title, author from books'
       con.cursor.execute(sql)

       rows = con.cursor.fetchall()
       for row in rows:
           print(row)

    if choice == 2:
        title = input('제목 입력 ')
        author = input('저자명 입력 ')
        sql = f"insert into books(title, author) values('{title}', '{author}')"
        con.cursor.execute(sql)
        con.connect.commit()

    if choice == 7:
        sql = 'select count(*) from books'
        con.cursor.execute(sql)
        (count, ) = con.cursor.fetchone()
        print('전체 건수 =', count)

    if choice == 9:
        exit(0)