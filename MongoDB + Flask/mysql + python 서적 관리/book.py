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
    print ('1: 전체 조회  2: 등록  3: 업데이트  4: 삭제  5: 제목 유사검색  7: 건수 표시   9: 종료 ', end='')
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
        affected_rows = con.cursor.execute(sql)
        if affected_rows == 1:
            print('정상 등록 되었음')
        con.connect.commit()

    if choice == 3:  # 업데이트
        sql = 'select id, title, author from books'
        con.cursor.execute(sql)
        rows = con.cursor.fetchall()
        for row in rows:
            (id, title, author) = row
            print(f'{id}, {title},   {author}')

        id = input('업데이트할 id 입력 ')
        id = int(id)
        sql = f'select id, title, author from books where id={id}'
        con.cursor.execute(sql)
        (id, title, author) = con.cursor.fetchone()
        print(id, title, author)

        title = input('변경할 제목 입력 ')
        author = input('변경할 저자명 입력 ')

        sql = f"update books set title = '{title}', author='{author}' where id={id}"
        affeted_rows = con.cursor.execute(sql)
        if affeted_rows == 1:
            print('정상 변경 되었음')
        con.connect.commit()

    if choice == 4:  # 삭제
        sql = 'select id, title, author from books'
        con.cursor.execute(sql)
        rows = con.cursor.fetchall()
        for row in rows:
            (id, title, author) = row
            print(f'{id}, {title},   {author}')

        id = input('삭제할 id 입력 ')
        id = int(id)

        sql = f'delete from books where id={id}'
        affeted_rows = con.cursor.execute(sql)
        if affeted_rows == 1:
            print('정상 삭제 되었음')
        con.connect.commit()

    if choice == 5:  # 제목 유사검색
        title = input('유사 검색할 제목 입력 ')
        sql = f"select id, title, author from books where title like '{title}%'"
        con.cursor.execute(sql)

        rows = con.cursor.fetchall()
        for row in rows:
            (id, title, author) = row
            print(f'{id}, {title},   {author}')

    if choice == 7:
        sql = 'select count(*) from books'
        con.cursor.execute(sql)
        (total_count, ) = con.cursor.fetchone()
        print('전체 건수=', total_count)

    if choice == 9:
        exit(0)