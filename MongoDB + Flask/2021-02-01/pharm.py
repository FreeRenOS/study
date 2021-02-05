import csv

with open('pharm_2019.csv', encoding='utf-8') as csvf:
    csvReader = csv.DictReader(csvf)

    for row in csvReader:
        phar_dic = {}

        phar_dic['약국명'] = row['약국명']
        phar_dic['주소'] = row['주소']

        if row['위도'] is '' or row['경도'] is '':
            continue
        phar_loc = {'type': 'Point'}
        phar_loc['coordinates'] = [float(row['위도']), float(row['경도'])]

        phar_dic['location'] = phar_loc
        print(phar_dic)