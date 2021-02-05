import math

height = input('키 입력(cm) >')
weight = input('체중 입력(kg) >')

height_meter = float(height) / 100.0
weight = int(weight)

bmi_value = weight / \
            (math.pow(height_meter,2))
bmi_value = round(bmi_value, 1)

bmi_result = ''
if bmi_value < 20:
    bmi_result = '저체중'
elif bmi_value <= 24:
    bmi_result = '정상'
elif bmi_value <= 29:
    bmi_result = '과체중'
elif bmi_value > 29:
    bmi_result = '비만'

print(f'BMI 수치={bmi_value} BMI 판정={bmi_result}')