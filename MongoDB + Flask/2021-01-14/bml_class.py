# 클래스 버전
import math


class Bmi:
    def __init__(self):
        self.height = 0.0
        self.weight = 0
        self.bmi = 0.0

    def __init__(self, height, weight):
        self.height = float(height) / 100.0
        self.weight = weight

    def bmi_calc(self):
        self.bmi = self.weight / math.pow(self.height, 2)
        self.bmi = round(self.bmi, 1)
        return self.bmi


# 클래스를 오브젝트로 생성 방법
height = input('키(cm) 입력 ')
height = int(height)

weight = input('몸무게(kg) 입력 ')
weight = int(weight)

bmi = Bmi(height, weight)
bmi = bmi.bmi_calc()
print(bmi)
