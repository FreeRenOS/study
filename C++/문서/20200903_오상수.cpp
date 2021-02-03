#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <time.h>


using namespace std;

//[����6]
class Employee {
private:
	string name;

public:
	Employee(string name) {
		this->name = name;
	}
	void ShowYourName() const {
		cout << "�̸� : " << name << endl;
	}
	virtual int GetPay() const {
		return 0;
	}
	virtual void ShowSalaryInfo() const {}
};


class PermanentWorker : public Employee
{
private:
	int salary;
public:
	PermanentWorker(string name, int salary) : Employee(name)
	{
		this->salary = salary;
	}
	int GetPay() const {
		return this->salary;
	}
	void ShowSalaryInfo() const
	{
		ShowYourName();
		cout << "�̴��� ���� : " << GetPay() << endl;;
	}
};



class SalesWorker : public PermanentWorker
{
private:
	int salesResult;
	double bonusRatio;
public:
	SalesWorker(string name, int salary, int bonusRatio) : PermanentWorker(name, salary)
	{
		this->salesResult = 0;
		this->bonusRatio = bonusRatio;

	}
	int GetPay() const
	{
		return PermanentWorker::GetPay() + int(salesResult * bonusRatio);
	}
	void ShowSalaryInfo() const
	{
		ShowYourName();
		cout << "�̴��� ����(�󿩱�) : " << GetPay() << endl;;

	}

};

class TemporaryWoker :public Employee
{
private:
	int workTime;
	int payPerHour;

public:
	TemporaryWoker(string name, int payPerHour) : Employee(name),
		workTime(0),
		payPerHour(payPerHour) {}

	void AddWorkTime(int time)
	{
		this->workTime += time;
	}

	int GetPay() const
	{
		return payPerHour * workTime;

	}
	void ShowSalaryInfo() const
	{
		ShowYourName();
		cout << "�̹��� ����(�ñ�) : " << GetPay() << endl;

	}

};



class EmployHandler {
private:
	Employee* empList[50];
	int empNum;
public:
	EmployHandler() : empNum(0), empList{ 0 } {}
	void AddEmployee(Employee* emp)
	{
		empList[empNum++] = emp;

	}
	void ShowAllSalaryInfo() const
	{
		for (int i = 0; i < empNum; i++) {
			empList[i]->ShowSalaryInfo();
		}

	}
	void ShowTotalSalary() const
	{
		double sum = 0;
		for (int i = 0; i < empNum; i++) {
			sum += empList[i]->GetPay();
		}
		cout << "���� �޿� ���� : " << sum << endl;

	}
	~EmployHandler()
	{
		for (int i = 0; i < empNum; i++) {
			delete empList[i];
		}
	}
};

int main()
{

	////[����1]

	//vector<double> Num = { 1,-2,3,-5,8,-3 };
	//vector<double> filter;


	//copy_if(Num.begin(), Num.end(), back_inserter(filter), [](const auto& num) {return num > 0; });


	//for (auto i = filter.begin(); i < filter.end(); i++) {
	//	cout << *i << endl;
	//}




	////[����2]

	//vector<double> Num = { 1,2,3,4 };
	//vector<double> map;

	//transform(Num.begin(), Num.end(), back_inserter(map), [](const auto& num) {return num * 3; });

	//for (auto i = map.begin(); i < map.end(); i++) {
	//cout << *i << endl;
	//}




	//////[����3]

	//string passward, passward1;
	//int count = 1;
	//cout << "��ȣ�� �����ϼ��� : ";
	//cin >> passward;
	//while (true) {
	//	cout << "������ ��ȣ�� �Է��ϼ��� : ";
	//	cin >> passward1;
	//	//strcmp
	//	if (passward.compare(passward1) == 0) {
	//		cout << "���� �����մϴ�" << endl;
	//		break;
	//	}
	//	else
	//	{
	//		cout << "��ȣ�� Ʋ�Ƚ��ϴ�" << endl;
	//		count++;
	//	}
	//	if (count == 6) {
	//		cout << "��ȣ�� 5ȸ Ʋ�Ƚ��ϴ�. �����մϴ�!!!" << endl;
	//		break;
	//	}
	//}




	////[����4]

	//srand((unsigned)time(NULL));

	//int number = rand() % 101 + 1; //1���� 100����
	//int user;
	//while (true) {
	//	cout << "���ڸ� �Է��ϼ��� : ";
	//	cin >> user;

	//	if (user > number) {
	//		cout << "down" << endl;
	//	}
	//	else if (user < number) {
	//		cout << "up" << endl;

	//	}
	//	else if (user == number) {
	//		cout << "����" << endl;
	//		break;
	//	}

	//}




	////[����5]
	//cout << "--------------------------------------------------" << endl;
	//cout << "���� �߱� ����" << endl;
	//cout << "--------------------------------------------------" << endl;
	//cout << "��ǻ�Ͱ� ������� �Ǿ� �� �ڸ� ���� �ϳ� ������ϴ�." << endl;
	//cout << "�� ���ڴ� 0~9 �� �ϳ��� �ߺ��Ǵ� ���ڴ� �����ϴ�" << endl;
	//cout << "��� ���ڿ� ��ġ�� ���߸� �¸��մϴ�" << endl;
	//cout << "���ڿ� ���簡 �Ѵ� ������ ��Ʈ����ũ �Դϴ�" << endl;
	//cout << "���ڸ� �°� ������ Ʋ���� ���Դϴ�" << endl;
	//cout << "���ڰ� Ʋ���� �ƿ��Դϴ�" << endl;
	//cout << "--------------------------------------------------" << endl;



	//srand((unsigned)time(NULL));
	//int number[3];
	//int user[3];

	////�ߺ����� ���� 0~9����
	//for (int i = 0; i < 3; i++) {
	//	number[i] = rand() % 10;
	//	for (int j = 0; j < i; j++) {
	//		if (number[i] == number[j]) {
	//			i--;
	//			break;
	//		}
	//	}
	//}

	//while (true) {
	//	int ball = 0, strike = 0, out = 3;
	//	cout << "ù ��° ���ڸ� �Է��ϼ��� : ";
	//	cin >> user[0];
	//	cout << "�� ��° ���ڸ� �Է��ϼ��� : ";
	//	cin >> user[1];
	//	cout << "�� ��° ���ڸ� �Է��ϼ��� : ";
	//	cin >> user[2];

	//	//�������ڰ˻�
	//	for (int i = 0; i < 3; i++) {
	//		for (int j = 0; j < i; j++) {
	//			if (user[i] == user[j]) {
	//				cout << "���� ���ڸ� �Է��ϸ� �ȵ˴ϴ�." << endl;
	//				break;
	//			}
	//		}
	//	}

	//	//����
	//	for (int i = 0; i < 3; i++) {
	//		for (int j = 0; j < 3; j++) {
	//			if (number[i] == user[j] && i == j) {
	//				strike++; out--;
	//			}
	//			else if (number[i] == user[j]) {
	//				ball++; out--;
	//			}
	//		}
	//	}

	//	cout << "���� : " << user[0] << " " << user[1] << " " << user[2] << " ";
	//	cout << "���� : " << ball << "B" << " " << strike << "S" << " " << out << "O" << " " << endl;

	//	//�׽�Ʈ�� ���� ������ �����
	//	cout << "����" << endl;
	//	for (int i = 0; i < 3; i++) {
	//		cout << number[i] << " ";
	//	}
	//	cout << endl;

	//	if (strike == 3) {
	//		cout << "����" << endl;
	//		for (int i = 0; i < 3; i++) {
	//			cout << number[i] << " ";
	//		}
	//		cout << endl;
	//		break;
	//	}
	//}





//[����6] main
EmployHandler hand;

//�������
hand.AddEmployee(new PermanentWorker("����1", 200));
hand.AddEmployee(new PermanentWorker("����2", 400));
hand.AddEmployee(new PermanentWorker("����3", 500));

TemporaryWoker* parttime = new TemporaryWoker("�˹ٻ�", 8);
parttime->AddWorkTime(80);
hand.AddEmployee(parttime);


//�̹��޿� �����ؾ��� �޿��� ����
hand.ShowAllSalaryInfo();


//�̹��޿� �����ؾ��� �޿��� ����
hand.ShowTotalSalary();





	return 0;
}