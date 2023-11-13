from class_HMM import HMM
from class_SV import SV, NormSV, ExpSV

#testing
def call_exp():
    print("Пожалуйста, введите мощность выборки:")
    count = input()
    count = int(count)
    print("Пожалуйста, введите значение лямбды (обратная к среднему выборки):")
    Lambda = input()
    Lambda = float(Lambda)
    ExpRasp = ExpSV(count, Lambda)
    ExpRasp.generate_exp_rasp()
    return ExpRasp.rasp


def call_norm():
    print("Пожалуйста, введите мощность выборки:")
    count = input()
    count = int(count)
    print("Пожалуйста, введите значение среднего выборки (Тср):")
    Tsp = input()
    Tsp = int(Tsp)
    print("Пожалуйста, введите значение сигма (среднеквадратичное отклонение):")
    Sigma = input()
    Sigma = int(Sigma)
    NormRasp = NormSV(count, Tsp, Sigma)
    NormRasp.generate_norm_rasp()
    return NormRasp.rasp


if __name__ == '__main__':
    print("Добро пожаловать в программу по решению задачи разделения смеси случайных величин!\n")
    print("Пожалуйста, выберите тип первой выборки:\n1 - Экспоненциальная\n2 - Нормальная\n0 - Выйти")
    while True:
        choice = input()
        if choice == '0':
            quit()
        elif choice == '1':
            X = call_exp()
            break
        elif choice == '2':
            X = call_norm()
            break
        else:
            print("Данная команда не распознана :(")

    print("Пожалуйста, выберите тип второй выборки:\n1 - Экспоненциальная\n2 - Нормальная\n0 - Выйти")
    while True:
        choice = input()
        if choice == '0':
            quit()
        elif choice == '1':
            Y = call_exp()
            break
        elif choice == '2':
            Y = call_norm()
            break
        else:
            print("Данная команда не распознана :(")

    hmc = HMM(X, Y)
    hmc.shuffle_matrix()

    print(
        "Пожалуйста, выберите тип модели скрытой цепи Маркова:\n1 - Гауссовская\n2 - Гауссовская с возможными выбросами смеси\n0 - Выйти")
    while True:
        choice = input()
        if choice == '0':
            quit()
        elif choice == '1':
            hmc.Gaussian_model()
            hmc.check_correct_predict()
            print("\n\nСпасибо за использование данной программы!")
            break
        elif choice == '2':
            hmc.GMMHMM_model()
            hmc.check_correct_predict()
            print("\n\nСпасибо за использование данной программы!")
            break
        else:
            print("Данная команда не распознана :(")
