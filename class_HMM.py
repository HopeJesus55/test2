import random
from hmmlearn import hmm


class HMM:
    """класс - скрытая модель Маркова (Hidden Markov Model). Свойства:
        @ X - первый список случ величин
        @ Y - второй список случ величин
        @ matrix - объединенный список (Х + У)
        @ shuffled - случайно перемешанная список (matrix)
        @ Z2 - предсказания"""

    def __init__(self, X, Y):
        """Метод инициализации. Свойства:
            @ X - первый список случ величин
            @ Y - второй список случ величин"""
        self.X = X
        self.Y = Y
        self.matrix = []
        self.shuffled = []
        self.Z2 = []

    @staticmethod
    def concat_SV(X, Y):
        """Статичный метод объединения двух списков случайных величин
            @ Возвращает список X+Y"""
        concat_list = []
        for i in range(len(X)):
            concat_list.append([X[i]])
        for j in range(len(Y)):
            concat_list.append([Y[j]])
        return concat_list

    def shuffle_matrix(self):
        """Метод перемешивания списка случайных величин
            @ Возвращает shuffled - перемешанный список"""
        self.matrix = self.concat_SV(self.X, self.Y)
        self.shuffled = self.matrix
        random.shuffle(self.shuffled)
        return self.shuffled

    def Gaussian_model(self):
        """Метод реализующий Гауссовскую модель скрытой Марковской цепи
            @ Возврашает Z2 - список предсказаний о разделении списка случайных величин, где:
                @ 0 - случайные величины из первого списка (X)
                @ 1 - случайные величины из второго списка (Y)"""
        model2 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=500, random_state=0)
        model2.fit(self.shuffled)
        # Z2 - предположения о скрытых состояниях. Используется алгоритм Витерби
        self.Z2 = model2.predict(self.shuffled)
        return self.Z2

    def GMMHMM_model(self):
        """Метод реализующий Гауссовскую модель скрытой Марковской цепи с возможными выбросами смеси
            @ Возврашает Z2 - список предсказаний о разделении списка случайных величин, где:
                @ 0 - случайные величины из первого списка (X)
                @ 1 - случайные величины из второго списка (Y)"""
        model2 = hmm.GMMHMM(n_components=2, covariance_type="full", n_iter=500, random_state=0)
        model2.fit(self.shuffled)
        # Z2 - предположения о скрытых состояниях. Используется алгоритм Витерби
        self.Z2 = model2.predict(self.shuffled)
        return self.Z2

    def check_correct_predict(self):
        """Метод реализующий проверку корректности предположений модели
            @ Возврашает:
                    True - верное предсказание
                    False - неверное предсказание"""
        zero_count = 0
        for z in self.Z2:
            if z == 0:
                zero_count += 1
        one_count = len(self.Z2) - zero_count
        print(zero_count, one_count)
        if zero_count == len(self.X) and one_count == len(self.Y):
            print("Модель разделила смесь случайных величин верно! :)")
            return True
        else:
            print("Модель разделила смесь случайных величин неверно! :(")
            return False
