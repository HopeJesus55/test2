[![Automated tests](https://github.com/HopeJesus55/test2/actions/workflows/run_tests.yml/badge.svg)](https://github.com/HopeJesus55/test2/actions/workflows/run_tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/HopeJesus55/test2/badge.svg?branch=main)](https://coveralls.io/github/HopeJesus55/test2?branch=main)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=HopeJesus55_test2&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=HopeJesus55_test2)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=HopeJesus55_test2&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=HopeJesus55_test2)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=HopeJesus55_test2&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=HopeJesus55_test2)

Данная программа предназначена для решения задачи расщепления смеси случайных величин при помощи скрытого марковского процесса. Пользователь задаёт желаемые типы распределения случайных величин (экспоненциальное, нормальное), и соответствующие параметры для рассчета (мощность выборки, лямбда, среднее выборки и другие).

Сгенерированные выборки перемешиваются случайным образом, после чего, скрытый марковский процесс разделяет получившуюся смесь. 

Результатом работы программы является вывод, верно ли программа разделила смесь.

# План тестирования

# Аттестационное тестирование

  - Тест А1 (положительный)
    - Начальное состояние: Программа запущена и предлагает пользователю 3 действия (1 - найти Фибоначчи, 2 - найти корни, 0 - выход)
    - Действие: Пользователь вводит цифру, не предлагаемую программой (например, 4)
    - Ожидаемый результат:
        ```            
      	Команда не распознана!
        ```   

  - Тест А1
    - Начальное состояние: Программа запущена. Программа запрашивает выбор типа распределения для первой выборки. Программа предлагает на выбор три варианта:
      ``` 
      1 - экспоненциальное распределение
      2 - нормальное распределение
      0 - выход из программы
      ``` 
    - Действие: Пользователь вводит в консоль неверное значение варианта, например, 5
    - Ожидаемый результат: Вывод в крнсоль сообщения: "Данная команда не распознана". Программа повторно выводит на экран запрос выбора

  - Тест А2
    - Начальное состояние: Программа запущена. Программа запрашивает выбор типа распределения для первой выборки. Программа предлагает на выбор три варианта: 
      ``` 
      1 - экспоненциальное распределение
      2 - нормальное распределение
      0 - выход из программы
      ``` 
    - Действие: Пользователь вводит в консоль 0
    - Ожидаемый результат: Программа завершается

  - Тест А3
    - Начальное состояние: Программа запущена. Программа запрашивает выбор типа распределения для первой выборки. Программа предлагает на выбор три варианта:
      ``` 
      1 - экспоненциальное распределение
      2 - нормальное распределение
      0 - выход из программы
      ``` 
    - Действие: Пользователь вводит в консоль 1
    - Ожидаемый результат: Запускается функция рассчета экспоненциального распределения. Вывод сообщения: "Введите мощность выборки"

  - Тест А4
    - Начальное состояние: Функция рассчета экспоненциального распределения, после вывода сообщения "Введите мощность выборки"
    - Действие: Пользователь вводит число 10
    - Ожидаемый результат: Сохранение значения, вывод сообщения "Введите значение лямбда"

  - Тест А5
    - Начальное состояние: Функция рассчета экспоненциального распределения, после сообщения "Введите значение лямбда"
    - Действие: Пользователь вводит значение 0.3
    - Ожидаемый результат: Успешный рассчет выборки из 10 экспоненциально-распределенных величин. Переход к запросу от пользователя типа распределения второй выборки.

  - Тест А6
    - Начальное состояние: Выбор пользователем распределения второй выборки. Программа предлагает три варианта:
      ``` 
      1 - экспоненциальное распределение
      2 - нормальное распределение
      0 - выход из программы
      ``` 
    - Действие: Пользователь вводит в консоль 2
    - Ожидаемый результат: Запускается функция рассчета нормального распределения. Вывод сообщения "Введите мощность выборки"

  - Тест А7
    - Начальное состояние: Функция рассчета нормально распределения, ввод мощности выборки
    - Действие: Пользователь вводит отрицательное значение, например -14
    - Ожидаемый результат: Вывод в консоль сообщения: "Мощность не может быть отрицательной!". Программа повторно выводит на экран запрос ввода мощности

  - Тест А8
    - Начальное состояние: Функция рассчета нормального распределения, после корректного ввода мощности. Программа выводит сообщение: "Введите значение среднего выборки"
    - Действие: Пользователь вводит 50
    - Ожидаемый результат: Сохранение значения, вывод сообщения "Введите среднеквадратичное отклонение"

  - Тест А9
    - Начальное состояние: Функция рассчета нормального распределения, ввод среднеквадратичного отклонения
    - Действие: Пользователь вводит 20
    - Ожидаемый результат: Успешный рассчет выборки нормально-распределенных величин. Переход к запросу от пользователя типа модели марковской цепи

  - Тест А10
    - Начальное состояние: Выбор модели марковской цепи. Программа предлагает на выбор три варианта:
      ``` 
      1 - экспоненциальное распределение
      2 - нормальное распределение
      0 - выход из программы
      ``` 
    - Действие: Пользователь вводит в консоль 1
    - Ожидаемый результат: Программа строит гауссовскую модель на основе рассчитанных ранее выборок случайных величин.
   
   - Тест А11
    - Начальное состояние: Построение гауссовской модели. Модель разделила выборки неверно
    - Действие: -
    - Ожидаемый результат: Вывод на экран сообщения "Модель разделила выборки неверно! :("

  - Тест А12
    - Начальное состояние: Построение гауссовской модели. Модель разделила выборки верно
    - Действие: -
    - Ожидаемый результат: Вывод на экран сообщения "Модель разделила выборки верно! :)"


# Блочное тестирование

## Класс SV
(Скинуть описание из пайтана)

  - Тест Б1
    - Описание:
    - Метод: init
    - Входные данные:
    - Ожидаемый результат:

  - Тест Б2
    - Описание:
    - Метод: _generate
    - Входные данные:
    - Ожидаемый результат:

### Класс-наследник ExpRasp

  - Тест Б3
    - Описание:
    - Метод:
    - Входные данные:
    - Ожидаемый результат:

  - Тест Б4
    - Описание:
    - Метод:
    - Входные данные:
    - Ожидаемый результат:
   
### Класс-наследник NormRasp

  - Тест Б5
    - Описание:
    - Метод:
    - Входные данные:
    - Ожидаемый результат:

  - Тест Б6
    - Описание:
    - Метод:
    - Входные данные:
    - Ожидаемый результат:


## Класс HMM
(Описание)
