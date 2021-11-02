class Student(object):

    @property
    def birth(self, ):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self, ):
        return 2021 - self._birth


class Person:
    def __init__(self):
        self.__age = 18
        self.name = 'bangxi'

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, v):
        self.__age = v






