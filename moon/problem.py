from enum import Enum, auto
import pandas as pd

class Grade:
    def __init__(self, grade, usergrade, prefer_user=True):
        self._grade = grade
        self._usergrade = usergrade
        self.grade = usergrade if prefer_user and not pd.isna(usergrade) else grade
        # TODO one-hot, int, ordinal representations
    
    def __repr__(self):
        return self.grade


class Setter:
    def __init__(self, first, last, nick, city, country):
        self.first = first
        self.last = last
        self.nick = nick
        self.city = city
        self.country = country
    
    def __repr__(self):
        return self.nick


class ProblemType(Enum):
    Crimp = auto()  # Not all-caps so that item access works
    Other = auto()


class Problem:
    def __init__(self, data, prefer_user_grade=True):
        self.name = data.Name
        self.grade = Grade(data.Grade, data.UserGrade, prefer_user=prefer_user_grade)
        self.setter = Setter(data.Setter_Firstname, data.Setter_Lastname, data.Setter_Nickname, data.Setter_City, data.Setter_Country)
        self.rating = data.UserRating
        self.repeats = data.Repeats
        self.benchmark = data.IsBenchmark
        self.master = data.IsMaster
        self.assessment = data.IsAssessmentProblem
        if not pd.isna(data.ProblemType) and data.ProblemType in ProblemType.__members__:
            self.type = ProblemType[data.ProblemType]
        else:
            self.type = ProblemType.Other
        # self.date = data.DateTimeString
        self.holds_start = data.Holds_Start
        self.holds_intermed = data.Holds_Intermed
        self.holds_end = data.Holds_End
        # TODO different holds representations: binary 3d array, sequence, ...

    def __repr__(self):
        return f'Problem(name={self.name}, grade={self.grade}, setter={self.setter}, rating={self.rating}, repeats={self.repeats}, benchmark={self.benchmark}, master={self.master}, assessment={self.assessment}, type={self.type})'