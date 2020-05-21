from enum import Enum, auto
import moon.viz
import numpy as np
import pandas as pd


class Grade:
    # Our data is 6B to 8B+, but Moonboard 2019 goes down to 5+.
    ALL_GRADES = ['5+', '6A', '6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']
    N_GRADES = len(ALL_GRADES)
    ONEHOTS = np.identity(N_GRADES, dtype=np.int8)
    ORDINALS = np.tri(N_GRADES, dtype=np.int8)

    def __init__(self, grade, usergrade, prefer_user=True):
        self._grade = grade
        self._usergrade = usergrade
        self.grade = usergrade if prefer_user and not pd.isna(usergrade) else grade
        # Save other possible representations based on self.grade
        self.rank = Grade.ALL_GRADES.index(self.grade)
        self.onehot = Grade.ONEHOTS[self.rank]
        self.ordinal = Grade.ORDINALS[self.rank]
    
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
    GRID_SHAPE = (18, 11)
    _BLANK = np.zeros(GRID_SHAPE, dtype=np.uint8)
    BOS, EOS, SEP = '<P>', '</P>', '.'

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
        # Save various representations of holds
        self.array_3d = self._to_3darray()
        self.array = np.sum(self.array_3d, axis=-1, dtype=np.uint8)
        self.sentence = self._to_sentence()

    def __repr__(self):
        return f'Problem(name={self.name}, grade={self.grade}, setter={self.setter}, rating={self.rating}, repeats={self.repeats}, benchmark={self.benchmark}, master={self.master}, assessment={self.assessment}, type={self.type}, holds={self.sentence})'

    def _to_3darray(self):
        array = np.stack([Problem._BLANK, Problem._BLANK, Problem._BLANK], axis=-1)
        for i, holds in enumerate([self.holds_start, self.holds_intermed, self.holds_end]):
            for hold in holds:
                row = Problem._BLANK.shape[0] - int(hold[1:len(hold)])  # Holds are on a grid from 'A18' (18=bottom A=left) to 'K1' (1=top K=right).
                col = ord(hold[0].upper()) - ord('A')
                array[row, col, i] = 1
        return array

    def _to_sentence(self):
        return [Problem.BOS, *self.holds_start, Problem.SEP, *self.holds_intermed, Problem.SEP, *self.holds_end, Problem.SEP, Problem.EOS]