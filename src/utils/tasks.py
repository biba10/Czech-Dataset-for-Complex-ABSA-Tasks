from enum import Enum


class Task(Enum):
    APD = "apd"
    ATE = "ate"
    ACD = "acd"
    E2E = "e2e"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
