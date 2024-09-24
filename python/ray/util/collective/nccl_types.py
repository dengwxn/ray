from enum import Enum


class CollectiveOp(Enum):
    pass


class ReduceOp(CollectiveOp):
    SUM = 0
    PRODUCT = 1
    MAX = 2
    MIN = 3
    AVG = 4

    def __str__(self):
        return f"{self.name.lower()}"