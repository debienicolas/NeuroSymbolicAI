from abc import ABC, abstractmethod
import torch

class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass


# from page 21 of  "From Statistical Relational to Neurosymbolic Artificial Intelligence"
class SumProductSemiring(Semantics):

    def conjunction(self, a, b):
        return a * b

    def disjunction(self, a, b):
        return a + b

    def negation(self, a):
        return 1 - a
    

class LukasieviczTNorm(Semantics):

    def conjunction(self, a, b):
        return max(torch.tensor(0), a + b - 1)

    def disjunction(self, a, b):
        return min(torch.tensor(1), a + b)

    def negation(self, a):
        return 1 - a


class GodelTNorm(Semantics):

    def conjunction(self, a, b):
        return min(a, b)

    def disjunction(self, a, b):
        return max(a, b)

    def negation(self, a):
        return 1 - a


class ProductTNorm(Semantics):

    def conjunction(self, a, b):
        return a * b

    def disjunction(self, a, b):
        return a + b - (a * b)

    def negation(self, a):
        return 1 - a