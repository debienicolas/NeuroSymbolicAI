import re
import torch
from functools import reduce
from nesy.semantics import SumProductSemiring, LukasieviczTNorm,GodelTNorm,ProductTNorm

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_tree, queries):
        # TODO: Implement this
        result = []
        #print("Queries: ", queries)
        #print("And or tree: ", and_or_tree)
        for i in and_or_tree:
            result.append(self.__traverse_and_or_tree(i,tensor_sources))
            # print(i)
            # print("Return traversal: ", self.__traverse_and_or_tree(i,tensor_sources))
        return torch.stack(result)

        # Our dummy And-Or-Tree (addition(img0, img1,0) is represented by digit(img0,0) AND digit(img1,0)
        # The evaluation is:
        # p(addition(img0, img1,0)) = p(digit(img0,0) AND digit(img1,0)) =
        p_digit_0_0 = self.neural_predicates["digit"](tensor_sources["images"][:,0])[:,0]
        p_digit_1_0 = self.neural_predicates["digit"](tensor_sources["images"][:,1])[:,0]
        p_sum_0 =  p_digit_0_0 * p_digit_1_0

        # Here we trivially return the same value (p_sum_0[0]) for each of the queries to make the code runnable
        if isinstance(queries[0], list):
            res = [torch.stack([p_sum_0[0] for q in query]) for query in queries]
        else:
            res = [p_sum_0[0] for query in queries]
        return torch.stack(res)

    def __traverse_and_or_tree(self, node, tensor_sources):
        if node.kind == "Or":
            values = [self.__traverse_and_or_tree(n,tensor_sources) for n in node.children]
            return reduce(self.label_semantics.disjunction, values)
        elif node.kind == "And":
            values = [self.__traverse_and_or_tree(n,tensor_sources) for n in node.children]
            return reduce(self.label_semantics.conjunction, values)
        elif node.kind == "Neg":
            return self.label_semantics.negation(node.children[0])
        elif node.kind == "Leaf":
            image_number = int(node.value.arguments[0].arguments[1].functor)
            # print("Leaf: ",self.neural_predicates[node.value.functor](tensor_sources["images"][:,image_number])[:,0])
            return self.neural_predicates[node.value.functor](tensor_sources["images"][:,image_number])[:,0]