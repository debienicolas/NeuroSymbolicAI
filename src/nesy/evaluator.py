import re
import torch
from functools import reduce
from nesy.semantics import SumProductSemiring, LukasieviczTNorm,GodelTNorm,ProductTNorm
import time

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_tree, queries, index, train=False):
        # TODO: Implement this
        #start = time.time()
        # print("Tesnor sources: ", tensor_sources["images"].shape)
        # print("tensor indexed: ", tensor_sources["images"][0].shape)
        # print(tensor_sources["images"][0][0].shape)
        # print(self.neural_predicates["digit"](tensor_sources["images"][0][0]))
        # print(tensor_sources["images"][0][0][:,0].shape)
        results = []
        
        for i,tree in enumerate(and_or_tree):
            if train:
                index = i
            result = self.__traverse_and_or_tree(tree,tensor_sources,index)
            results.append(result)
            # print(i)
            # print("Return traversal: ", self.__traverse_and_or_tree(i,tensor_sources))
        #print("Time evaluation: ", time.time() - start)
        return torch.stack(results)

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

    def __traverse_and_or_tree(self, node, tensor_sources, index):
        if node.kind == "Or":
            values = [self.__traverse_and_or_tree(n,tensor_sources, index) for n in node.children]
            return reduce(self.label_semantics.disjunction, values)
        elif node.kind == "And":
            values = [self.__traverse_and_or_tree(n,tensor_sources,index) for n in node.children]
            return reduce(self.label_semantics.conjunction, values)
        elif node.kind == "Neg":
            return self.label_semantics.negation(node.children[0])
        elif node.kind == "Leaf":
            image_number = int(node.value.arguments[0].arguments[1].functor)
            n_class = int(node.value.arguments[1].functor)
            
            return self.neural_predicates[node.value.functor](tensor_sources["images"][index,image_number])[n_class]