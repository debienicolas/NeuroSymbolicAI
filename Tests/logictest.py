from nesy.model import NeSyModel, MNISTEncoder
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring
from nesy.parser import parse_program, parse_term
from itertools import product
from nesy.term import Fact


def test_forward_chaining_1():
    program_string = "addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z).\n"
    program_string += "\n".join(
        [f"add({x}, {y}, {x + y})." for x in range(2) for y in range(2)])
    program_string += "\n"
    program_string += "\n".join(
        [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
            product(range(2), range(2))])
    program = parse_program(program_string)

    query = parse_term("addition(tensor(images,0), tensor(images,1), 0)")
    return ForwardChaining().reason(program, [query])
    
def test_forward_chaining_2():
    program_string = "addition(X,Y,1) :- digit(X,0), digit(Y,1).\n"
    program_string += "addition(X,Y,1) :- digit(X,1), digit(Y,0).\n"
    program_string += "\n".join(
        [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
            product(range(2), range(2))])
    program = parse_program(program_string)

    query = parse_term("addition(tensor(images,0), tensor(images,1), 1)")
    return ForwardChaining().reason(program, [query])

def test_foward_chaining_multiple_queries():
    program_string = "addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z).\n"
    program_string += "\n".join(
        [f"add({x}, {y}, {x + y})." for x in range(2) for y in range(2)])
    program_string += "\n"
    program_string += "\n".join(
        [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
            product(range(2), range(2))])
    program = parse_program(program_string)

    queries = [
        parse_term("addition(tensor(images,0), tensor(images,1), 0)"),
        parse_term("addition(tensor(images,1), tensor(images,0), 0)")
    ]
    return ForwardChaining().reason(program, queries)

print(test_forward_chaining_1())
# print(test_forward_chaining_2())
# print(test_foward_chaining_multiple_queries())


def test_reasonEfficient_1():
    program_string = "addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z).\n"
    program_string += "\n".join(
        [f"add({x}, {y}, {x + y})." for x in range(2) for y in range(2)])
    program_string += "\n"
    program_string += "\n".join(
        [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
            product(range(2), range(2))])
    program = parse_program(program_string)

    query = parse_term("addition(tensor(images,0), tensor(images,1), 0)")
    return ForwardChaining().reasonEfficient(program, [query])

print(test_reasonEfficient_1())