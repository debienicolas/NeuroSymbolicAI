import enum
from math import comb
from scipy import constants
from sympy import substitution
from nesy.term import Term, Clause, Fact, Variable
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import product
from nesy.tree import Node
from itertools import combinations_with_replacement, product, permutations
import time

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass

class ForwardChaining(LogicEngine):

    def getVarsClause(self, clause):
        return set([var for var in clause.head.arguments if isinstance(var, Variable)] + [var for term in clause.body for var in term.arguments if isinstance(var, Variable)])
    
    def getConstants(self, known_terms):
        constants = set()
        for term in known_terms:
            assert isinstance(term, Term)
            if term.arguments != () and not all([isinstance(arg, Variable) for arg in term.arguments]):
                # Hardcoded assumptions: not constant in a neural predicate 
                if isinstance(term, Term) and term.functor == "digit":
                    continue
                for i,arg in enumerate(term.arguments):
                    if i == len(term.arguments) - 1:
                        continue
                    constants.add(arg)
        return constants
    
    def generateSubstitutions(self, variables, constants):
        
        substitutions = []
        combos = []
        for combo in product(constants, repeat=len(variables)):
            combos.append(combo)
            substitutions.append(dict(zip(variables, combo)))
        return substitutions

    def apply_substitution(self, substitution, term):
        if isinstance(term, Variable):
            return substitution.get(term, term)
        elif isinstance(term, Term):
            new_args = [self.apply_substitution(substitution, arg) for arg in term.arguments]
            return Term(term.functor, new_args)
        else:
            # return unchanged if not a variable or term
            return term
    
    def isNeuralPredicate(self, term, program):
        neural_terms_program = [term.term for term in program if isinstance(term, Fact) and term.weight is not None]
        return term in neural_terms_program

    def reason(self, program: tuple[Clause], queries: list[Term]):
        return self.reasonEfficient(program, queries)
        start = time.time()
        trees = []

        for query in queries:

            # initialize the and-or tree
            and_or_tree = Node("Or", [])
        
            known_terms = [] # store inferred facts
            inferred_terms = [] # store inferred facts

            # initialize the known terms 
            for clause in program:
                if isinstance(clause, Fact):
                    if clause.weight is None:
                        known_terms.append(clause.term)
                    else:
                        known_terms.append(clause.term)
            # # add the query as a term
            # for query in queries:
            #     known_terms.append(query)
            assert all([isinstance(term, Term) for term in known_terms])
            #print("Known terms: ", known_terms)

            clauses = [clause for clause in program if isinstance(clause, Clause)]
            assert all([isinstance(clause, Clause) for clause in clauses])

            cont = True
            while cont:
                cont = False
                # for every clause check if all the premises are satisfied
                for clause in clauses:
                    #print("Clause: ", clause)
                    clause_vars = self.getVarsClause(clause)
                    #print("Clause vars: ", clause_vars)
                    assert all([isinstance(var, Variable) for var in clause_vars])

                    constants = self.getConstants(known_terms)
                    #print("Constants: ", constants)
                    assert all([isinstance(term, Term) for term in constants])

                    substitutions = self.generateSubstitutions(clause_vars, constants)
                    #print(len(substitutions))
                    assert all([isinstance(value,Term) for value in substitutions[0].values()])
                    

                    body = [term for term in clause.body]
                    assert all([isinstance(term, Term) for term in body])
                    
                    for substitution in substitutions:
                        
                        # substitution = {Variable('X'): Term('tensor', [Variable('images'), 0]), Variable('Y'): Term('tensor', [Variable('images'), 1]), Variable('Z'): Term(1,()), Variable('N1'): Term(0,()), Variable('N2'): Term(1,())}
                        substituted_body = [self.apply_substitution(substitution, term) for term in body]
                        assert all([isinstance(term, Term) for term in substituted_body])
                        # check if all the premises are satisfied                
                        if all([term in known_terms for term in substituted_body]):    
                            inferred_fact = self.apply_substitution(substitution, clause.head)
                            # if inferred_fact not in known_terms and inferred_fact not in inferred_terms:
                            #     inferred_terms.append(inferred_fact)
                            #     known_terms.append(inferred_fact)
                            #     cont = True
                            if inferred_fact == query:
                                
                                # add the inferred fact to the and-or tree
                                # only add the terms in the body that are neural predicates
                                # Assumption that the tree only has leaf then And and then Or
                                and_or_tree.children.append(
                                    Node("And", [Node("Leaf", [], value=term) for term in substituted_body if self.isNeuralPredicate(term, program)])
                                )
                                cont = False
            trees.append(and_or_tree)    

        print("Time reason: ", time.time() - start)
        return trees
    
    def getVarsTerm(self, term):
        return [var for var in term.arguments if isinstance(var, Variable)]
    
    def getConstrainedSubstitutions(self,clause, query):
        # check the clause and see if the query can constrain the substitution
        # check if query functor is in the clause head
        if query.functor == clause.head.functor:
            # take the variables in the clause head and create a dictionary with the query arguments
            head_vars = self.getVarsTerm(clause.head)
            query_constant = [arg for arg in query.arguments if not isinstance(arg, Variable)]
            constr_sub = {var: query_constant[i] for i, var in enumerate(head_vars)}
            
            return constr_sub
    
    def reasonEfficient(self, program: tuple[Clause], queries: list[Term]):
        # look at the query and perform that substitution
        #start = time.time()
        trees = []

        for query in queries:

            # initialize the and-or tree
            and_or_tree = Node("Or", [])
        
            known_terms = [] # store inferred facts
            inferred_terms = [] # store inferred facts

            # initialize the known terms 
            for clause in program:
                if isinstance(clause, Fact):
                    if clause.weight is None:
                        known_terms.append(clause.term)
                    else:
                        known_terms.append(clause.term)

            clauses = [clause for clause in program if isinstance(clause, Clause)]

            cont = True
            while cont:
                
                # check every clause
                
                for clause in clauses:
                    
                    # check if the query can constrain the substitution
                    constrained_subs = self.getConstrainedSubstitutions(clause, query)

                    all_vars = self.getVarsClause(clause)
                    assert all([isinstance(var, Variable) for var in all_vars])
                    empty_vars = [var for var in all_vars if var not in constrained_subs.keys()]
                    assert all([isinstance(var, Variable) for var in empty_vars])


                    constants = self.getConstants(known_terms)
                    assert all([isinstance(term, Term) for term in constants])

                    substitutions = self.generateSubstitutions(empty_vars, constants)

                    body = [term for term in clause.body]

                    # substitution = {Variable('A'): Term('tensor', [Variable('images'), 0]), Variable('B'): Term('tensor',  [Variable('images'), 1]),Variable('C'): Term('tensor', [Variable('images'), 0]), Variable('Z'): Term(0,()), Variable('N1'): Term(0,()), Variable('N2'): Term(0,()),Variable('N3'): Term(0,())}
                    
                    for substitution in substitutions:
                        

                        # merge the constrained substitution with the generated substitution
                        substitution.update(constrained_subs)
                        substituted_body = [self.apply_substitution(substitution, term) for term in body]
                        assert all([isinstance(term, Term) for term in substituted_body])
                        # check if all the premises are satisfied                
                        if all([term in known_terms for term in substituted_body]):    
                            inferred_fact = self.apply_substitution(substitution, clause.head)
                            
                            if inferred_fact == query:
                                
                                and_or_tree.children.append(
                                    Node("And", [Node("Leaf", [], value=term) for term in substituted_body if self.isNeuralPredicate(term, program)])
                                )
                                cont = False
            trees.append(and_or_tree)
        #print("Time reason efficient: ", time.time() - start)
        return trees


                    


                    

