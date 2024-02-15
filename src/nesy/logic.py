from nesy.term import Term, Clause, Fact, Variable
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import product
from nesy.tree import Node
from itertools import combinations_with_replacement, product, permutations

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
                for arg in term.arguments:
                    constants.add(arg)
        return constants
    
    def generateSubstitutions(self, variables, constants):
        
        substitutions = []
        for combo in product(constants, repeat=len(variables)):
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

        return trees

