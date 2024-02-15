class Node:
    def __init__(self, kind, children=[], value=None):
        self.kind = kind
        self.value = value
        self.children = children