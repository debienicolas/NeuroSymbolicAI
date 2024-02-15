class Node:
    def __init__(self, kind, children, value=None):
        self.kind = kind
        self.value = value
        self.children = children

    def __repr__(self):
        return f"{self.kind}({self.children})"
