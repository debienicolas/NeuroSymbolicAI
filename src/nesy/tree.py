class Node:
    def __init__(self, kind, children, value=None):
        self.kind = kind
        self.value = value
        self.children = children

    def __repr__(self):
        if self.kind == "Leaf":
            return f"{self.kind}({self.value})"
        else:
            return f"{self.kind}({self.children})"
