class Node:
    def __init__(self, value) -> None:
        self.left = None
        self.right = None
        self.value = value

    def insert(self, value: int) -> None:
        if value < self.value:
            if self.left is None:
                self.left = Node(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = Node(value)
            else:
                self.right.insert(value)

    def print_tree(self, depth=0) -> None:
        if self.right:
            self.right.print_tree(depth+1)
        print("   "*depth, self.value)
        if self.left:
            self.left.print_tree(depth+1)

root = Node(10)

# Insert some nodes
root.insert(6)
root.insert(15)
root.insert(3)
root.insert(8)
root.insert(12)
root.insert(18)

# Print the tree
root.print_tree()