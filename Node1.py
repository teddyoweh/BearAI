class TreeNode:
    def __init__(self, val) -> None:
        self.left_child = None
        self.right_child = None
        self.value = val

    def insert(self, val: int) -> None:
        if val < self.value:
            if self.left_child is None:
                self.left_child = TreeNode(val)
            else:
                self.left_child.insert(val)
        else:
            if self.right_child is None:
                self.right_child = TreeNode(val)
            else:
                self.right_child.insert(val)

    def print_tree(self, depth=0) -> None:
        if self.right_child:
            self.right_child.print_tree(depth+1)
        print("   "*depth, self.value)
        if self.left_child:
            self.left_child.print_tree(depth+1)

root = TreeNode(10)
