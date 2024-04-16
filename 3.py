def calculate_pos_on_tree(root, pos):
    if root is None:
        return pos
    else:
        return calculate_pos_on_tree(root.left, pos + 1) + calculate_pos_on_tree(root.right, pos + 1)
                                                                                 

ans()   