class Node:
    def __init__(self, x=0, y=0):
        self.x=x
        self.y=y

firstNode = Node(1, 2)
second = Node(3, 4)

print("Node val: " + str(firstNode.x))