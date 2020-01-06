class Node:
    def __init__(self, output, edge, name, list=None):
        self.__children = []
        self.__output = output
        self.__edge = edge
        self.__name = name
        self.__list = list

    def getOutput(self):
        return self.__output

    def setOutput(self, output):
        self.__output = output

    def getEdge(self):
        return self.__edge

    def setEdge(self, edge):
        self.__edge = edge

    def getChildren(self):
        return self.__children

    def addChild(self, node):
        self.__children.append(node)

    def getName(self):
        return self.__name

    def getList(self):
        return self.__list


class Tree:
    def __init__(self, root):
        self.__root = root
        self.__current = root

    def getCurrent(self):
        return self.__current
