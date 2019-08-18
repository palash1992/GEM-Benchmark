import numpy as np
import networkx as nx


class InitMatrix():

    def __init__(self, numNodes, W=None):
        self.numNodes = numNodes
        # initially we will take in the number of nodes when the object is created

    def getNumNodes(self):
        return self.numNodes

    def setNumNodes(self, v):
        self.numNodes = v

    def getValue(self, node1, node2):
        return self.W[node1, node2]

    def setValue(self, newVal, node1, node2):
        self.W[node1, node2] = newVal

    def getMtxSum(self):
        n = self.getNumNodes()
        s = 0.0
        for i in range(n):
            for j in range(n):
                s += self.getValue(i, j)
        int(s)
        return s

    def make(self):  # This makes a init matrix manual (user adds edges)
        n = self.numNodes  # getNumNodes(self)
        initMat = np.zeros((n, n))  # Creates corret size of init matrix with all 0s
        self.W = initMat

    def makeStochasticCustom(self, probArr):  # takes np array of probs for each position in init matrix
        n = self.numNodes
        length = n * n
        if (probArr.shape[0] != length):
            raise IOError("Your array must be the length of postitions in your initMatrix")
        self.W = probArr.reshape((n,n))
        # for i in range(n):
        #     for j in range(n):
        #         for k in range(length):
        #             self.setValue(probArr[k], i, j)

    def makeStochasticAB(self, alpha, beta, selfloops=True):
        # parm check
        if (not (0.00 <= alpha <= 1.00)):
            raise IOError("alpha (arguement 1) must be a value equal to or between 0 and 1; it is a probability")
        if (not (0.00 <= beta <= 1.00)):
            raise IOError("beta (arguement 2) must be a value equal to or between 0 and 1; it is a probability")

        n = self.getNumNodes()

        # switch 1s and 0s for alpha and beta, keep self loops
        for i in range(n):
            for j in range(n):
                if (i == j):
                    if (selfloops == False):
                        self.setValue(alpha, i, j)
                elif (self.getValue(i, j) == 0):
                    self.setValue(beta, i, j)
                else:
                    self.setValue(alpha, i, j)

    def makeStochasticABFromNetworkxGraph(self, nxgraph, alpha,
                                          beta):  # takes a nxgraph, alpha, and beta. Returns stochastic initMatrix.
        adjMatrix = nx.to_numpy_matrix(nxgraph)  # return graph adj matrix as a np matrix

        n = adjMatrix.shape[0]  # get num nodes

        init = InitMatrix(n)
        init.make()
        for i in range(n):
            for j in range(n):
                init.setValue(adjMatrix[i, j], i, j)
        init.makeStochasticAB(alpha, beta)

        return init  # there is no gaurentee of self loops since these are other graph types generated as seeds

    def makeFromNetworkxGraph(self, nxgraph):  # takes a nxgraph, Returns initMatrix.
        adjMatrix = nx.to_numpy_matrix(nxgraph)  # return graph adj matrix as a np matrix

        n = adjMatrix.shape[0]  # get num nodes

        init = InitMatrix(n)
        init.make()
        for i in range(n):
            for j in range(n):
                init.setValue(adjMatrix[i, j], i, j)

        return init  # there is no gaurentee of self loops since these are other graph types generated as seeds

    def addEdge(self, node1, node2, edge=1):
        node1 = int(node1)
        node2 = int(node2)
        if edge == 0 or edge == float('inf'):
            raise ValueError("Cannot add a zero or infinite edge")

        self.W[node1, node2] = edge

    def addSelfEdges(self):
        n = self.getNumNodes()
        for i in range(n):
            self.addEdge(i, i)