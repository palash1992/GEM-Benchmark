import numpy as np
import networkx as nx
import math
import random


def convert(something):  # use networkx conversion from numpy array
    # g = nx.from_numpy_matrix(someNPMat)
    # print(type(something))
    g = nx.from_numpy_array(something)
    return g


def deleteSelfLoops(graph, nNodes):  # used to take away self loops in final graph for stat purposes
    nNodes = int(nNodes)
    for i in range(nNodes):
        for j in range(nNodes):
            if (i == j):
                graph[i, j] = 0
    return graph


def generateStochasticKron(initMat, k, deleteSelfLoopsForStats=False, directed=False, customEdges=False, edges=0):
    initN = initMat.getNumNodes()
    nNodes = int(math.pow(initN, k))  # get final size and make empty 'kroned' matrix
    mtxDim = initMat.getNumNodes()
    mtxSum = initMat.getMtxSum()
    if (customEdges == True):
        nEdges = edges
        if (nEdges > (nNodes * nNodes)):
            raise ValueError("More edges than possible with number of Nodes")
    else:
        nEdges = math.pow(mtxSum, k)  # get number of predicted edges
    collisions = 0

    # create vector for recursive matrix probability
    probToRCPosV = []
    cumProb = 0.0
    for i in range(mtxDim):
        for j in range(mtxDim):
            prob = initMat.getValue(i, j)
            if (prob > 0.0):
                cumProb += prob
                probToRCPosV.append((cumProb / mtxSum, i, j))
                # print "Prob Vector Value:" #testing
                # print cumProb/mtxSum #testing

    # add Nodes

    finalGraph = np.zeros((nNodes, nNodes))
    # add Edges
    e = 0
    # print nEdges #testing
    while (e < nEdges):
        rng = nNodes
        row = 0
        col = 0

        for t in range(int(k)):
            prob = random.uniform(0, 1)
            # print "prob:" #testing
            # print prob #testing
            n = 0
            while (prob > probToRCPosV[n][0]):
                n += 1
            mrow = probToRCPosV[n][1]
            mcol = probToRCPosV[n][2]
            rng /= mtxDim
            row += mrow * rng
            col += mcol * rng

        row = int(row)
        col = int(col)
        if (finalGraph[row, col] == 0):  # if there is no edge
            finalGraph[row, col] = 1
            e += 1
            if (not directed):  # symmetry if not directed
                if (row != col):
                    finalGraph[col, row] = 1
                    e += 1
        else:
            collisions += 1
    # delete self loops if needed for stats

    if (deleteSelfLoopsForStats):
        finalGraph = deleteSelfLoops(finalGraph, nNodes)
    finalGraph = convert(finalGraph)

    return finalGraph