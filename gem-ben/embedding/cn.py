disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util, evaluation_util
from gem.evaluation import visualize_embedding as viz


class CommonNeighbors(StaticGraphEmbedding):

    """`Common Neighbors`_.
    Common Neighbors defines the similarity between nodes 
    as the number of common neighbors between them.
    
    Args:
        hyper_dict (object): Hyper parameters.
        kwargs (dict): keyword arguments, form updating the parameters
    
    Examples:
        >>> from gemben.embedding.cn import CommonNeighbors
        >>> edge_f = 'data/karate.edgelist'
        >>> G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
        >>> G = G.to_directed()
        >>> res_pre = 'results/testKarate'
        >>> graph_util.print_graph_stats(G)
        >>> t1 = time()
        >>> embedding = CommonNeighbors(4, 0.01)
        >>> embedding.learn_embedding(graph=G, edge_f=None,
                                  is_weighted=True, no_python=True)
        >>> print('Common Neighbors:\n\tTraining time: %f' % (time() - t1))
    .. _Common Neighbors:
        https://arxiv.org/pdf/cond-mat/0104209.pdf
    """

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the AdamicAdar class

        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        '''
        hyper_params = {
            'method_name': 'common_neighbors'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        self._G = graph.to_undirected()
        return None, 0

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return len(list(nx.common_neighbors(self._G, i, j)))

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._G.number_of_nodes()
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


if __name__ == '__main__':
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = CommonNeighbors(4, 0.01)
    embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
    print('Common Neighbors:\n\tTraining time: %f' % (time() - t1))
