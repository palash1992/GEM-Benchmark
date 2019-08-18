try: import cPickle as pickle
except: import pickle
from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util
import networkx as nx
import numpy as np


def evaluateStaticGraphReconstruction(digraph, graph_embedding,
                                      X_stat, node_l=None, file_suffix=None,
                                      sample_ratio_e=None, is_undirected=True,
                                      is_weighted=False):
    """This function evaluates the graph reconstruction accuracy of the embedding algorithms.
        Args:
            digraph (Object): directed networkx graph object.
            graph_embedding (object): Object of the embedding algorithm class defined in gemben/embedding.
            X_stat (Vector): Embedding of the the nodes of the graph.
            node_l (Int): Number of nodes in the graph.
            file_suffix (Str): The name of the algorithm and dataset used to save the embedding.
            sample_ratio_e (Float): The ratio used to sample the original graph for evaluation purpose.
            is_undirected (bool): Boolean flag to denote whether the graph is directed or not.
            is_weighted (bool): Boolean flag to denote whether the edges of the graph is weighted.
        Returns:
            Numpy Array: Consiting of Mean average precision precision curve, errors and error baselines.
    """
    node_num = digraph.number_of_nodes()
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(
            node_num,
            sample_ratio_e,
            is_undirected
        )
    else:
        eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = graph_embedding.get_reconstructed_adj(X_stat, node_l)
    else:
        estimated_adj = graph_embedding.get_reconstructed_adj(
            X_stat,
            file_suffix,
            node_l
        )
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    if 'partition' in digraph.node[0]:
        predicted_edge_list = [e for e in predicted_edge_list if digraph.node[e[0]]['partition'] != digraph.node[e[1]]['partition']]

    MAP = metrics.computeMAP(predicted_edge_list, digraph)
    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)
    # If weighted, compute the error in reconstructed weights of observed edges
    if is_weighted:
        digraph_adj = nx.to_numpy_matrix(digraph)
        estimated_adj[digraph_adj == 0] = 0
        err = np.linalg.norm(digraph_adj - estimated_adj)
        err_baseline = np.linalg.norm(digraph_adj)
    else:
        err = None
        err_baseline = None
    return (MAP, prec_curv, err, err_baseline)


def expGR(digraph, graph_embedding,
          X, n_sampled_nodes_l, rounds,
          res_pre, m_summ,
          K=10000,
          is_undirected=True,
          sampling_scheme="u_rand"):

    """This function is used to experiment graph reconstruction.
        Args:
            digraph (Object): directed networkx graph object.
            graph_embedding (object): Object of the embedding algorithm class defined in gemben/embedding.
            X (Vector): Embedding of the the nodes of the graph.
            n_sampled_node_l (Int): Number of nodes in the graph.
            rounds (Int): The number of times the graph reconstruction is performed.
            res_pre (Str): Prefix to be used to save the result.
            m_summ (Str): String to denote the name of the summary file. 
            K (Int): The maximum value to be use to get the precision curves.
            sampling_scheme (Str): Sampling schme used to sample nodes to be reconstructed.
            is_undirected (bool): Boolean flag to denote whether the graph is directed or not.
        Returns:
            Numpy Array: Consisting of Mean average precision.
    """
    print('\tGraph Reconstruction')
    summ_file = open('%s_%s_%s.grsumm' % (res_pre, m_summ, sampling_scheme), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    n_sample_nodes_l = [min(int(n), digraph.number_of_nodes()) for n in n_sample_nodes_l]
    if not n_sample_nodes_l:
        n_sample_nodes_l = [node_num]
    MAP = {}
    prec_curv = {}
    err = {}
    err_b = {}
    n_nodes = {}
    n_edges = {}
    # if digraph.number_of_nodes() <= n_sampled_nodes:
    #     rounds = 1
    for n_s in n_sampled_nodes_l:
        n_s = int(n_s)
        MAP[n_s] = [None] * rounds
        prec_curv[n_s] = [None] * rounds
        err[n_s] = [None] * rounds
        err_b[n_s] = [None] * rounds
        n_nodes[n_s] = [None] * rounds
        n_edges[n_s] = [None] * rounds
        for rid in range(rounds):
            if sampling_scheme == "u_rand":
                sampled_digraph, node_l = graph_util.sample_graph(
                    digraph,
                    n_sampled_nodes=n_s
                )
            else:
                sampled_digraph, node_l = graph_util.sample_graph_rw(
                    digraph,
                    n_sampled_nodes=n_s
                )
            n_nodes[n_s][rid] = sampled_digraph.number_of_nodes()
            n_edges[n_s][rid] = sampled_digraph.number_of_edges()
            print('\t\tRound: %d/%d, n_nodes: %d, n_edges:%d\n' % (rid,
                                                                   rounds,
                                                                   n_nodes[n_s][rid],
                                                                   n_edges[n_s][rid]))
            sampled_X = X[node_l]
            MAP[n_s][rid], prec_curv[n_s][rid], err[n_s][rid], err_b[n_s][rid] = \
                evaluateStaticGraphReconstruction(sampled_digraph, graph_embedding,
                                                  sampled_X, node_l,
                                                  is_undirected=is_undirected)
            prec_curv[n_s][rid] = prec_curv[n_s][rid][:K]
        summ_file.write('n_s:%d' % n_s)
        try:
            summ_file.write('\tErr: %f/%f\n' % (np.mean(err[n_s]), np.std(err[n_s])))
            summ_file.write('\tErr_b: %f/%f\n' % (np.mean(err_b[n_s]), np.std(err_b[n_s])))
        except TypeError:
            pass
        summ_file.write('\t%f/%f\t%s\n' % (np.mean(MAP[n_s]), np.std(MAP[n_s]),
                                           metrics.getPrecisionReport(prec_curv[n_s][0],
                                                                      n_edges[n_s][0])))
    pickle.dump([n_nodes,
                 n_edges,
                 MAP,
                 prec_curv,
                 err,
                 err_b,
                 n_sampled_nodes_l],
                open('%s_%s_%s.gr' % (res_pre, m_summ, sampling_scheme), 'wb'))
    return MAP[list(MAP.keys())[0]]
