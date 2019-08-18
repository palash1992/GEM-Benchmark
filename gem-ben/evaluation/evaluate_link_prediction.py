try: import cPickle as pickle
except: import pickle
from gem.evaluation import metrics
from gem.utils import evaluation_util, graph_util
import numpy as np
import networkx as nx
import pdb
from time import time

import sys
sys.path.insert(0, './')
from gem.utils import embed_util


def evaluateStaticLinkPrediction(train_digraph, test_digraph,
                                 graph_embedding, X,
                                 node_l=None,
                                 sample_ratio_e=None,
                                 is_undirected=True,
                                store_predictions=1):
    """This function evaluates the static link prediction accuracy of the embedding algorithms.
        Args:
            train_digraph (Object): directed networkx graph object used for training the algorithm.
            test_digraph (Object): directed networkx graph object to be used for testing the algorithm.
            graph_embedding (object): Object of the embedding algorithm class defined in gemben/embedding.
            X (Vector): Embedding of the the nodes of the graph.
            node_l (Int): Number of nodes in the graph.
            sample_ratio_e (Float): The ratio used to sample the original graph for evaluation purpose.
            is_undirected (bool): Boolean flag to denote whether the graph is directed or not.
            store_prediction (Int): Stores the predicted values.
        Returns:
            Numpy Array: Consiting of Mean average precision and the precision curve values.
    """
    node_num = train_digraph.number_of_nodes()
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(
            node_num,
            sample_ratio_e,
            is_undirected
        )
    else:
        eval_edge_pairs = None
    if X is None:
        # If not an embedding approach, store the new subgraph
        graph_embedding.learn_embedding(train_digraph)
    estimated_adj = graph_embedding.get_reconstructed_adj(X, node_l)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    filtered_edge_list = [e for e in predicted_edge_list if not train_digraph.has_edge(e[0], e[1])]
    if 'partition' in train_digraph.node[0]:
        filtered_edge_list = [e for e in predicted_edge_list if train_digraph.node[e[0]]['partition'] != train_digraph.node[e[1]]['partition']]
    pickle.dump(filtered_edge_list, open('gem/nodeListMap/preds.pickle', 'wb'))
    pickle.dump(test_digraph, open('gem/nodeListMap/test_graph.pickle', 'wb'))
    t1 = time()
    MAP = metrics.computeMAP(filtered_edge_list, test_digraph)
    t2 = time()
    prec_curv, _ = metrics.computePrecisionCurve(
        filtered_edge_list,
        test_digraph
    )
    t3 = time()
    print('MAP computation time: %f sec, prec: %f sec' % (t2 - t1, t3 - t2))
    return (MAP, prec_curv)


def expLPT(digraph, graph_embedding,
           res_pre, m_summ,
           K=100000,
           is_undirected=True):
    """This function is used to experiment graph reconstruction for temporally varying graphs.
        Args:
            digraph (Object): directed networkx graph object.
            graph_embedding (object): Object of the embedding algorithm class defined in gemben/embedding.
            res_pre (Str): Prefix to be used to save the result.
            m_summ (Str): String to denote the name of the summary file. 
            K (Int): The maximum value to be use to get the precision curves.
            is_undirected (bool): Boolean flag to denote whether the graph is directed or not.
    """
    print('\tLink Prediction Temporal')

    t1 = time()
    # learn graph embedding on whole graph
    X, _ = graph_embedding.learn_embedding(graph=digraph)
    t2 = time()
    print('\t\tTime taken to learn the embedding: %f sec' % (t2 - t1))
    estimated_adj = graph_embedding.get_reconstructed_adj(X)
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected
    )
    filtered_edge_list = [e for e in predicted_edge_list if not digraph.has_edge(e[0], e[1])]
    if 'partition' in digraph.node[0]:
        filtered_edge_list = [e for e in predicted_edge_list if digraph.node[e[0]]['partition'] != digraph.node[e[1]]['partition']]
    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)
    print('\t\tPredicted edge list computed in %f sec. Saving edge list.' % (time() - t2))
    pickle.dump(
        sorted_edges[:K],
        open('%s_%s_predEdgeList.pickle' % (res_pre, m_summ), 'wb')
    )
    print('\t\tSaved edge list.')


def expLP(digraph, graph_embedding,
          n_sample_nodes_l, rounds,
          res_pre, m_summ, train_ratio=0.8,
          no_python=True, K=32768,
          is_undirected=True, sampling_scheme="u_rand"):
    print('\tLink Prediction')
    MAP = {}
    prec_curv = {}
    n_sample_nodes_l = [min(int(n), digraph.number_of_nodes()) for n in n_sample_nodes_l]

    # Randomly hide (1-train_ratio)*100% of links
    node_num = digraph.number_of_nodes()
    train_digraph, test_digraph = evaluation_util.splitDiGraphToTrainTest(
        digraph,
        train_ratio=train_ratio,
        is_undirected=is_undirected
    )

    # Ensure the resulting train subgraph is connected
    if not nx.is_connected(train_digraph.to_undirected()):
        train_digraph = max(
            nx.weakly_connected_component_subgraphs(train_digraph),
            key=len
        )
        tdl_nodes = train_digraph.nodes()
        nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
        train_digraph = nx.relabel_nodes(train_digraph, nodeListMap, copy=True)
        test_digraph = test_digraph.subgraph(tdl_nodes)
        ### unfroze the graph
        test_digraph = nx.Graph(test_digraph)
        ####nx.relabel_nodes(test_digraph, nodeListMap, copy=False)
        test_digraph = nx.relabel_nodes(test_digraph, nodeListMap, copy=True)
        
    pickle.dump(nodeListMap, open('gem/nodeListMap/lp_lcc.pickle', 'wb'))

    t1 = time()
    # learn graph embedding on train subgraph
    print(
        'Link Prediction train graph n_nodes: %d, n_edges: %d' % (
            train_digraph.number_of_nodes(),
            train_digraph.number_of_edges())
    )
    X, _ = graph_embedding.learn_embedding(
        graph=train_digraph,
        no_python=no_python
    )
    if X is not None and X.shape[0] != train_digraph.number_of_nodes():
        pdb.set_trace()
    print('Time taken to learn the embedding: %f sec' % (time() - t1))

    # sample test graph for evaluation and store results
    node_l = None
    if not n_sample_nodes_l:
        n_sample_nodes_l = [node_num]
    summ_file = open('%s_%s_%s.lpsumm' % (res_pre, m_summ, sampling_scheme), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    for n_s in n_sample_nodes_l:
        n_s = int(n_s)
        n_s = min(n_s, train_digraph.number_of_nodes())
        MAP[n_s] = [None] * rounds
        prec_curv[n_s] = [None] * rounds
        for round_id in range(rounds):
            if sampling_scheme == "u_rand":
                train_digraph_s, node_l = graph_util.sample_graph(
                    train_digraph,
                    n_s
                )
            else:
                train_digraph_s, node_l = graph_util.sample_graph_rw(
                    train_digraph,
                    n_s
                )
            if X is not None:
                X_sub = X[node_l]
            else:
                X_sub = None
            test_digraph_s = test_digraph.subgraph(node_l)
            nodeListMap = dict(zip(node_l, range(len(node_l))))
            pickle.dump(nodeListMap, open('gem/nodeListMap/lp_lcc_samp.pickle', 'wb'))
            test_digraph_s = nx.relabel_nodes(test_digraph_s, nodeListMap, copy=True)
            MAP[n_s][round_id], prec_curv[n_s][round_id] = \
                evaluateStaticLinkPrediction(train_digraph_s, test_digraph_s,
                                             graph_embedding, X_sub,
                                             node_l=node_l,
                                             is_undirected=is_undirected)
            prec_curv[n_s][round_id] = prec_curv[n_s][round_id][:K]
        summ_file.write('\tn_s:%d, %f/%f\t%s\n' % (
            n_s,
            np.mean(MAP[n_s]),
            np.std(MAP[n_s]),
            metrics.getPrecisionReport(
                prec_curv[n_s][0],
                len(prec_curv[n_s][0])
            )
        ))
    summ_file.close()
    #if len(prec_curv[-1][0]) < 100:
        #pdb.set_trace()
    pickle.dump([MAP, prec_curv, n_sample_nodes_l],
                open('%s_%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme, str(train_ratio)),
                     'wb'))
    print('Link prediction evaluation complete. Time: %f sec' % (time() - t1))
    # prec_curv2 = [p[4096] for p in prec_curv[prec_curv.keys()[0]]]
    return MAP[list(MAP.keys())[0]]  # prec_curv2
