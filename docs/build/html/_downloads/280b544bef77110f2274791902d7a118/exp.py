'''
=========================
Simple Experiment
=========================
A simple experiment to run the gemben library.
'''

try: import cPickle as pickle
except: import pickle
from time import time
from argparse import ArgumentParser
import importlib
import json
# import cPickle
import networkx as nx
import itertools
import pdb
import sys
import numpy as np
import pandas as pd
# sys.path.insert(0, './')
import os
from gemben.utils      import graph_util, plot_util
from gemben.evaluation import visualize_embedding as viz
from gemben.evaluation.evaluate_graph_reconstruction import expGR
from gemben.evaluation.evaluate_link_prediction import expLP, expLPT
from gemben.evaluation.evaluate_node_classification import expNC
from gemben.evaluation.visualize_embedding import expVis

methClassMap = {"gf": "GraphFactorization",
                "hope": "HOPE",
                "lap": "LaplacianEigenmaps",
                "node2vec": "node2vec",
                "sdne": "SDNE",
                "pa": "PreferentialAttachment",
                "rand": "RandomEmb",
                "cn": "CommonNeighbors",
                "aa": "AdamicAdar",
                "jc": "JaccardCoefficient"}
expMap = {"gf": "GF MAP", "lp": "LP MAP",
          "nc": "NC MAP"}


def learn_emb(MethObj, di_graph, params, res_pre, m_summ):
    if params["experiments"] == ["lp"]:
        X = None
    else:
        print('Learning Embedding: %s' % m_summ)
        if not bool(int(params["load_emb"])):
            X, learn_t = MethObj.learn_embedding(graph=di_graph,
                                                 edge_f=None,
                                                 no_python=True)
            print('\tTime to learn embedding: %f sec' % learn_t)
            pickle.dump(X, open('%s_%s.emb' % (res_pre, m_summ), 'wb'))
            pickle.dump(learn_t,
                        open('%s_%s.learnT' % (res_pre, m_summ), 'wb'))
        else:
            X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ),
                                 'rb'))
            try:
                learn_t = pickle.load(open('%s_%s.learnT' % (res_pre, m_summ),
                                           'rb'))
                print('\tTime to learn emb.: %f sec' % learn_t)
            except IOError:
                print('\tTime info not found')
    return X


def run_exps(MethObj, meth, dim, di_graph, data_set, node_labels, params):
    m_summ = '%s_%d' % (meth, dim)
    res_pre = "gemben/results/%s" % data_set
    n_r = params["rounds"]
    X = learn_emb(MethObj, di_graph, params, res_pre, m_summ)
    gr, lp, nc = [0] * n_r, [0] * n_r, [0] * n_r
    if "gr" in params["experiments"]:
        gr = expGR(di_graph, MethObj,
                   X, params["n_sample_nodes"].split(","),
                   n_r, res_pre,
                   m_summ, is_undirected=params["is_undirected"],
                   sampling_scheme=params["samp_scheme"])
    if "lpt" in params["experiments"]:
        expLPT(di_graph, MethObj, res_pre, m_summ,
               is_undirected=params["is_undirected"])
    if "lp" in params["experiments"]:
        lp = expLP(di_graph, MethObj,
                   params["n_sample_nodes"].split(","),
                   n_r, res_pre,
                   m_summ, train_ratio=params["train_ratio_lp"],
                   is_undirected=params["is_undirected"],
                   sampling_scheme=params["samp_scheme"])
    if "nc" in params["experiments"]:
        if "nc_test_ratio_arr" not in params:
            print('NC test ratio not provided')
        else:
            nc = expNC(X, node_labels, params["nc_test_ratio_arr"],
                       n_r, res_pre,
                       m_summ)
    if "viz" in params["experiments"]:
        if MethObj.get_method_name() == 'hope_gsvd':
            d = X.shape[1] / 2
            expVis(X[:, :d], res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)
        else:
            expVis(X, res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)
    return gr, lp, nc


def get_max(val, val_max, idx, idx_max):
    if val > val_max:
        return val, idx
    else:
        return val_max, idx_max


def choose_best_hyp(data_set, di_graph, node_labels, params):
    # Load range of hyper parameters to test on
    try:
        model_hyp_range = json.load(
            open('gemben/experiments/config/%s_hypRange.conf' % data_set, 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('gemben/experiments/config/default_hypRange.conf', 'r')
        )
    try:
        os.makedirs("gemben/temp_hyp_res")
    except:
        pass
    # Test each hyperparameter for each method and store the best
    for meth in params["methods"]:
        dim = int(params["dimensions"][0])
        MethClass = getattr(
            importlib.import_module("gemben.embedding.%s" % meth),
            methClassMap[meth]
        )
        meth_hyp_range = model_hyp_range[meth]
        gr_max, lp_max, nc_max = 0, 0, 0
        gr_hyp, lp_hyp, nc_hyp = 0, 0, 0
        gr_hyp, lp_hyp, nc_hyp = {meth: {}}, {meth: {}}, {meth: {}}

        # Test each hyperparameter
        ev_cols = ["GR MAP", "LP MAP", "NC F1 score"]
        hyp_df = pd.DataFrame(
            columns=list(meth_hyp_range.keys()) + ev_cols + ["Round Id"]
        )
        hyp_r_idx = 0
        for hyp in itertools.product(*meth_hyp_range.values()):
            hyp_d = {"d": dim}
            hyp_d.update(dict(zip(meth_hyp_range.keys(), hyp)))
            print(hyp_d)
            if meth == "sdne":
                hyp_d.update({
                    "modelfile": [
                        "gemben/intermediate/enc_mdl_%s_%d.json" % (data_set, dim),
                        "gemben/intermediate/dec_mdl_%s_%d.json" % (data_set, dim)
                    ],
                    "weightfile": [
                        "gemben/intermediate/enc_wts_%s_%d.hdf5" % (data_set, dim),
                        "gemben/intermediate/dec_wts_%s_%d.hdf5" % (data_set, dim)
                    ]
                })
            elif meth == "gf" or meth == "node2vec":
                hyp_d.update({"data_set": data_set})
            MethObj = MethClass(hyp_d)
            gr, lp, nc = run_exps(MethObj, meth, dim, di_graph,
                                  data_set, node_labels, params)
            gr_m, lp_m, nc_m = np.mean(gr), np.mean(lp), np.mean(nc)
            gr_max, gr_hyp[meth] = get_max(gr_m, gr_max, hyp_d, gr_hyp[meth])
            lp_max, lp_hyp[meth] = get_max(lp_m, lp_max, hyp_d, lp_hyp[meth])
            nc_max, nc_hyp[meth] = get_max(nc_m, nc_max, hyp_d, nc_hyp[meth])
            hyp_df_row = dict(zip(meth_hyp_range.keys(), hyp))
            f_hyp_temp = open("gemben/temp_hyp_res/%s_%s.txt" % (data_set, meth), "a")
            hyp_str = '_'.join("%s=%s" % (key, str(val).strip("'")) for (key, val) in hyp_d.items())
            f_hyp_temp.write('%s: MAP: %f\n' % (hyp_str, lp_max))
            f_hyp_temp.close()
            for r_id in range(params["rounds"]):
                hyp_df.loc[hyp_r_idx, meth_hyp_range.keys()] = \
                    pd.Series(hyp_df_row)
                hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                    [gr[min(r_id, len(gr) -1)], lp[r_id], nc[r_id], r_id]
                hyp_r_idx += 1
        exp_param = params["experiments"]
        for exp in exp_param:
            hyp_df.to_hdf(
                "gemben/intermediate/%s_%s_%s_%s_hyp.h5" % (data_set, meth,
                                                         exp,
                                                         params["samp_scheme"]),
                "df"
            )
        ###plot_util.plot_hyp(meth_hyp_range.keys(), exp_param,
           ##:                meth, data_set, s_sch=params["samp_scheme"])

        # Store the best hyperparameter
        ####### put the file into synthetic
        opt_hyp_f_pre = 'gemben/experiments/config/synthetic/%s_%s_%s' % (
            data_set,
            meth,
            params["samp_scheme"]
        )
        if gr_max:
            with open('%s_gr.conf' % opt_hyp_f_pre, 'w') as f:
                f.write(json.dumps(gr_hyp, indent=4))
        if lp_max:
            with open('%s_lp.conf' % opt_hyp_f_pre, 'w') as f:
                f.write(json.dumps(lp_hyp, indent=4))
        if nc_max:
            with open('%s_nc.conf' % opt_hyp_f_pre, 'w') as f:
                f.write(json.dumps(nc_hyp, indent=4))


def call_plot_hyp(data_set, params):
    # Load range of hyper parameters tested on to plot
    try:
        model_hyp_range = json.load(
            open('gemben/experiments/config/%s_hypRange.conf' % data_set, 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('gemben/experiments/config/default_hypRange.conf', 'r')
        )
    for meth in params["methods"]:
            meth_hyp_range = model_hyp_range[meth]
            exp_param = params["experiments"]
            plot_util.plot_hyp(meth_hyp_range.keys(), exp_param,
                               meth, data_set,
                               s_sch=params["samp_scheme"])


def call_plot_hyp_all(data_sets, params):
    # Load range of hyper parameters tested on to plot
    try:
        model_hyp_range = json.load(
            open('gemben/experiments/config/%s_hypRange.conf' % data_sets[0], 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('gemben/experiments/config/default_hypRange.conf', 'r')
        )
    for meth in params["methods"]:
            meth_hyp_range = model_hyp_range[meth]
            exp_param = params["experiments"]
            plot_util.plot_hyp_all(meth_hyp_range.keys(), exp_param,
                                   meth, data_sets,
                                   s_sch=params["samp_scheme"])


def call_exps(params, data_set):
    # Load Dataset
    print('Dataset: %s' % data_set)




########  for SBM, r_mat, hyperbolic
    #if data_set[10:13] == 'r_m' or data_set[10:13] == 'sto' or data_set[10:13] == 'hyp':
     #   di_graph = nx.read_gpickle('gem/data/%s/graph.gpickle' % data_set)[0]
    #else:
   
    #di_graph = nx.read_gpickle('gem/data/%s/graph.gpickle' % data_set)[0]
    di_graph = nx.read_gpickle('gemben/data/%s/graph.gpickle' % data_set)
    
    di_graph, nodeListMap = graph_util.get_lcc(di_graph)
    try:
      os.makedirs('gemben/nodeListMap')
    except:
      pass
    pickle.dump(nodeListMap, open('gemben/nodeListMap/%s.pickle' % data_set, 'wb'))
    graph_util.print_graph_stats(di_graph)


    # Load node labels if given
    if bool(params["node_labels"]):
        node_labels = cPickle.load(
            open('gemben/data/%s/node_labels.pickle' % data_set, 'rb')
        )
        node_labels_gc = np.zeros(
            (di_graph.number_of_nodes(), node_labels.shape[1]))
        for k, v in nodeListMap.iteritems():
            try:
                node_labels_gc[v, :] = node_labels[k, :].toarray()
            # Already a numpy array
            except AttributeError:
                node_labels_gc[v, :] = node_labels[k, :]
        node_labels = node_labels_gc
    else:
        node_labels = None

    # Search through the hyperparameter space
    if params["find_hyp"]:
        choose_best_hyp(data_set, di_graph, node_labels, params)

    # Load best hyperparameter and test it again on new test data
    for d, meth, exp in itertools.product(
        params["dimensions"],
        params["methods"],
        params["experiments"]
    ):
        dim = int(d)
        MethClass = getattr(
            importlib.import_module("gemben.embedding.%s" % meth),
            methClassMap[meth]
        )
        opt_hyp_f_pre = 'gemben/experiments/config/synthetic/%s_%s_%s' % (
            data_set,
            meth,
            params["samp_scheme"]
        )
        try:
            if exp != "viz":
                if exp == 'lpt':
                    model_hyp = json.load(
                        open('%s_lp.conf' % opt_hyp_f_pre, 'r')
                    )
                else:
                    model_hyp = json.load(
                        open('%s_%s.conf' % (opt_hyp_f_pre, exp), 'r')
                    )
            else:
                model_hyp = json.load(
                    open(
                        '%s_%s.conf' % (opt_hyp_f_pre, params["viz_params"]), 'r'
                    )
                )
        except IOError:
            print('Default hyperparameter of the method chosen')
            model_hyp = json.load(
                open('gemben/experiments/config/%s.conf' % meth, 'r')
            )
        hyp = {}
        hyp.update(model_hyp[meth])
        hyp.update({"d": dim})
        if meth == "sdne":
                hyp.update({
                    "modelfile": [
                        "gemben/intermediate/en_mdl_%s_%d.json" % (data_set, dim),
                        "gemben/intermediate/dec_mdl_%s_%d.json" % (data_set, dim)
                    ],
                    "weightfile": [
                        "gemben/intermediate/enc_wts_%s_%d.hdf5" % (data_set, dim),
                        "gemben/intermediate/dec_wts_%s_%d.hdf5" % (data_set, dim)
                    ]
                })
        elif meth == "gf" or meth == "node2vec":
            hyp.update({"data_set": data_set})
        MethObj = MethClass(hyp)
        run_exps(MethObj, meth, dim, di_graph, data_set, node_labels, params)


if __name__ == '__main__':
    ''' Sample usage
    python experiments/exp.py -data sbm -dim 128 -meth sdne -exp gr,lp
    '''
    t1 = time()
    parser = ArgumentParser(description='Graph Embedding Experiments')
    parser.add_argument('-data', '--data_sets',
                        help='dataset names (default: sbm)')
    parser.add_argument('-dim', '--dimensions',
                        help='embedding dimensions list(default: 2^1 to 2^8)')
    parser.add_argument('-meth', '--methods',
                        help='method list (default: all methods)')
    parser.add_argument('-exp', '--experiments',
                        help='exp list (default: gr,lp,viz,nc)')
    parser.add_argument('-lemb', '--load_emb',
                        help='load saved embeddings (default: False)')
    parser.add_argument('-lexp', '--load_exp',
                        help='load saved experiment results (default: False)')
    parser.add_argument('-node_labels', '--node_labels',
                        help='node labels available or not (default: False)')
    parser.add_argument('-rounds', '--rounds',
                        help='number of rounds (default: 5)')
    parser.add_argument('-plot', '--plot',
                        help='plot the results (default: True)')
    parser.add_argument('-plot_d', '--plot_d',
                        help='plot the results wrt dims(default: True)')
    parser.add_argument('-hyp_plot', '--hyp_plot',
                        help='plot the hyperparameter results (default: True)')
    parser.add_argument('-hyp_plot_all', '--hyp_plot_all',
                        help='plot the hyperparameter results (all) (default: True)')
    parser.add_argument('-train_ratio_lp', '--train_ratio_lp',
                        help='fraction of data used for training(default: 0.8)')
    parser.add_argument('-viz_params', '--viz_params',
                        help='which params to use for viz (default: gr)')
    parser.add_argument('-find_hyp', '--find_hyp',
                        help='find best hyperparameters (default: False)')
    parser.add_argument('-saveMAP', '--save_MAP',
                        help='save MAP in a latex table (default: False)')
    parser.add_argument('-n_samples', '--n_sample_nodes',
                        help='number of sampled nodes (default: 1024)')
    parser.add_argument('-s_sch', '--samp_scheme',
                        help='sampling scheme (default: u_rand)')

    params = json.load(open('gem/experiments/config/params.conf', 'r'))

    args = vars(parser.parse_args())

    for k, v in args.items():
        if v is not None:
            params[k] = v

    params["experiments"] = params["experiments"].split(',')
    params["data_sets"] = params["data_sets"].split(',')
    params["experiments"] = list(set(params["experiments"]))
    params["data_sets"] = list(set(params["data_sets"]))
    params["rounds"] = int(params["rounds"])
    params["node_labels"] = int(params["node_labels"])
    params["train_ratio_lp"] = float(params["train_ratio_lp"])
    # params["n_sample_nodes"] = int(params["n_sample_nodes"])
    params["is_undirected"] = bool(int(params["is_undirected"]))
   

    
    params["plot_d"] = bool(int(params["plot_d"]))
    params["plot"] = bool(int(params["plot"]))
    params["hyp_plot"] = bool(int(params["hyp_plot"]))
    params["hyp_plot_all"] = bool(int(params["hyp_plot_all"]))
    params["find_hyp"] = bool(int(params["find_hyp"]))
    
    if params["methods"] == "all":
        params["methods"] = methClassMap.keys()
    else:
        params["methods"] = params["methods"].split(',')
    params["methods"] = list(set(params["methods"]))       
        
    params["dimensions"] = params["dimensions"].split(',')
    params["dimensions"] = list(set(params["dimensions"]))
    if "nc_test_ratio_arr" in params:
        params["nc_test_ratio_arr"] = params["nc_test_ratio_arr"].split(',')
        params["nc_test_ratio_arr"] = \
            [float(ratio) for ratio in params["nc_test_ratio_arr"]]
    try:
      os.makedirs("gemben/intermediate")
    except:
      pass
    try:
      os.makedirs("gemben/results")
    except:
      pass

    for data_set in params["data_sets"]:
        if not int(params["load_exp"]):
            call_exps(params, data_set)
        if int(params["plot"]):
            res_pre = "gemben/results/%s" % data_set
            plot_util.plotExpRes(res_pre, params["methods"],
                                 params["experiments"], params["dimensions"],
                                 'gemben/plots/%s_%s' % (data_set, params["samp_scheme"]),
                                 params["rounds"], params["plot_d"],
                                 params["train_ratio_lp"],
                                 params["samp_scheme"])
        if int(params["hyp_plot"]):
            call_plot_hyp(data_set, params)
    if int(params["hyp_plot_all"]):
            call_plot_hyp_all(params["data_sets"], params)
