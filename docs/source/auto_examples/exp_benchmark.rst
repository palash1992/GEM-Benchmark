.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_exp_benchmark.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_exp_benchmark.py:


==========================
Experiment with Benchmark
==========================
Example to run the benchmark across all the baseline embedding algorithms.


.. code-block:: default


    from subprocess import call
    import itertools
    try: import cPickle as pickle
    except: import pickle
    import json
    from argparse import ArgumentParser
    import networkx as nx
    import pandas as pd
    import pdb
    import os
    import sys
    from time import time
    # sys.path.insert(0, './')
    from gemben.utils import graph_gens

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

    if __name__ == "__main__":
        ''' Sample usage
        python experiments/exp_synthetic.py -syn_names all -plot_hyp_data 1 -meths all
        '''
        t1 = time()
        parser = ArgumentParser(description='Graph Embedding Benchmark Experiments')
        parser.add_argument('-data', '--data_sets',
                            help='dataset names (default: barabasi_albert_graph)')
        parser.add_argument('-dims', '--dimensions',
                            help='embedding dimensions list(default: 128)')
        parser.add_argument('-meth', '--methods',
                            help='method list (default: all methods)')
        parser.add_argument('-plot_hyp_data', '--plot_hyp_data',
                            help='plot the hyperparameter results (default: False)')
        parser.add_argument('-rounds', '--rounds',
                            help='number of rounds (default: 20)')
        parser.add_argument('-s_sch', '--samp_scheme',
                            help='sampling scheme (default: rw)')
        parser.add_argument('-lexp', '--lexp',
                            help='load experiment (default: False)')
        params = json.load(
            open('gemben/experiments/config/params_benchmark.conf', 'r')
        )
        args = vars(parser.parse_args())
        print (args)
        syn_hyps = json.load(
            open('gemben/experiments/config/syn_hypRange.conf', 'r')
        )
        for k, v in args.items():
            if v is not None:
                params[k] = v
        params["rounds"] = int(params["rounds"])
        if params["data_sets"] == "all":
            params["data_sets"] = syn_hyps.keys()
        else:
            params["data_sets"] = params["data_sets"].split(',')
        params["lexp"] = bool(int(params["lexp"]))
        params["plot_hyp_data"] = bool(int(params["plot_hyp_data"]))
        if params["methods"] == "all":
            params["methods"] = methClassMap.keys()
        else:
            params["methods"] = params["methods"].split(',')
        params["dimensions"] = params["dimensions"].split(',')
        samp_scheme = params["samp_scheme"]
        for syn_data in params["data_sets"]:
            syn_hyp_range = syn_hyps[syn_data]
            hyp_keys = list(syn_hyp_range.keys())
            if syn_data == "binary_community_graph":
                graphClass = getattr(graph_gens, syn_data)
            else:
                graphClass = getattr(nx, syn_data)
            ev_cols = ["GR MAP", "LP MAP", "LP P@100", "NC F1 score"]
            for dim in params["dimensions"]:
                dim = int(dim)
                for meth in params["methods"]:
                    if not params["lexp"]:
                        hyp_df = pd.DataFrame(
                            columns=hyp_keys + ev_cols + ["Round Id"]
                        )
                        hyp_r_idx = 0
                        for hyp in itertools.product(*syn_hyp_range.values()):
                            hyp_dict = dict(zip(hyp_keys, hyp))
                            hyp_str = '_'.join(
                                "%s=%r" % (key, val) for (key, val) in hyp_dict.items()
                            )
                            syn_data_folder = 'benchmark_%s_%s' % (syn_data, hyp_str)
                            hyp_df_row = dict(zip(hyp_keys, hyp))
                            for r_id in range(params["rounds"]):
                                G = graphClass(**hyp_dict)
                                if not os.path.exists("gemben/data/%s" % syn_data_folder):
                                    os.makedirs("gemben/data/%s" % syn_data_folder)
                                nx.write_gpickle(
                                    G, 'gemben/data/%s/graph.gpickle' % syn_data_folder
                                )
                                os.system(
                                    "python gem/experiments/exp.py -data %s -meth %s -dim %d -rounds 1 -s_sch %s -exp lp" % (syn_data_folder, meth, dim, samp_scheme)
                                )
                                MAP, prec, n_samps = pickle.load(
                                    open('gemben/results/%s_%s_%d_%s.lp' % (syn_data_folder, meth, dim, samp_scheme), 'rb')
                                )        
                                hyp_df.loc[hyp_r_idx, hyp_keys] = \
                                    pd.Series(hyp_df_row)
                                prec_100 = prec[int(n_samps[0])][0][100]
                                hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                                    [0, MAP[int(n_samps[0])][0], prec_100, 0, r_id]
                                hyp_r_idx += 1
                        hyp_df.to_hdf(
                            "gemben/intermediate/%s_%s_lp_%s_dim_%d_data_hyp.h5" % (syn_data, meth, samp_scheme, dim),
                            "df"
                        )
                if params["plot_hyp_data"]:
                    from gem.utils import plot_util
                    plot_util.plot_hyp_data2(
                        hyp_keys, ["lp"], params["methods"], syn_data, samp_scheme, dim
                    )
        print('Total time taken: %f sec' % (time() - t1))



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_auto_examples_exp_benchmark.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: exp_benchmark.py <exp_benchmark.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: exp_benchmark.ipynb <exp_benchmark.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
