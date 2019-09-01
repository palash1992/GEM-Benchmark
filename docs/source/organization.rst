Organization
=============

The various components of the gemben library are as follows:

1) ``embedding`` - It consists of some of the baseline state-of-the-art graph embedding algorithms againsts which the benchmark can be evaluated.

2) ``evaluation`` - This module consists of the evaluation functions for graph reconstruction, link prediction, and node classification along with definition of the metrics used for evaluation. It also consists of function to visualize the embeddings. 

3) ``experiments`` - This is a self-contained module which can be used to test various state-of-the-art algorithms agains the benchmarks. 

4) ``plots`` -This modules handles the plotting of the compartive quantitative results.

5) ``utils`` - The mosudle consists of various utility function succ as bayesian hyper-parameter optimization, evaluation, graph generation, etc. 
