Background
===============

This section introduces the notation used in this benchmark library, and provides a brief overview of graph embedding methods. In-depth analysis of graph embedding theory we refer the reader to `here`_.

:math:`G (V, E)` denotes a weighted graph where :math:`V` is the set of vertices and :math:`E` is the set of edges. We represent :math:`W` as the adjacency matrix of :math:`G`, where :math:`W_{ij} = 1` represents the presence of an edge between :math:`i` and :math:`j`. A graph embedding is a mapping :math:`f: V -> \mathbb{R}^d`, where :math:`d  << |V|` and the function :math:`f` preserves some proximity measure defined on graph :math:`G`. It aims to map similar nodes close to each other. Function :math:`f` when applied on the graph :math:`G` yields an embedding :math:`Y`.

In this benchmark library, we evaluate four state-of-the-art graph embedding methods on a set of real graphs denoted by :math:`\mathcal{R}` and synthetic graphs denoted by :math:`\mathcal{S}`. To analyze the performance of methods, we categorize the graphs into a set of domains :math:`\mathcal{D} = \lbrace` Social, Economic, Biology, Technological:math:`\rbrace`. The set of graphs in a domain :math:`D \in \mathcal{D}` is represented as :math:`\mathcal{R}_D`. We use multiple evaluation metrics on graph embedding methods to draw insights into each approach. We denote this set of metrics as :math:`\mathcal{M}`. The notations are summarized as follows:.

* :math:`G` : Graphical representation of the data 
* :math:`E` : Set of edges in the graph 
* :math:`W` : Adjacency matrix of the graph, :math:`|V| \times |V|` 
* :math:`f` : Embedding function 
* :math:`\mathcal{S}` : Set of synthetic graphs  
* :math:`\mathcal{R}_D` : Set of real graphs in domain :math:`D` 
* :math:`\mathcal{D}` : Set of domains  
* :math:`\mathcal{M}` : Set of evaluation metrics  
* :math:`e_m` : Evaluation function for metric :math:`m`
* :math:`\mathcal{A}` : Set of graph and embedding attributes 
* :math:`d` : Number of embedding dimensions
* :math:`Y` : Embedding of the graph, :math:`|V| \times d` 

Graph Embedding Methods
------------------------

Graph embedding methods embed graph vertices into a low-dimensional space. The goal of graph embedding is to preserve certain properties of the original graph such as distance between nodes and neighborhood structure. Based upon the function :math:`f` used for embedding the graph, existing methods can be classified into three categories: **factorization based**, **random walk based** and **deep learning based**. 

Factorization based approaches
-------------------------------

Factorization based approaches apply factorization on graph related matrices to obtain the node representation. Graph matrices such as the adjacency matrix, Laplacian matrix, and Katz similarity matrix contain information about node connectivity and the graph's structure. Other matrix factorization approaches use the eigenvectors from spectral decomposition of a graph matrix as node embeddings. For example, to preserve locality, LLE_ uses :math:`d` eigenvectors corresponding to eigenvalues from second smallest to :math:`(d+1)^{th}` smallest from the sparse matrix :math:`(I-W)^\intercal(I-W)`. It assumes that the embedding of each node is a linear weighted combination of the neighbor's embeddings. `Laplacian Eigenmaps`_ take the first :math:`d` eigenvectors with the smallest eigenvalues of the normalized Laplacian :math:`D^{-1/2}LD^{-1/2}`. Both LLE and Laplacian Eigenmaps were designed to preserve the local geometric relationships of the data. Another type of matrix factorization methods learn node embeddings under different optimization functions in order to preserve certain properties. `Structural Preserving Embedding`_ builds upon Laplacian Eigenmaps to recover the original graph. `Cauchy Graph Embedding`_ uses a quadratic distance formula in the objective function to emphasize similar nodes instead of dissimilar nodes. `Graph Factorization`_  uses an approximation function to factorize the adjacency matrix in a more scalable manner. `GraRep`_ and `HOPE`_ were invented to keep the high order proximity in the graph. Factorization based approaches have been widely used in practical applications due to their scalability. The methods are also easy to implement and can yield quick insights into the data set.

Random walk approaches
-------------------------

Random walk based algorithms are more flexible than factorization methods to explore the local neighborhood of a node for high-order proximity preservation. `DeepWalk`_ and `Node2vec`_  aim to learn a low-dimensional feature representation for nodes through a stream of random walks. These random walks explore the nodes' variant neighborhoods. Thus, random walk based methods are much more scalable for large graphs and they generate informative embeddings. Although very similar in nature, DeepWalk simulates uniform random walks and Node2vec employs search-biased random walks, which enables embedding to capture the community or structural equivalence via different bias settings. `LINE`_ combines two phases for embedding feature learning: one phase uses a breadth-first search (BFS) traversal across first-order neighbors, and the second phase focuses on sampling nodes from second-order neighbors. `HARP`_ improves DeepWalk and Node2vec by creating a hierarchy for nodes and using the embedding of the coarsened graph as a better initialization in the original graph. `Walklets`_  extended Deepwalk by using multiple skip lengths in random walking. Random walk based approaches tend to be more computationally expensive than factorization based approaches but can capture complex properties and longer dependencies between nodes.

Neural network approaches
---------------------------

The third category of graph embedding approaches is based on neural networks. Deep neural networks based approaches capture highly non-linear network structure in graphs, which is neglected by factorization based and random walk based methods. One type of deep learning based methods such as `SDNE`_ uses a deep autoencoder to provide non-linear functions to preserve the first and second order proximities jointly. Similarly, `DNGR`_  applies random surfing on input graph before a stacked denoising autoencoder and makes the embedding robust to noise in graphs. Another genre of methods use Graph Neural Networks(*GNNs*) and  Graph Convolutional Networks (*GCNs*) (`bruna2013spectral`_, `henaff2015deep`_, `li2015gated`_, `hamilton2017inductive`_) to aggregate the neighbors embeddings and features via convolutional operators, including spatial or spectral filters. GCNs learn embeddings in a semi-supervised manner and have shown great improvement and scalability on large graphs compared to other methods. `SEAL`_ learns a wide range of link prediction heuristics from extracted local enclosing subgraphs with GNN. `DIFFPOOL`_  employs a differentiable graph pooling module on GNNs to learn hierarchical embeddings of graphs. Variational `Graph Auto Encoders`_ (*VGAE*) utilizes a GCN as encoder and inner product as decoder, which provides embedding with higher quality than autoencoders. Deep neural network based algorithms like *SDNE* and *DNGR* can be computational costly since they require the global information such as adjacency matrix for each node as input. GCNs based methods are more scalable and flexible to characterize global and local neighbours through variant convolutional and pooling layers.


.. _LLE:
	https://science.sciencemag.org/content/290/5500/2323

.. _Structural Preserving Embedding:
	http://www.cs.columbia.edu/~jebara/papers/spe-icml09.pdf

.. _Cauchy Graph Embedding:
	http://www.icml-2011.org/papers/353_icmlpaper.pdf

.. _GraRep:
	https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf

.. _HOPE:
	https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf

.. _DeepWalk:
	https://arxiv.org/pdf/1403.6652.pdf

.. _Node2vec:
	https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf

.. _LINE:
	https://arxiv.org/pdf/1503.03578.pdf

.. _HARP:
	https://arxiv.org/pdf/1706.07845.pdf

.. _Walklets:
	https://pdfs.semanticscholar.org/37cf/46e45777e67676f80c9110bed675a9840590.pdf

.. _DNGR:
	https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf

.. _bruna2013spectral:
	https://arxiv.org/pdf/1312.6203.pdf

.. _henaff2015deep:
	https://arxiv.org/pdf/1506.05163.pdf

.. _li2015gated:
	https://arxiv.org/pdf/1511.05493.pdf

.. _hamilton2017inductive:
	https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf

.. _SEAL:
	https://papers.nips.cc/paper/7763-link-prediction-based-on-graph-neural-networks.pdf

.. _DIFFPOOL:
	https://arxiv.org/pdf/1811.01287v1.pdf

.. _Graph Auto Encoders:
	https://arxiv.org/pdf/1611.07308.pdf

.. _here:
	https://arxiv.org/abs/1705.02801

.. _community detection:
    http://homes.sice.indiana.edu/filiradi/Mypapers/pre78_046110_2008.pdf

.. _Preferential Attachment:
	https://science.sciencemag.org/content/286/5439/509

.. _Common Neighbors:
	https://arxiv.org/pdf/cond-mat/0104209.pdf

.. _Adamic Adar:
	https://reader.elsevier.com/reader/sd/pii/S0378873303000091?token=6F43C18383A6F25A71900BE3D0FC6C10251CCB28A020DD02EB00C3758F0DBDB4E69D3C3A41DE87D28C79A03F0EED5157

.. _Jaccards Coefficient:
	https://dl.acm.org/citation.cfm?id=576628

.. _Laplacian Eigenmaps:
	http://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NIPS_01.pdf

.. _Graph Factorization:
	https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf

.. _Higher Order Proximity Preserving:
	https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf

.. _SDNE:
	https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf
