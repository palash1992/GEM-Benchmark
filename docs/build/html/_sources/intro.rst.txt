Introduction
===============

Graphs are a natural way to represent relationships and interactions between entities in real systems. For example, people on social networks, proteins in biological networks, and authors in publication networks can be represented by nodes in a graph, and their relationships such as friendships, protein-protein interactions, and co-authorship are represented by edges in a graph. These graphical models enable us to understand the behavior of  systems and to gain insight into their structure. These insights can further be used to predict future interactions and missing information in the system. These tasks are formally defined as link prediction and node classification. Link prediction estimates the likelihood of a relationship among two entities. This is used, for example, to recommend friends on social networks and to sort probable protein-protein interactions on biological networks. Similarly, node classification estimates the likelihood of a node's label. This is used, for example, to infer missing meta-data on social media profiles, and genes in proteins.

Numerous graph analysis methods have been developed. Broadly, these methods can be categorized as non-parametric and parametric. Non-parametric methods operate directly on the graph whereas parametric methods represent the properties of nodes and edges in the graph in a low-dimensional space. Non-parametric methods such as `Common Neighbors`_, `Adamic Adar`_  and `Jaccard coefficient`_ require access and knowledge of the entire graph for the prediction. On the other hand, parametric models such as Thor_ et. al. employ graph summarization and define super nodes and super edges to perform link prediction. Kim_ et. al. use Expectation Maximization to fit the real network as a Kronecker graph and estimate the parameters. Another class of parametric models that have gained much attention recently are `graph embeddings`_. Graph embedding methods define a low-dimensional vector for each node and a distance metric on the vectors. These methods learn the representation by preserving certain properties of the graph. `Graph Factorization`_ preserves visible links, HOPE_ aims to preserve higher order proximity, and node2vec_ preserves both structural equivalence and higher order proximity. In this benchmark library, we focus our attention on graph embedding methods. While this is a very active area of research that continues to gain popularity among researchers, there are several challenges that must be addressed before graph embedding algorithms become mainstream.

	.. _Common Neighbors:
	    https://arxiv.org/pdf/cond-mat/0104209.pdf

	.. _Adamic Adar:
		https://reader.elsevier.com/reader/sd/pii/S0378873303000091?token=6F43C18383A6F25A71900BE3D0FC6C10251CCB28A020DD02EB00C3758F0DBDB4E69D3C3A41DE87D28C79A03F0EED5157

	.. _Jaccard coefficient:
		https://dl.acm.org/citation.cfm?id=576628

	.. _Thor:
		https://people.cs.umass.edu/~barna/paper/iswc2011.pdf

	.. _Kim:
		https://cs.stanford.edu/people/jure/pubs/kronEM-sdm11.pdf

	.. _graph embeddings:
		https://arxiv.org/abs/1705.02801

	.. _Graph Factorization:
		https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf

	.. _HOPE:
		https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf

	.. _node2vec:
		https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf	


