Challenges
===============

Most research on graph embedding has focused on the development of mechanisms to preserve various characteristics of the graph in the low-dimensional space. However, very little attention has been dedicated to the development of mechanisms to rigorously compare and evaluate different graph embedding methods. To make matters worse, most of the existing work use simple synthetic data sets for visualization and a few real networks for quantitative comparison. Goyal_ et. al. use Stochastic Block Models to visualize the results of graph embedding methods. Salehi_ at. al. use the Barabasi-Albert graph to understand the properties of embeddings. Such evaluation strategy suffers from the following challenges:

* Properties of real networks vary according to the domain. Therefore it is often difficult to ascertain the reason behind the performance improvement of a given method on a particular real dataset (as shown in Goyal_ et. al.

* As demonstrated in this benchmark library, the performance of embedding approaches vary greatly, and according to the properties of different graphs. Therefore, the utility of any specific method is difficult to establish and to characterize. In practice, the performance improvement of a method can be attributed to stochasticity.

* Different methods use different metrics for evaluation. This makes it very difficult to compare the performance of different graph embedding methods on a given problem.

* Typically, each graph embedding method has a reference implementation. This implementation makes specific assumptions about the data, representation, etc. This further complicates the comparison between methods.
 
.. _Goyal:
	https://arxiv.org/abs/1705.02801

.. _Salehi:
	https://www.google.com/search?q=Properties+of+Vector+Embeddings+in+Social+Networks&oq=Properties+of+Vector+Embeddings+in+Social+Networks&aqs=chrome..69i57.2611j0j4&sourceid=chrome&ie=UTF-8

