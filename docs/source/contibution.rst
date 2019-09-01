Why this Benchmark Package?
==============================

In this benchmark libarry, we aim to: 

* provide a unifying framework for comparing the performance of state-of-the-art and future graph embedding methods; 

* establish a benchmark comprised of 100 real-world graphs that exhibit different structural properties; and 

* provide users with a fully automated library that selects the best graph embedding method for their graph data. 

We address the above challenges with the following contributions:

* We propose an evaluation benchmark to compare and evaluate embedding methods. This benchmark consists of 100 real-world graphs categorized into four domains: social, biology, technological and economic.

* Using our evaluation benchmark, we evaluate and compare 8 state-of-the-art methods and provide, for the first time, a characterization of their performance against graphs with different properties. We also compare their scores with traditional link prediction methods and ascertain the general utility of embedding methods.

* A new score, GFS-score, is introduced to compare various graph embedding methods for link prediction. The GFS-score provides a robust metric to evaluate a graph embedding approach by averaging over 100 graphs. It further has many components based on the type and property of graph yielding insights into the methods.

* A Python library comprised of 4 state-of-the-art embedding methods, and 4 traditional link prediction methods. This library automates the evaluation, comparison against all the other methods, and performance plotting of any new graph embedding method.

