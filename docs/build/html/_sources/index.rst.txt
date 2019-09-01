.. gemben documentation master file, created by
   sphinx-quickstart on Sun Jun  9 22:12:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################################################
Welcome to GemBen: Graph Embedding Methods Benchmark documentation!
####################################################################

GEM-Benchmark is a principled framework to compare graph embedding methods. We test embedding methods on a corpus of real-world networks with varying properties and provide insights into existing state-of-the-art embedding approaches. We cluster the input networks in terms of their properties to get a better understanding of embedding performance. Furthermore, we compare embedding methods with traditional link prediction techniques to evaluate the utility of embedding approaches. We use the comparisons on benchmark graphs to define a score, called GFS-score, that can apply to measure any embedding method. We rank the state-of-the-art embedding approaches using the GFS-score and show that it can be used to understand and evaluate a novel embedding approach. We envision that the proposed framework may serve as a community benchmark to test and compare the performance of future graph embedding techniques.

See the source code repository on github: https://github.com/palash1992/GEM-benchmark

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   dependency
   install
   testing

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Documentation

   intro
   challenge
   contibution
   organization
   background
   gem_ben

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Dataset

   dataset

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Addtional Information

   authors
   citations
   license