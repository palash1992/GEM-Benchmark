#!/usr/local/bin/python
# coding: utf-8
import os
from time import time
from subprocess import call
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import scipy
import networkit as nk
from scipy import special
from numpy import pi
import itertools
import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from gem.utils import graph_util,kronecker_generator,kronecker_init_matrix
import math
import multiprocessing

########################################################################
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

########################################################################
def plot_hist(title, data):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.hist(x=data)
    plt.savefig(title + '.png')
########################################################################

def barbell_graph(m1,m2):
    graph = nx.barbell_graph(m1,m2)
    ## for com_nc, one hot 
    #onehot_com = np.array([[1,0,0]]*m1+[[0,1,0]]*m2+[[0,0,1]]*m1)  is slower when num of nodes > 2000
    node_labels_com = np.zeros(m1*2+m2).astype(int)
    node_labels_com[m1:m1+m2] = 2
    node_labels_com[m1+m2:] = 1
    ## one hot
    onehot_com = np.zeros((m1*2+m2,3)).astype(int)
    onehot_com[np.arange(m1*2+m2), node_labels_com] = 1
    
    ## for role_nc, one hot
    node_labels_role = np.zeros(m1*2+m2).astype(int)
    p,q = divmod(m2, 2) 
    for i in range(p+1):
        node_labels_role[[m1-1+i,m1+m2-i]] = i+1
    if q:
        node_labels_role[m1+p] = p+2
    onehot_role = np.zeros((m1*2+m2,p+q+2)).astype(int)
    onehot_role[np.arange(m1*2+m2), node_labels_role] = 1

    return graph, scipy.sparse.csr_matrix(onehot_com), scipy.sparse.csr_matrix(onehot_role)

##########################################################################

def binary_community_graph(N, k, maxk, mu):
    ## OS system is windows 
    if sys.platform[0] == "w":
        args = ["gem/c_exe/benchm.exe"]
        fcall = "gem/c_exe/benchm.exe"
    else:
        args = ["gem/c_exe/benchm"]
        fcall = "gem/c_exe/benchm"
    args.append("-N %d" % N)
    args.append("-k %d" % k)
    args.append("-maxk %d" % maxk)
    args.append("-mu %f" % mu)
    t1 = time()
    print(args)
    try:
        os.system("%s -N %d -k %d -maxk %d -mu %f" % (fcall, N, k, maxk, mu))
        # call(args)
    except Exception as e:
        print('ERROR: %s' % str(e))
        print('gem/c_exe/benchm not found. Please compile gf, place benchm in the path and grant executable permission')
    t2 = time()
    print('\tTime taken to generate random graph: %f sec' % (t2 - t1))
    try:
        graph = graph_util.loadGraphFromEdgeListTxt('gem/c_exe/network.dat')
        node_labels = np.loadtxt('gem/c_exe/community.dat')
    except:
        graph = graph_util.loadGraphFromEdgeListTxt('network.dat')
        node_labels = np.loadtxt('community.dat')
    node_labels = node_labels[:, -1].reshape(-1, 1)
    enc = OneHotEncoder()
    return graph, enc.fit_transform(node_labels)


########################################################################
def barabasi_albert_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n: Number of Nodes
    m: Number of edges to attach from a new node to existing nodes
    Formula for m:  (m^2)- (Nm)/2 + avg_deg * (N/2) = 0  =>  From this equation we need to find m :
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'

    ## As does not have for Diameter Variance
    if dia > 0:
        return None

    strt_time = time()

    m = int(round((N - np.sqrt(N**2 - 4*deg*N))/4))
     
       
    G = nx.barabasi_albert_graph(n=N, m=m)

    lcc, _ = graph_util.get_lcc_undirected(G)
             




    best_G = lcc

    best_diam = nx.algorithms.diameter(best_G)

    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))



    end_time = time()
     
    print('Graph_Name: Barabasi Albert Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ' , end_time - strt_time, ' secs')

    return best_G, best_avg_deg, best_diam

########################################################################################################################

def random_geometric_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int or iterable) – Number of nodes or iterable of nodes

    radius (float) – Distance threshold value

    Average Degree is given by formula: Avg_Deg = (pi*(r^2)*num_nodes)/(l^2)
    Formula for r: avg_deg * l
    where l can be considered a constant where its square can be approximated to 1.04 [ength of square] Empirically Found
    :return: Graph Object
    '''
    strt_time = time()

    l = 1.04

    count = 0
    tolerance = 0.5
    curr_deg_error = float('inf')

    while tolerance < curr_deg_error:

        r = np.round(np.sqrt((deg * l ) / (3.14 * N)), 3)

        G = nx.random_geometric_graph(n=N, radius=r)

        curr_avg_deg = np.mean(list(dict(nx.degree(G)).values()))

        lcc = graph_util.get_lcc_undirected(G)[0]

        curr_deg_error = abs(curr_avg_deg - deg)



        count += 1

        if count == 1000:

            break

    best_G = lcc
    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = curr_avg_deg

    end_time = time()

    print('Graph_Name: Random_Geometric_Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################################################################

def waxman_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int or iterable) – Number of nodes or iterable of nodes

    beta (float) – Model parameter

    alpha (float) – Model parameter


    Average Degree is given by formula: k
    where P = beta * exp(-d/alpha*L)
    alpha = (gamma((k/2)+1) * (beta^k))/((n-1)*(pi^(k/2))*gamma(k))
    where beta is chosen randomly to satisfy the average degree criterion
    So we fix the parameter beta = 0.1, and we know the default value of d/L is in range: 0.25 to 0.3 (Empiricially calculated)
    so we only tweak alpha to get the required avg deg.

    :return: Graph Object
    '''
    strt_time = time()

    bands = 10
    lower_lim = 2.5
    upper_lim = 3.5
    tolerance = 0.5


    k = 2

    curr_avg_deg_error = float('inf')
    flag = False


    while curr_avg_deg_error >= tolerance:

        s_space = np.linspace(lower_lim, upper_lim, bands)

        avg_deg_error_list = []

        s_gap = s_space[1] - s_space[0]

        for s in s_space:

            g_s = (k * (pi ** (k / 2)) * special.gamma(k)) / (special.gamma((k / 2) + 1) * (s ** k))

            q = deg/((N-1)*g_s)

            G = nx.waxman_graph(n=N, alpha=s, beta=q)

            lcc = graph_util.get_lcc_undirected(G)[0]

            curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))

            avg_deg_err = abs(curr_avg_deg - deg)

            if avg_deg_err <= tolerance:

                best_G = G
                best_avg_deg = curr_avg_deg
                best_diam = nx.algorithms.diameter(lcc)
                flag = True
                break

            avg_deg_error_list.append((lcc,avg_deg_err , curr_avg_deg, s))

        if flag == True:
            break

        sorted_avg_err = sorted(avg_deg_error_list, key=lambda x: x[1])

        curr_avg_deg_error = sorted_avg_err[0][1]

        if sorted_avg_err[0][1] <= tolerance:

            best_G = sorted_avg_err[0][0]

            best_avg_deg = sorted_avg_err[0][2]

            best_diam = nx.algorithms.diameter(best_G)

            break

        else:

            lower_lim = sorted_avg_err[0][3] - s_gap
            upper_lim = sorted_avg_err[0][3] + s_gap


    end_time = time()

    print('Graph_Name: waxman_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################
def watts_strogatz_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int) – The number of nodes
    k (int) – Each node is joined with its k nearest neighbors in a ring topology.
    p (float) – The probability of rewiring each edge

    Average Degree is solely decided by k
    Diameter depends on the value of p
    :return: Graph Object
    '''
    strt_time = time()

    p = 0.2

    G = nx.watts_strogatz_graph(n=N, k=deg, p=p)

    lcc, _ = graph_util.get_nk_lcc_undirected(G)

    best_G = lcc

    best_diam = nx.algorithms.diameter(best_G)

    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))

    end_time = time()

    print('Graph_Name: Watts_Strogatz_Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)

    return best_G, best_avg_deg, best_diam

########################################################################
def duplication_divergence_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int) – The desired number of nodes in the graph.
    p (float) – The probability for retaining the edge of the replicated node.
    :return: Graph Object
    '''
    strt_time = time()

    tolerance = 0.5

    if deg == 4:
        lower_lim = 0.3
        upper_lim = 0.4
        bands = 10

    elif deg == 6:
        lower_lim = 0.4
        upper_lim = 0.6
        bands = 15

    elif deg == 8:
        lower_lim = 0.46
        upper_lim = 0.60
        bands = 15
    elif deg == 10:
        lower_lim = 0.50
        upper_lim = 0.65
        bands = 15
    elif deg == 12:
        lower_lim = 0.55
        upper_lim = 0.68
        bands = 15


    flag = False
    curr_avg_deg_error = float('inf')

    while curr_avg_deg_error > tolerance:
        p_space = np.linspace(lower_lim, upper_lim, bands)
        avg_deg_err_list = []


        p_gap = p_space[1] - p_space[0]
        for p_val in p_space:

            G = nx.duplication_divergence_graph(n=N, p=p_val)

            lcc, _ = graph_util.get_nk_lcc_undirected(G)

            curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))

            curr_avg_deg_error = abs(deg - curr_avg_deg)

            avg_deg_err_list.append((lcc, curr_avg_deg_error, p_val, curr_avg_deg))

            if deg - curr_avg_deg < 0:
                break
            if curr_avg_deg_error <= tolerance:

                best_G = lcc

                best_avg_deg = curr_avg_deg

                best_diam = nx.algorithms.diameter(best_G)

                flag = True

                break


        if flag == True:
            break

        sorted_avg_err = sorted(avg_deg_err_list, key=lambda x: x[1])

        curr_avg_deg_error = sorted_avg_err[0][1]

        if sorted_avg_err[0][1] <= tolerance:

            best_G = sorted_avg_err[0][0]
            best_avg_deg = sorted_avg_err[0][3]
            best_diam = nx.algorithms.diameter(best_G)
            break
        else:

            lower_lim = sorted_avg_err[0][2] - p_gap
            upper_lim = sorted_avg_err[0][2] + p_gap

    end_time = time()
    print('Graph_Name: duplication divergence graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################
def powerlaw_cluster_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int) – the number of nodes
    m (int) – the number of random edges to add for each new node
    p (float,) – Probability of adding a triangle after adding a random edge
    Formula for m:  (m^2)- (Nm)/2 + avg_deg * (N/2) = 0  =>  From this equation we need to find m :
    p : Does not vary the average degree or diameter so much. : Higher value of p may cause average degree to overshoot intended average_deg
    so we give the control of average degree to parameter m: by setting a lower value of p: 0.1
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'
    strt_time = time()

    m = int(round((N - np.sqrt(N ** 2 - 4 * deg * N)) / 4))
    p = 0.2

    ## G at center:
    G = nx.powerlaw_cluster_graph(n=N, m=m, p=p)

    lcc, _ = graph_util.get_nk_lcc_undirected(G)

    best_G = lcc

    best_diam = nx.algorithms.diameter(best_G)

    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))


    end_time = time()
    print('Graph_Name: powerlaw_cluster_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

#####################################################################
def stochastic_block_model(N, deg, dia, dim, domain):
    '''

    :param N: Number of Nodes
    :param p:   Element (r,s) gives the density of edges going from the nodes of group r
                to nodes of group s. p must match the number of groups (len(sizes) == len(p)),
                and it must be symmetric if the graph is undirected.

    Formula for p: Through Empirical Studies - p = 0.001 * Deg gives perfect result for Num_of_Nodes = 1024
                    But if N  >1024: scaler = N/1024 : then p = (0.001*deg)/scaler
                    And if N < 1024 : Scaler = 1024/N : then p = (0.001*deg)*scaler
                    and if N == 1024: p = (0.001*deg)
    For each
    :return:
    '''
    tolerance = 0.5

    curr_deg_error = float('inf')

    count = 0

    p_default = 0.001 * deg
    N_default = 1024

    if N_default > N:
        p_scaler = N_default/N
        p = p_default * p_scaler

    elif N_default < N:
        p_scaler = N / N_default
        p = p_default / p_scaler
    else:
        p = p_default

    strt_time = time()

    while curr_deg_error > tolerance:

        G = nx.generators.stochastic_block_model([N],[[p]])

        lcc,_ = graph_util.get_nk_lcc_undirected(G)

        curr_avg_deg = np.mean(list(dict(nx.degree(G)).values()))

        curr_deg_error = abs(curr_avg_deg - deg)

        count += 1

        if count == 1000:

            break

    best_G = lcc
    best_avg_deg = curr_avg_deg
    best_diam = nx.algorithms.diameter(lcc)

    end_time = time()

    print('Graph_Name: Stochastic Block Model')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)

    return best_G, best_avg_deg, best_diam

#####################################################################
def r_mat_graph(N, deg, dia, dim, domain):

    tolerance = 0.5
    curr_deg_error = float('inf')
    count = 0
    strt_time = time()

    scale = np.log2(N)

    while curr_deg_error > tolerance:

        G_Nk = nk.generators.RmatGenerator(scale=scale,edgeFactor=deg/2, a=0.25,b=0.25,c=0.25,d=0.25).generate()

        G = graph_util.convertNkToNx(G_Nk)

        lcc,_ = graph_util.get_nk_lcc_undirected(G)

        curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))

        curr_deg_error = abs(curr_avg_deg - deg)

        count += 1

        if count == 1000:

            break


    if count == 1000:
        raise("MAX TRIES EXCEEDED, TRY AGAIN")

    best_G = lcc
    best_avg_deg = curr_avg_deg
    best_diam = nx.algorithms.diameter(lcc)

    end_time = time()

    print('Graph_Name: RMAT')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)

    return best_G, best_avg_deg, best_diam

#####################################################################
def hyperbolic_graph(N, deg, dia, dim, domain):
    '''
        Parameters of the graph:
        N = Num of nodes
        k = Average degree
        gamma = Target exponent in Power Law Distribution
        :return: Graph Object
        '''
    import networkit as nk

    tolerance = 0.5
    curr_deg_error = float('inf')
    count = 0
    strt_time = time()

    while curr_deg_error > tolerance:

        G_Nk = nk.generators.HyperbolicGenerator(n = N,k = deg,gamma = 3.5).generate()

        G = graph_util.convertNkToNx(G_Nk)

        lcc,_ = graph_util.get_nk_lcc_undirected(G)

        curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))

        curr_deg_error = abs(curr_avg_deg - deg)

        count += 1

        if count == 1000:

            break

    best_G = lcc
    best_avg_deg = curr_avg_deg
    best_diam = nx.algorithms.diameter(lcc)

    end_time = time()

    print('Graph_Name: Hyperbolic Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)

    return best_G, best_avg_deg, best_diam

########################################################################
def stochastic_kronecker_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    degree_seq
    :return: Graph Object
    '''
    strt_time = time()

    nodes = 2

    init = kronecker_init_matrix.InitMatrix(nodes)
    init.make()

    # Alpha Beta Method of Testing
    init.addEdge(0, 1)
    init.addSelfEdges()

    tolerance = 0.5

    ## Write Custom Params
    avg_deg_error = float('inf')

    max_tries = 1000

    count =0

    if domain == "social":
        alphas, betas, gammas = [0.999], np.linspace(0.45, 0.8, 10), np.linspace(0.2, 0.4, 10)
    elif domain == "biology":
        alphas, betas, gammas = [0.85], np.linspace(0.6, 0.95, 10), np.linspace(0.01, 0.15, 10)
    elif domain == "internet":
        alphas, betas, gammas = np.linspace(0.95, 0.99, 10), np.linspace(0.55, 0.8, 10), np.linspace(0.05, 0.25, 10)
    elif domain == "citation":
        alphas, betas, gammas = [0.999], np.linspace(0.35, 0.6, 10), np.linspace(0.2, 0.8, 10)
    else:
        alphas, betas, gammas = np.linspace(0.1, 1.0, 20), np.linspace(0.1, 1.0, 20), np.linspace(0.1, 1.0, 20)



    while count < max_tries:
        FLAG =  False
        for alpha, beta, gamma in itertools.product(*[alphas, betas, gammas]):

            init.makeStochasticCustom(np.asarray([alpha, beta, beta, gamma]))

            k = round(np.log2(N))

            best_G = kronecker_generator.generateStochasticKron(init, k)

            lcc = graph_util.get_lcc_undirected(best_G)[0]

            curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))
            
            #print(curr_avg_deg)

            curr_diam = nx.algorithms.diameter(lcc)

            avg_deg_error = abs(curr_avg_deg-deg)

            if avg_deg_error < tolerance:
                FLAG = True
                break
        if FLAG:
            break
        count += 1


    end_time = time()

    print('Graph_Name: Stochastic Kronecker Graph')
    print('Num_Nodes: ', nx.number_of_nodes(lcc), ' Avg_Deg : ', curr_avg_deg, ' Diameter: ', curr_diam)
    print('TIME: ', end_time - strt_time)
    return lcc, curr_avg_deg, curr_diam
########################################################################################################################

def lfr_benchmark_graph(N, deg, dia, dim, domain):
    '''
    Parameters of the graph:
    n (int) – Number of nodes in the created graph.
    tau1 (float) – Power law exponent for the degree distribution of the created graph. This value must be strictly greater than one.
    tau2 (float) – Power law exponent for the community size distribution in the created graph. This value must be strictly greater than one.
    mu (float) – Fraction of intra-community edges incident to each node. This value must be in the interval [0, 1].
    average_degree (float) – Desired average degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and min_degree must be specified, otherwise a NetworkXError is raised.
    :return: Graph Object
    '''
    strt_time = time()

    tau1 = [1.5, 2, 2.5, 3]
    tau2 = [1.5, 2, 2.5, 3]
    mu = [0.1,0.5]
    min_community = [20, 40, 60, 80, 100]
    import itertools

    params = list(itertools.product(*[tau1, tau2, mu, min_community]))

    avg_deg_for_plot = []

    max_tries = 2000
    count = 0
    tolerance = 0.5
    flag = False

    def lfr_call(G, N, tau1, tau2, mu, deg, min_comm):
        try:
            # print('CALLED')
            G['graph'] = nx.community.community_generators.LFR_benchmark_graph(n=N, tau1=tau1, tau2=tau2, mu=mu,
                                                                  average_degree=deg, min_community=min_comm)
        except:

            G = None
            pass

    manager = multiprocessing.Manager()

    while count < max_tries:

        for tup in params:

            try:

                G1 = manager.dict()

                lfr_process = multiprocessing.Process(target=lfr_call, name="LFR", args=(G1, N, tup[0],tup[1], tup[2], deg, tup[3]))


                lfr_process.start()

                lfr_process.join(10)



                if lfr_process.is_alive():
                    lfr_process.terminate()
                    lfr_process.join()
                    continue

                if not G1:
                    continue
                G1 = G1['graph']

                lcc = graph_util.get_lcc_undirected(G1)[0]

                curr_avg_deg = np.mean(list(dict(nx.degree(lcc)).values()))

                avg_deg_for_plot.append(curr_avg_deg)


                # print(deg, ' : CURR_AVG_DEG : ', curr_avg_deg)
                curr_deg_error = abs(curr_avg_deg - deg)

                if curr_deg_error < tolerance:
                    best_G = G1
                    best_avg_deg = curr_avg_deg
                    best_diam = nx.algorithms.diameter(lcc)
                    flag = True
                    break
            except:
                continue
        if flag == True:
            break
        count += 1


    # plot_hist('LFR_PLOT_DEG',avg_deg_for_plot)

    if count >= max_tries:
        raise('MAX_NUM_ITERATIONS Reached. Retry')

    end_time = time()

    print('Graph_Name: lfr_benchmark_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam




#####################################################################
if __name__=='__main__':
    N= [256, 512, 1024, 2048, 4096]
    Deg = [4, 6, 8, 10, 12]

    default_dia = 0
    default_dim = 128
    
    for domain in ["social", "biology", "internet"]:
        for deg in [4, 6, 8, 10, 12]:
            stochastic_kronecker_graph(1024, deg, 0, 128, domain)

        for n in [256, 512, 1024, 2048, 4096]:
            stochastic_kronecker_graph(n, 8, 0, 128, domain)

    # G, _, _ = barabasi_albert_graph(1024, 8, 0, 128)

    # G,something = graph_util.get_lcc(G.to_directed())
    # print(type(G))
    # print(G)

    import itertools
    all_combos = list(itertools.product(*[N, Deg]))
    # all_graphs = [watts_strogatz_graph, duplication_divergence_graph, powerlaw_cluster_graph, stochastic_block_model, r_mat_graph, hyperbolic_graph, stochastic_block_model]
    # all_graph_names = ["watts_strogatz_graph",
    #               "duplication_divergence_graph", "powerlaw_cluster_graph", "stochastic_block_model", "r_mat_graph",
    #               "hyperbolic_graph", "stochastic_block_model"]


    G,_,_ = lfr_benchmark_graph(1024,8,0,128)
    print(type(G))

    # all_graphs = [lfr_benchmark_graph]
    # all_graph_names = ["lfr_benchmark_graph"]
    #
    # for ind, active_graph in enumerate(all_graphs):
    #     print('********************************** ',all_graph_names[ind])
    #     for combi in all_combos:
    #         best_graph, best_avg_deg, best_dia = active_graph(combi[0], combi[1], default_dia, default_dim)
    #         print('N : '+str(combi[0]) + ' Deg : ' + str(combi[1]) +' CURR_AVG_DEG : ',str(best_avg_deg), ' BEST_DIA : ', str(best_dia))
    #         print('_____________________________________________________________________________________')



    # diam_list = []
    # avg_deg_list = []
    #
    # graph_name = 'Barabasi Albert Graph'
    # max_iters = 1000
    # graph_name_diam = '../diam_plots/'+graph_name+'_diam_'+str(max_iters)
    # graph_name_deg = '../diam_plots/' + graph_name + '_deg'+str(max_iters)
    #
    # while(len(diam_list))<max_iters:
    #     print('_____________________ ITER:',len(diam_list))
    #     G, avg_deg,diam = hyperbolic_graph(N=1024, deg=8,dia=0,dim=128)
    #     if np.round(abs(avg_deg - 8),1) <= 0.3:
    #
    #         diam_list.append(diam)
    #         avg_deg_list.append(avg_deg)

    # plot_hist(graph_name_diam, diam_list)
    # plot_hist(graph_name_deg, avg_deg_list)
