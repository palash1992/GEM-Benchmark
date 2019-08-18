try: import cPickle as pickle
except: import pickle
from os import environ
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import functools

font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=False)
rc('font', weight='bold')
rc('font', size=2)
rc('lines', markersize=2.5)
rc('lines', linewidth=0.5)
rc('xtick', labelsize=1)
rc('ytick', labelsize=1)
rc('axes', labelsize='small')
rc('axes', labelweight='bold')
rc('axes', titlesize='small')
rc('axes', linewidth=1)
rc("axes", labelsize = 15)
rc("axes", labelpad=4)
plt.rc('font', **font)
sns.set_style("darkgrid", {"xtick.bottom":True, "ytick.left":True})
sns.set_context("paper", font_scale=0.55 , rc={"lines.linewidth": 0.1, "xtick.major.size":0.1, "ytick.major.size":0.1, "legend.fontsize":0.001})
print(sns.plotting_context())


import pdb
import networkx as nx

def get_diameter(edge_list):
    G = nx.from_edgelist(edge_list)
    dia = nx.algorithms.diameter(G)
    print(dia)
    return dia


def get_degree_distribution(edge_list):
    G = nx.from_edgelist(edge_list)
    hist = nx.degree_histogram(G)
    return hist


def get_clustering_coeff(edge_list):
    G = nx.from_edgelist(edgelist=edge_list)
    clust_coeff = nx.average_clustering(G)
    # print(nx.number_of_nodes(G), ' : ',clust_coeff)
    return clust_coeff


def plot_real_stats(in_file='gem-ben/real_graphs_list_100.h5', out_file='realgraphProps.pdf'):

    df = pd.read_hdf(in_file, 'df')
    df['density'] = df.apply(lambda x: x['density']*1000,axis=1)

    print(df.keys())
    df['# of nodes'] = df['N'].apply(lambda x: round(x, 3))
    df['Avg. density (* 10^-3)'] = ((df['ave_degree']/df['N']).apply(lambda x: round(x, 4))*(1000)).round(3)
    df['Diameter'] = df['diameter'].apply(lambda x: round(x, 3))


    domains = ['Biological', 'Technological', 'Economic','Social']

    lines = None
    labels = None
    fin1, axarray1 = plt.subplots(len(domains), 3, figsize=(9.5, 5))


    lines_list, labels_list = [], []
    for idx, domain in enumerate(domains):
        if domain != "Social":
        ## Plot all the three below for each one of the domains
            df2 = df[df['networkDomain'] == domain]
        else:
            df2 = df
        ## Plot all the three below for each one of the domains
        

        ### Plot for Number of Nodes
        if domain != "Social":
            df_plott = df2['# of nodes']
        else:
            df_plott = df2['N']
        ax = sns.distplot(df_plott,
            kde=True, rug=True,
            ax=axarray1[idx, 0],
            rug_kws={"color": "#3cb371","label": "Data","lw":2},
            kde_kws={"color": "#e74c3c", "shade":False, "lw": 1, "label": "Gaussian KDE"},
            hist_kws={"histtype": "stepfilled", "linewidth": 0.5,
             "alpha": 0.9, "color": "darkgrey","label": "Histogram",  "edgecolor":'#1560bd'})
        lines, labels = ax.get_legend_handles_labels()


        ax.set_ylabel('')
        ax.get_legend().remove()
        if idx < len(domains) - 1:
            ax.set_xlabel('')
        print(domain)
        ### Sub Plot for Average Degree

        if domain != "Social":
            df_plott = df2['Avg. density (* 10^-3)']
        else:
            df_plott = df['density']
        ax = sns.distplot(df_plott,
            kde=True,
            rug=True,
            ax=axarray1[idx, 1],
            rug_kws={"color": "#3cb371","label": "Data", "lw":2},
            kde_kws={"color": "#e74c3c", "shade":False, "lw": 1, "label": "Gaussian KDE"},
            hist_kws={"histtype": "stepfilled", "linewidth": 0.5,
             "alpha": 0.9, "color": "darkgrey","label": "Histogram","edgecolor":'#1560bd'})
        lines,labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()

        if idx < len(domains) - 1:
            ax.set_xlabel('')

        ### Sub Plot for Diameter
        if domain != "Social":
            df_plott = df2['Diameter']
        else:
            df_plott = df['Diameter']
        ax = sns.distplot(df_plott,
                          kde=True,
                          rug=True,
                          ax=axarray1[idx, 2],
                          rug_kws={"color": "#3cb371","label": "Data", "lw": 2},
                          kde_kws={"color": "#e74c3c", "shade": False, "lw": 1, "label": "Gaussian KDE"},
                          hist_kws={"histtype": "stepfilled", "linewidth": 0.5,
                                    "alpha": 0.9, "color": "darkgrey", "label": "Histogram", "edgecolor": '#1560bd'})
        lines, labels = ax.get_legend_handles_labels()


        ## For different domains
        ax.set_ylabel(domain)
        ax.yaxis.set_label_position("right")
        if idx < len(domains) - 1:
            ax.set_xlabel('')


        ax.get_legend().remove()

    plt.savefig(
           out_file,
           dpi=300, format='pdf', bbox_inches='tight'
        )
