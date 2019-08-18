try: import cPickle as pickle
except: import pickle
from os import environ

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
from matplotlib import rc, gridspec
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn
font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=False)
rc('font', weight='bold')
rc('font', size=8)
rc('lines', markersize=2.5)
rc('lines', linewidth=0.5)
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)
rc('axes', labelsize='small')
rc('axes', labelweight='bold')
rc('axes', titlesize='small')
rc('axes', linewidth=1)
plt.rc('font', **font)
seaborn.set_style("darkgrid")
import pdb

graph_attrs = {
    "Domain": ["Economic", "Biological", "Technological", "Social"],
    "# of Nodes": [256, 512, 1024],
    "Density": [0.002, 0.008, 0.02]
}

palette = {"gf":"#9b59b6", "rand":"#3498db", "pa":"#F08A4B", "lap":"#DE369D", 
           "hope":"#59CD90", "cn":"#F9ECCC", "aa": "#EFD6D2","sdne":"#EE6352",
           "jc":"#C6D8FF"}
marker = {"gf":"o", "rand":"v", "pa":"s", "lap":"P", "hope":"D",
          "cn":"P", "aa":"*","sdne":"X", "jc":"<"}
labelorder = ["rand","cn", "aa",  "pa","jc","lap","gf", "hope","sdne"]
labelmap = {"gf":"Graph Factorization", "rand":"Random", 
            "pa":"Preferential Attachment", "lap":"Laplacian Eigmaps", 
            "hope":"HOPE", "cn":"Common Neighbor", "aa":"Adamic-Adar",
            "sdne":"SDNE", "jc":"Jaccard Coeff"}

def plot_benchmark(methods, metric='MAP', s_sch='rw'):
    graph_info = pd.read_hdf('real_graphs_list_100.h5', 'df')
    graph_names = range(len(graph_info))
    path = 'gem/intermediate/'
    for m , g in itertools.product(*[methods, graph_names]):
        d = graph_info.at[g, "networkDomain"]
        g = str(g)
        try:
            df = pd.read_hdf(path+
                "%s_%s_%s_lp_%s_data_hyp.h5" % (d, g, m, s_sch),
                "df"
            )
            df = df[df["Round Id"] == 0]
        except:
            print(path + '%s_%s_%s_lp_%s_data_hyp.h5 not found. Ignoring data set' % (d, g, m, s_sch))
            continue
        df["Domain"], df["Method"], df["Graph"] = d, m, g
        df_all = df_all.append(df).reset_index()
        df_all = df_all.drop(['index'], axis=1)
    df_all['# of Nodes'] = df_all['N']
    df_all['Density'] = df_all['deg'] / df_all['# of Nodes']
    if df_all.empty:
        return
    gs = gridspec.GridSpec(3, 12)
    plot_shape = (3, 12)
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(wspace=1, hspace=0.2)
    
    data_idx = 0
    lines =None
    labels = None
    ylimits = {"P@100":[[[0, 0.05],[0, 0.25],[0, 0.12],[0, 0.5]],
               [[0, 0.25],[0, 0.26],[0, 0.18]],
               [[0, 0.18],[0, 0.4],[0, 0.8]]],
              "MAP":[[[0, 0.12],[0, 0.15],[0, 0.12],[0, 0.22]],
               [[0, 0.18],[0, 0.1],[0, 0.15]],
               [[0, 0.15],[0, 0.15],[0, 0.3]]]}
    for attr_idx, attr in enumerate(graph_attrs.keys()):
        for val_idx, attr_val in enumerate(graph_attrs[attr]):
            plot_idx = np.unravel_index(data_idx, (3, 4))
            if attr == "Domain":
                ax0 = plt.subplot(gs[0, val_idx * 3: (val_idx + 1) * 3])
            else:
                ax0 = plt.subplot(gs[attr_idx, val_idx * 4: (val_idx + 1) * 4])
            data_idx+=1
            if attr == 'Density':
                df_grouped = df_all[df_all[attr] > (attr_val - 0.005)]
                df_grouped = df_grouped[df_grouped[attr] < (attr_val + 0.005)]
            else:
                df_grouped = df_all[df_all[attr]==attr_val]
                        
            df_grouped = df_grouped[["dim", "Round Id", "LP %s" % metric, "Method", "Graph"]]
            df_grouped['LP %s' % metric] = df_grouped['LP %s' % metric].astype('float')
            df_grouped = df_grouped.groupby(
                ["dim", "Round Id", "Method", "Graph"]
            ).mean().reset_index()
            try:
                df_grouped['unit']=df_grouped.apply(
                    lambda x:'%s_%s' % (x['Round Id'],x['Graph']),
                    axis=1
                )
            except:
                pdb.set_trace()
            df_grouped = df_grouped.drop(['Round Id', "Graph","unit"], axis=1)
            value = "LP %s" % metric
            ax = seaborn.lineplot(
                x="dim", y=value, err_style="bars", hue="Method",
                dashes=False, palette=palette,markers =marker,
                legend='brief', style ="Method", data=df_grouped, ax=ax0
            )
            ax.set_title(
                attr + " : " + str(attr_val),
                pad=0.1,  fontdict ={'fontsize': 7}
            )
            ax.set_ylim(ylimits[metric][attr_idx][val_idx])
            ax.set_ylabel('')    
            ax.set_xlabel('')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', which='major', pad=-0.4, labelsize=4.8)
            
            ax.set_xscale('log', basex=2)
            ax.get_legend().remove()
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_markeredgecolor('k')
                ax.lines[line_i].set_markeredgewidth('0.2')
                ax.lines[line_i].set_markersize('3.5')

            if data_idx==10:
                lines,labels = ax.get_legend_handles_labels()

    line_label ={k: v for k, v in zip(labels, lines)}
    labels = [labelmap[l] for l in labelorder ]
    lines = [line_label[k] for k in labelorder]     
    plt.legend(lines,labels,loc='upper center',
               bbox_to_anchor=(-0.6,-0.24),
               ncol=len(methods)//2+1, fancybox=True,
               shadow=True,prop={'size': 5})
    labelposy={"MAP":[-0.06,0.5], "P@100":[-0.14,1.4]}
    labelposx={"MAP":[3,0.05], "P@100":[3,0.05]}
    plt.text(labelposx[metric][0], labelposy[metric][0],
             'Dimension', ha='center', fontdict ={'fontsize': 8})
    plt.text(labelposx[metric][1],labelposy[metric][1], metric,
             va='center', rotation='vertical', fontdict ={'fontsize': 8})
    plt.savefig(
       'benchmark_real_%s.pdf' % metric,
       dpi=300, format='pdf', bbox_inches='tight'
    )
    plt.show()
    plt.clf()
