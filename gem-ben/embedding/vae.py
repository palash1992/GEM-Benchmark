disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try: import cPickle as pickle
except: import pickle

import numpy as np
import scipy.io as sio
import networkx as nx

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.evaluation import visualize_embedding as viz
from .sdne_utils import *

from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as KBack
from keras import callbacks

from theano.printing import debugprint as dbprint, pprint
from time import time


class VAE(StaticGraphEmbedding):
   """`Variational Graph Auto-Encoders`_.
    Variational Graph Auto-Encoders utilizes a GCN as encoder and inner product as decoder, 
    which provides embedding with higher quality than autoencoders.
    
    Args:
        hyper_dict (object): Hyper parameters.
        kwargs (dict): keyword arguments, form updating the parameters
    
    Examples:
        >>> from gemben.embedding.vae import VAE
        >>> file_prefix = "gemben/data/sbm/graph.gpickle"
        >>> G = nx.read_gpickle(file_prefix)
        >>> node_colors = pickle.load(
            open('gemben/data/sbm/node_labels.pickle', 'rb') )
        >>> embedding = VAE(d=128, beta=5, nu1=1e-6, nu2=1e-6,
                        K=3, n_units=[300, 100, ],
                        beta_vae=10.0,
                        n_iter=500, xeta=1e-3,
                        n_batch=500,
                        modelfile=['gemben/intermediate/enc_model.json',
                                   'gemben/intermediate/dec_model.json'],
                        weightfile=['gemben/intermediate/enc_weights.hdf5',
                                    'gemben/intermediate/dec_weights.hdf5'])
        >>> embedding.learn_embedding(G)
        >>> X = embedding.get_embedding()
        >>> mi = np.var(X, axis=0)
        >>> highest_z = np.argmax(mi)
        >>> highest_z_highest = np.max(X[:, highest_z])
        >>> highest_z_lowest = np.min(X[:, highest_z])
        >>> X[:512, highest_z] = highest_z_lowest
        >>>  X[512:, highest_z] = highest_z_highest
        >>> mi2 = np.log(mi)
        >>> G_X_hat = embedding.get_reconstructed_adj(embed=X)
        >>> node_colors_arr = [None] * node_colors.shape[0]
        >>> for idx in range(node_colors.shape[0]):
                node_colors_arr[idx] = np.where(node_colors[idx, :].toarray() == 1)[1][0]
                MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(
                G, embedding, X, None )
        >>> print('MAP:')
        >>> print(MAP)
        >>> viz.plot_embedding2D( G_X_hat,
                                di_graph=G,
                                node_colors=node_colors_arr )
        >>> plt.savefig('vae_sbm_g_x_hat.pdf', bbox_inches='tight')
        >>> plt.figure()
        >>> viz.plot_embedding2D(X,
                                 di_graph=G,
                                 node_colors=node_colors_arr )
        >>> plt.savefig('vae_sbm_z.pdf', bbox_inches='tight')
    .. _Variational Graph Auto-Encoders:
        https://arxiv.org/pdf/1611.07308.pdf
    """
    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the Autoencoder class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden
                     layers of encoder/decoder, not including the units
                     in the embedding layer
            beta: coefficent of KL-divergence in VAE
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            n_iter_subs: number of sgd iterations for subsequent embeddings (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
            node_frac: Fraction of nodes to use for random walk
            n_walks_per_node: Number of random walks to do for each selected nodes
            len_rw: Length of every random walk
        '''
        hyper_params = {
            'method_name': 'vae',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        S = nx.to_scipy_sparse_matrix(graph)
        self._node_num = graph.number_of_nodes()
        t1 = time()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        self._encoder = get_variational_encoder(self._node_num, self._d,
                                                self._n_units,
                                                self._nu1, self._nu2,
                                                self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_variational_autoencoder(
            self._encoder,
            self._decoder
        )

        # Initialize self._model
        # Input
        x_in = Input(shape=(self._node_num,), name='x_in')
        # Process inputs
        # [x_hat, y] = self._autoencoder(x_in)
        [x_hat, y_mean, y_std, y2] = self._autoencoder(x_in)
        # Outputs
        x_diff = merge([x_hat, x_in],
                       mode=lambda (a, b): a - b,
                       output_shape=lambda L: L[1])
        y_log_var = KBack.log(KBack.square(y_std))
        vae_loss = merge(
            [y_mean, y_std],
            mode=lambda (a, b): -0.5 * KBack.sum(
                1 + KBack.log(KBack.square(b)) - KBack.square(a) - KBack.square(b),
                axis=-1
            ),
            output_shape=lambda L: (L[1][0], 1)
        )

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains b
            '''
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1
            )

        def weighted_mse_vae(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains KL-divergence
                y_true: Contains np.zeros(mini_batch)
            '''
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.mean(
                # KBack.abs(y_pred),
                KBack.abs(KBack.reshape(y_pred, [min_batch_size, 1])),
                axis=-1
            )

        # Model
        self._model = Model(input=x_in, output=[x_diff, vae_loss])
        # sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=adam,
            loss=[weighted_mse_x, weighted_mse_vae],
            loss_weights=[1, self._beta_vae]
        )

        history = self._model.fit_generator(
            generator=batch_generator_vae(S, self._beta, self._n_batch, True),
            nb_epoch=self._num_iter,
            samples_per_epoch=S.shape[0] // self._n_batch,
            verbose=1,
            callbacks=[callbacks.TerminateOnNaN()]
        )
        loss = history.history['loss']
        # Get embedding for all points
        if loss[0] == np.inf or np.isnan(loss[0]):
            print 'Model diverged. Assigning random embeddings'
            self._Y = np.random.randn(self._node_num, self._d)
        else:
            self._Y = model_batch_predictor(
                self._autoencoder,
                S,
                self._n_batch,
                meth='vae')

        submodel_gen = batch_generator_vae(S, self._beta, self._n_batch, True)
        x = np.concatenate([next(submodel_gen)[0] for _ in range(100)], axis=0)
        vae_submodel = Model(x_in, self._autoencoder(x_in))
        _, _, log_std, _ = vae_submodel.predict(x)
        mean = np.mean(log_std)
        std = np.std(log_std)
        print('log std mean and std')
        print(mean)
        print(std)

        t2 = time()
        # Save the autoencoder and its weights
        if(self._weightfile is not None):
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if(self._modelfile is not None):
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if(self._savefilesuffix is not None):
            saveweights(self._encoder,
                        'encoder_weights_' + self._savefilesuffix + '.hdf5')
            saveweights(self._decoder,
                        'decoder_weights_' + self._savefilesuffix + '.hdf5')
            savemodel(self._encoder,
                      'encoder_model_' + self._savefilesuffix + '.json')
            savemodel(self._decoder,
                      'decoder_model_' + self._savefilesuffix + '.json')
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt',
                       self._Y)
        return self._Y, (t2 - t1)

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(
                    embed,
                    batch_size=self._n_batch
                )[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open('decoder_model_' + filesuffix + '.json').read())
            except:
                print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_'+filesuffix+'.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
                exit()
            if node_l is not None:
                return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return decoder.predict(embed, batch_size=self._n_batch)


if __name__ == '__main__':
    # load synthetic graph
    file_prefix = "gem/data/sbm/graph.gpickle"
    G = nx.read_gpickle(file_prefix)
    node_colors = pickle.load(
        open('gem/data/sbm/node_labels.pickle', 'rb')
    )
    embedding = VAE(d=128, beta=5, nu1=1e-6, nu2=1e-6,
                    K=3, n_units=[300, 100, ],
                    beta_vae=10.0,
                    n_iter=500, xeta=1e-3,
                    n_batch=500,
                    modelfile=['gem/intermediate/enc_model.json',
                               'gem/intermediate/dec_model.json'],
                    weightfile=['gem/intermediate/enc_weights.hdf5',
                                'gem/intermediate/dec_weights.hdf5'])
    embedding.learn_embedding(G)
    X = embedding.get_embedding()
    mi = np.var(X, axis=0)
    highest_z = np.argmax(mi)
    highest_z_highest = np.max(X[:, highest_z])
    highest_z_lowest = np.min(X[:, highest_z])
    X[:512, highest_z] = highest_z_lowest
    X[512:, highest_z] = highest_z_highest

    mi2 = np.log(mi)
    # pdb.set_trace()
    G_X_hat = embedding.get_reconstructed_adj(embed=X)
    # import pdb
    # pdb.set_trace()
    node_colors_arr = [None] * node_colors.shape[0]
    for idx in range(node_colors.shape[0]):
        node_colors_arr[idx] = np.where(node_colors[idx, :].toarray() == 1)[1][0]
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(
        G, embedding, X, None
    )
    print('MAP:')
    print(MAP)
    viz.plot_embedding2D(
        G_X_hat,
        di_graph=G,
        node_colors=node_colors_arr
    )
    plt.savefig('vae_sbm_g_x_hat.pdf', bbox_inches='tight')
    plt.figure()
    viz.plot_embedding2D(
        X,
        di_graph=G,
        node_colors=node_colors_arr
    )
    plt.savefig('vae_sbm_z.pdf', bbox_inches='tight')
