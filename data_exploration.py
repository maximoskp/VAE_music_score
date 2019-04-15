#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:43:46 2019

@author: maximoskaliakatsos-papakostas
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
all_mnist = mnist.train.images

# load music data
rows = []
columns = []
with open('saved_data' + os.sep + 'data_tower.pickle', 'rb') as handle:
    d = pickle.load(handle)
    serialised_segments = d['serialised_segments']
    rows = d['rows']
    columns = d['columns']
all_music = serialised_segments

pca_mnist = PCA(n_components=2)
pca_music = PCA(n_components=2)

pca_mnist.fit( all_mnist.T )
pca_music.fit( all_music.T )

c_mnist = pca_mnist.components_
c_music = pca_music.components_

# plt.plot( c_mnist[0,:], c_mnist[1,:], 'x' )
# plt.plot( c_music[0,:], c_music[1,:], 'x' )

mnist_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
mnist_tsne.fit_transform( all_mnist )
music_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
mnist_tsne.fit_transform( all_music )