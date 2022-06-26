# Mapper_plus

(c) 2022 Esha Datta & Aditya Ballal

Description

# Installation

The scripts have been tested with python 3.8.5.

Required modules:
- numpy
- scipy
- sklearn
- walk-likelihood methods (file attached)
- mapper plus (link to installation)

# Overview

Here is an overview of the files included in the repository:
1. ```walk_likelihood.py```: File that defines the class walk_likelihood.
2. ```mapper_plus.py```: 
3. ```example.ipynb```: Jupyter notebook which explains the usage of the mapper plus.

# class mapper_plus

```class walk_likelihood```
## Initialization
```def __init__(self, X)```
### Parameters:
__X:__ ___{array-like, sparse matrix} of shape (N, N_p)___   


## Walk-likelihood algorithm (WLA):
```def WLA(self, m=None, U=None, init='NMF', lm=8, max_iter_WLA=100, thr_WLA=0.99, eps=0.00000001)```
### Parameters: 
__m:__ ___int, default=None___   
The number of communities for the partition of the network. Not required if initialization is custom.

__U:__ ___ndarray of shape (N, m), default=None___   
The matrix U refers to the initialization of the partition of the network of size N into m communities, only required to be specified if the intialization is custom.

__init:__ ___{'NMF', 'random', 'custom' }, default='NMF'___   
The method to initialize U: the partition of the network of size N into m communities for WLA. If U is provided, then the initialization is set to custom.

__l_max:__ ___int, default=8___   
The length of random-walks

__max_iter_WLA:__ ___int, default=100___   
The maximum number of iterations for WLA

__thr_WLA:__ ___float, default=0.99___   
The halting threshold for WLA

__eps:__ ___float, default=0.00000001___   
The lowest accepted non-zero value

### Attributes:

The following attributes of a network are obtained by WLA.

__N:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The number of nodes in the network.

__m:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The number of communities the network is partitioned into.

__w:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The outward rate of each node specified in a 1-dimensional array of size N .

__U:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The optimal partition of the network of size N into m communities specified in a 2-dimensional array of size N X m.

__comm_id:__ &nbsp; &nbsp; &nbsp; &nbsp; The community identity of each node for the partition of the network, specified in a 1-dimensional array of size N.

__communities:__ &nbsp;  A dictionary of communities with value as the nodes that belong to the community.

__modularity:__ &nbsp; &nbsp; &nbsp; The modularity of the partition of the network.

### Example

```
>>> import numpy as np
>>> from walk_likelihood import walk_likelihood
>>> X=np.load('Sample_networks/dolphins.npy')
>>> model=walk_likelihood(X)
>>> model.WLA(m=5)
>>> model.N
62
>>> model.m
5
>>> model.communities
{'Community 0': [12, 14, 16, 33, 34, 37, 38, 43, 44, 46, 49, 52, 53, 58, 61], 
'Community 1': [1, 5, 6, 9, 13, 17, 22, 25, 26, 27, 31, 32, 41, 48, 54, 56, 57, 60], 
'Community 2': [4, 11, 15, 18, 21, 23, 24, 29, 35, 45, 51, 55], 
'Community 3': [7, 19, 28, 30, 39, 47], 
'Community 4': [0, 2, 3, 8, 10, 20, 36, 40, 42, 50, 59]}
>>> model.comm_id
array([4, 1, 4, 4, 2, 1, 1, 3, 4, 1, 4, 2, 0, 1, 0, 2, 0, 1, 2, 3, 4, 2,
       1, 2, 2, 1, 1, 1, 3, 2, 3, 1, 1, 0, 0, 2, 4, 0, 0, 3, 4, 1, 4, 0,
       0, 2, 0, 3, 1, 0, 4, 2, 0, 0, 1, 2, 1, 1, 0, 4, 1, 0])
>>> model.modularity
0.47903563941299787
```



## Walk-likelihood Community Finder (WLCF):
```def WLCF(self, U=None, max_iter_WLCF=50, thr_WLCF=0.99, bifurcation_type='random', **WLA_params)```
### Parameters:

__U:__ ___ndarray of shape (N, m), default=None___   
The matrix U refers to the initialization of the partition of the network of size N into m communities, not required generally. 

__max_iter_WLCF:__ ___int, default=50___   
The maximum number of iterations for WLCF

__thr_WLCF:__ ___float, default=0.99___   
The halting threshold for WLCF

__bifurcation_type:__ ___{'random', 'NMF'}, default=random___   
The method used for initilizing bifurcation.

__**WLA_params:__   
The parameters that need to be specified to the Walk-likelihood Algorithm that will be used by WLCF

### Attributes:

The following attributes of a network are obtained by WLCF.

__N:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The number of nodes in the network.

__m:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The number of communities the network is partitioned into.

__w:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The outward rate of each node specified in a 1-dimensional array of size N .

__U:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The optimal partition of the network of size N into m communities specified in a 2-dimensional array of size N X m.

__comm_id:__ &nbsp; &nbsp; &nbsp; &nbsp; The community identity of each node for the partition of the network, specified in a 1-dimensional array of size N.

__communities:__ &nbsp;  A dictionary of communities with value as the nodes that belong to the community.

__modularity:__ &nbsp; &nbsp; &nbsp; The modularity of the partition of the network.
### Example

```
>>> import numpy as np
>>> from walk_likelihood import walk_likelihood
>>> X=np.load('Sample_networks/dolphins.npy')
>>> model=walk_likelihood(X)
>>> model.WLCF()
>>> model.N
62
>>> model.m
4
>>> model.communities
{'Community 0': [1, 5, 6, 7, 9, 13, 17, 19, 22, 25, 26, 27, 31, 32, 39, 41, 48, 54, 56, 57, 60], 
'Community 1': [4, 11, 15, 18, 21, 23, 24, 29, 35, 45, 51, 55], 
'Community 2': [3, 8, 12, 14, 16, 20, 33, 34, 36, 37, 38, 40, 43, 44, 46, 49, 50, 52, 53, 58, 59, 61], 
'Community 3': [0, 2, 10, 28, 30, 42, 47]}
>>> model.comm_id
array([3, 0, 3, 2, 1, 0, 0, 0, 2, 0, 3, 1, 2, 0, 2, 1, 2, 0, 1, 0, 2, 1,
       0, 1, 1, 0, 0, 0, 3, 1, 3, 0, 0, 2, 2, 1, 2, 2, 2, 0, 2, 0, 3, 2,
       2, 1, 2, 3, 0, 2, 2, 1, 2, 2, 0, 1, 0, 0, 2, 2, 0, 2])
>>> model.modularity
0.5202919188323247
```
