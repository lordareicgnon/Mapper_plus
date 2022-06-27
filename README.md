# Mapper_plus

(c) 2022 Esha Datta, Aditya Ballal & Leighton T. Izu

The Mapper Plus data pipeline is an extension of the algorithm Mapper which is based on topogical data analysis (TDA). Mapper Plus uses a community detection algorithm WLCF, first introduced by one is us, to get cluster assignments from a Mapper graph. We use the existing Mapper implementation provided by KeplerMapper in order to get a Mapper graph and then apply WLCF to get overlapping and disjoint clusters. 

More information on KeplerMapper can be found here: https://kepler-mapper.scikit-tda.org/en/latest/index.html
More information on WLCF can be found here: arxviv link: https://arxiv.org/abs/2202.11171
                                            github code: https://github.com/lordareicgnon/Walk_likelihood

# Installation

The scripts have been tested with python 3.8.5.

Required modules:
- numpy
- scipy
- sklearn
- kepler mapper (Can be pip installed with command 'pip install kmapper'. For other options, see: https://kepler-mapper.scikit-tda.org/en/latest/started.html)

# Overview

Here is an overview of the files included in the repository:
1. ```mapper_plus.py```: 
2. ```walk_likelihood.py```: File that defines the class walk_likelihood.
3. ```example.ipynb```: Jupyter notebook which explains the usage of the mapper plus.

# class mapper_plus

```class mapper_plus```
## Initialization
```def __init__(self)```

## get_mapper_graph:
```def get_mapper_graph(self,lens,data,**kepler_mapper_args)```
get_mapper_graph produces the first step of the Mapper Plus data pipeline, which is the Mapper graph. The nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions. This function implements on the KeplerMapper method map to produce a graph.


### Parameters: 
The parameters are same as used in kepler mapper. Some of the important parameters are

### Attributes:

__data:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The dataset on which mapper plus is used.

__N:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The size of the dataset.

__mapper_graph:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The Mapper graph produced by kepler mapper where the nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions stored in a dictionary. See KeplerMapper for more description. 

__Nb:__&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of nodes in the mapper graph.

__A:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The adjucancy matrix of the mapper graph specified in a 2-dimensional array of size Nb X Nb.

## get_overlapping_clusters:
```def get_mapper_graph(self,lens,data,**kepler_mapper_args)```
get_mapper_graph produces the first step of the Mapper Plus data pipeline, which is the Mapper graph. The nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions. This function implements on the KeplerMapper method map to produce a graph.


### Parameters: 
The parameters are same as used in kepler mapper. Some of the important parameters are

### Attributes:

__data:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The dataset on which mapper plus is used.

__N:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The size of the dataset.

__mapper_graph:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The Mapper graph produced by kepler mapper where the nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions stored in a dictionary. See KeplerMapper for more description. 

__Nb:__&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of nodes in the mapper graph.

__A:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The adjucancy matrix of the mapper graph specified in a 2-dimensional array of size Nb X Nb.

## get_non_overlapping_clusters:
```def get_mapper_graph(self,lens,data,**kepler_mapper_args)```
get_mapper_graph produces the first step of the Mapper Plus data pipeline, which is the Mapper graph. The nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions. This function implements on the KeplerMapper method map to produce a graph.


### Parameters: 
The parameters are same as used in kepler mapper. Some of the important parameters are

### Attributes:

__data:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The dataset on which mapper plus is used.

__N:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The size of the dataset.

__mapper_graph:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The Mapper graph produced by kepler mapper where the nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions stored in a dictionary. See KeplerMapper for more description. 

__Nb:__&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of nodes in the mapper graph.

__A:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The adjucancy matrix of the mapper graph specified in a 2-dimensional array of size Nb X Nb.



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
