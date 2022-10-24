# Mapper_plus

(c) 2022 Esha Datta, Aditya Ballal, Javier Lopez & Leighton T. Izu

The Mapper Plus data pipeline is an extension of the algorithm Mapper which is based on topogical data analysis (TDA). Mapper Plus uses a community detection algorithm WLCF, first introduced by one is us, to get cluster assignments from a Mapper graph. We use the existing Mapper implementation provided by KeplerMapper in order to get a Mapper graph and then apply WLCF to get overlapping and disjoint clusters. 

More information on KeplerMapper can be found here: https://kepler-mapper.scikit-tda.org/en/latest/index.html

More information on WLCF can be found here: 
- arxviv link: https://arxiv.org/abs/2202.11171
- github code: https://github.com/lordareicgnon/Walk_likelihood

# Installation

The scripts have been tested with python 3.8.5.

Required modules:
- numpy
- scipy
- sklearn
- kepler mapper (Can be pip installed with command 'pip install kmapper'. For other options, see: https://kepler-mapper.scikit-tda.org/en/latest/started.html)

In order to install the streamlit app of mapper-plus follow the steps below
1. Download the repository
2. Open Terminal at the folder
3. pip install -r requirements.txt
4. streamlit run sample_app.py

# Overview

Here is an overview of the files included in the repository:
1. ```mapper_plus.py```: File that contains the implementation of mapper plus.
2. ```walk_likelihood.py```: File that defines the class walk_likelihood.
3. ```example.ipynb```: Jupyter notebook which explains the usage of the mapper plus.

# class mapper_plus

```class mapper_plus```
## Initialization
```def __init__(self)```

## get_mapper_graph:
```def get_mapper_graph(self,lens,data,**kepler_mapper_args)```

This function produces the first step of the Mapper Plus data pipeline, which is the Mapper graph. The nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions. This function implements the KeplerMapper method map to produce a graph.


### Parameters: 
The parameters are same as used in the kepler mapper method map. Some of the important parameters are:
- lens (numpy array): define a N-by-M array that is an M-dimensional representation of your N observations of data. This is typically the output of a fit_transform command
- data (numpy array): dataset of N observations x P parameters to run clustering on 
- clusterer (Default: DBSCAN): provide your choice of a scikit-learn API compatible clustering algorithm that provides both fit and predict
- cover (kmapper.Cover): define the cover scheme for your lens, refer to Kepler Mapper documentation for full details   

### Attributes:

__data:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  The dataset on which mapper plus is used.

__N:__	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The size of the dataset.

__mapper_graph:__  &nbsp; The Mapper graph produced by kepler mapper where the nodes are clusters of obseravtions in your dataset and edges connect nodes that share obseravtions stored in a dictionary. See KeplerMapper for more description. 

__Nb:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of nodes in the mapper graph.

__A:__ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The adjacency matrix of the mapper graph specified in a 2-dimensional array of size Nb X Nb.

## get_overlapping_clusters:
```def get_overlapping_clusters(self,**WLCF_args)```
This function clusters the nodes of the Mapper graph into communities using WLCF, which in turn clusters the original data into overlapping communities. This function can only be used if the mapper graph is obtained using the function get_mapper_graph.

### Parameters: 
The parameters are same as used in Walk-likelihood Community Finder. For details go to https://github.com/lordareicgnon/Walk_likelihood.

### Attributes:

__mapper_nodes:__  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The names of nodes of the mapper graph that are clustered into communities.

__comm_id_mapper_nodes:__ &nbsp; Community identity of each node of the mapper graph specified in a 1-dimensional array of size Nb.

__overlapping_clusters:__ &nbsp; &nbsp; &nbsp; &nbsp; A list of the corresponding overlapping clusters of the original dataset where each entry of this list is a list of observations belonging to that cluster.

__m:__&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of overlapping clusters.


## get_non_overlapping_clusters:
```def get_non_overlapping_clusters(**WLA_args)```
This function clusters each observations into overlapping communities based on their previous non-overlapping community assignment and the mapper graph. This function can only be used if the mapper graph is obtained using the function get_overlapping_clusters.

### Parameters: 
The parameters are same as used in Walk-likelihood Algortihm. For details go to https://github.com/lordareicgnon/Walk_likelihood.


### Attributes:

__comm_id:__ &nbsp; Community identity of each observation specified in a 1-dimensional array of size Nb.

__non_overlapping_clusters:__ &nbsp; &nbsp; &nbsp; &nbsp; A list of the non-overlapping clusters of the original dataset where each entry of this list is a list of observations belonging to that cluster.

__m:__&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The total number of non-overlapping clusters.


### Example

Importing all necessary packages
```
>>> import pandas as pd
>>> import numpy as np
>>> import sklearn
>>> import mapper_plus as mp 
>>> import kmapper as km 
>>> from sklearn.decomposition import PCA 
```

Load in dataset
```
>>> X=np.load('Abalone_standardized.npy')
>>> X.shape
(4177, 8)
```

Using PCA lens on the data
```
>>> mapper = km.KeplerMapper(verbose=1)
KeplerMapper(verbose=1)
>>> lens = mapper.fit_transform(X,projection=PCA(n_components=2))
..Composing projection pipeline of length 1:
	Projections: PCA(n_components=2)
	Distance matrices: False
	Scalers: MinMaxScaler()
..Projecting on data shaped (4177, 8)

..Projecting data using: 
	PCA(n_components=2)


..Scaling with: MinMaxScaler()
```

Specifying cover
```
>>> cover = km.Cover(n_cubes = 10, perc_overlap = 0.7)
```

Obtaining mapper graph using mapper plus
```
>>> model=mp.mapper_plus()
>>> model.get_mapper_graph(lens,X,cover=cover,clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1618033),)
Mapping on data shaped (4177, 8) using lens shaped (4177, 2)

Creating 100 hypercubes.

Created 1558 edges and 162 nodes in 0:00:02.477715.
```

Finding overlapping clusters
```
>>> model.get_overlapping_clusters()
We found 7 overlapping clusters
```

Finding non-overlapping clusters
```
>>> model.get_non_overlapping_clusters()
We found 7 non-overlapping clusters
```
