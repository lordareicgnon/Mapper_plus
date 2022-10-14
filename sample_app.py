import streamlit as st
import pandas as  pd
import numpy as np
st.write("""
# Mapper Plus
""")
#from mapper_plus import mapper_plus
st.write("Write description of mapper plus here")

#import kmapper as km

#placeholder = st.empty()
#placeholder.title("A/B Test Comparison")

with st.sidebar:

    uploaded_file = st.file_uploader("Upload CSV", type=".csv")

#submit=False
#with st.form("parameters"):

st.markdown("## Mapper Parameters")

st.markdown("### Lens")
no_lenses = st.checkbox(
    "Use Raw Data with no lens", False, help="Use Raw Data with no lenses")
if not no_lenses:
    Nl = st.number_input('Number of lenses',min_value=1, max_value=100,step=1,value=2)
    #submit = st.form_submit_button()#on_click=False)
    lens_input=[]
    #with st.form("Lenses"):
    colslens=[]
    for i in range(Nl):

        a=int(i/2)
        b=i-2*a
        if b==0:
            colslens.append(st.columns((1, 1)))
        lens_input.append(colslens[a][b].selectbox(
        "Lens "+str(i+1)+":", ["PCA", "IsolationForest", "L2 Norm"], index=0))
    lenses=set(lens_input)
    if "PCA" in lenses:
        n_comps_PCA=sum(np.array(lens_input)=="PCA")

import sklearn.cluster
import sklearn.mixture
st.markdown("### Clusterer")
cols2=st.columns((1, 1))
clusterers={'K-Means':sklearn.cluster.KMeans,
'Affinity Propagation':sklearn.cluster.AffinityPropagation,
'Mean-shift':sklearn.cluster.MeanShift,
'Spectral Clustering':sklearn.cluster.SpectralClustering,
'Agglomerative clustering':sklearn.cluster.AgglomerativeClustering,
'DBSCAN':sklearn.cluster.DBSCAN,
'OPTICS':sklearn.cluster.OPTICS,
'Gaussian mixtures':sklearn.mixture.GaussianMixture,
'BIRCH':sklearn.cluster.Birch}#,
#'Bisecting K-Means':sklearn.cluster.BisectingKMeans}
clusterer_name=cols2[0].selectbox("Clustering Method :", list(clusterers.keys()), index=0)
no_num_clusterer={'DBSCAN','Affinity propagation','Mean-shift','OPTICS'}
if clusterer_name not in no_num_clusterer:
    n_cluster=cols2[1].number_input('Number of clusters',min_value=1, max_value=100,step=1,value=2)
if clusterer_name not in clusters:
    clusterer= clusterers[cluster_name](n_clusters=n_cluster)
else:
    clusterer= clusterers[cluster_name]()
#st.write(clusterers[clusterer])
st.markdown("### Gain and Resolution")
cols = st.columns((1, 1))
gain = cols[0].number_input('Gain',min_value=0.0000001, max_value=0.9999999,value=0.6)
resolution = cols[1].number_input('Resolution',min_value=1, max_value=100,step=1,value=8)
print('Haha')

#import kmapper as km
#cover = km.Cover(n_cubes = resolution, perc_overlap = gain)
#model=mp.mapper_plus()
#model.get_mapper_graph(lens,X,cover=cover,clusterer=clusterer,)
#model.get_overlapping_clusters()
#model.get_non_overlapping_clusters(new_method=1)
