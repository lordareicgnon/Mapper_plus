import streamlit as st
import pandas as  pd
import numpy as np
import sklearn
from mapper_plus import mapper_plus
st.write("""
# Mapper Plus
""")

st.write("Write description of mapper plus here")

#import kmapper as km

#placeholder = st.empty()
#placeholder.title("A/B Test Comparison")
see_results=0
uploaded_file='False'
with st.sidebar:
    Wine_data = st.checkbox(
        "Use Sample Data", False, help="Wine Dataset")

    if not Wine_data:


        uploaded_file = st.file_uploader("Upload CSV", type=".csv")
        #st.write(uploaded_file)
        if uploaded_file:
            data = np.loadtxt(uploaded_file, delimiter=',')
            file_name=uploaded_file

    else:
        from sklearn.datasets import load_wine
        file_name='wine.csv'
        data = load_wine()['data']
    if uploaded_file or Wine_data:
        normalize = st.checkbox(
            "Normalize Data", True, help="Normalize Data using standard method")
        if normalize:
            X=(data-np.mean(data,axis=0))/np.std(data,axis=0)
        else:
            X=data
#submit=False
#with st.form("parameters"):
if uploaded_file or Wine_data:
    st.markdown("## Mapper Parameters")

    st.markdown("### Lens")

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
    no_num_clusterer={'DBSCAN','Affinity Propagation','Mean-shift','OPTICS'}
    if clusterer_name not in no_num_clusterer:
        n_cluster=cols2[1].number_input('Number of clusters',min_value=1, max_value=100,step=1,value=2)
    if clusterer_name in no_num_clusterer:
        clusterer= clusterers[clusterer_name]()
    elif clusterer_name=='Gaussian mixtures':
        clusterer= clusterers[clusterer_name](n_components=n_cluster)
    else:
        clusterer= clusterers[clusterer_name](n_clusters=n_cluster)

    #st.write(clusterers[clusterer])
    st.markdown("### Gain and Resolution")
    cols = st.columns((1, 1))
    gain = cols[0].number_input('Gain',min_value=0.0000001, max_value=0.9999999,value=0.6)
    resolution = cols[1].number_input('Resolution',min_value=1, max_value=100,step=1,value=8)
    with st.form(key="my_form"):

        run=st.form_submit_button(label="Run Mapper Plus")
    if not run:
        st.stop()
    N=X.shape[0]
    import kmapper as km
    mapper=km.KeplerMapper()
    lens=np.zeros((N,0))
    if "PCA" in lenses:
        from sklearn.decomposition import PCA
        #st.write(X)
        lens1 = mapper.fit_transform(X,projection=PCA(n_components=n_comps_PCA))
        lens=np.concatenate((lens,lens1),axis=1)
    if "L2 Norm" in lenses:
        lens2 = mapper.fit_transform(X,projection='l2norm')
        lens = np.concatenate((lens,lens2),axis=1)
    if "IsolationForest" in lenses:
        from sklearn import ensemble
        isomodel = ensemble.IsolationForest(random_state=1729)
        isomodel.fit(X)
        lens3 = isomodel.decision_function(X).reshape((X.shape[0], 1))
        lens = np.concatenate((lens,lens3),axis=1)

    ranonce=1
    cover = km.Cover(n_cubes = resolution, perc_overlap = gain)
    model=mapper_plus()
    st.write("....... Running")
    st.write("....... Finding Mapper Graph")
    model.get_mapper_graph(lens,X,cover=cover,clusterer=clusterer,)
    st.write("....... Finding overlapping clusters")
    model.get_overlapping_clusters()
    st.write("##### "+str(len(model.overlapping_clusters))+" overlapping clusters found")
    overlap_str=''
    for comms in range(len(model.overlapping_clusters)):
        if comms>0:
            overlap_str+='\n'
        overlap_str+='Cluster '+str(comms+1)+','
        with st.expander("Cluster "+str(comms+1)):
            st.write(str(model.overlapping_clusters[comms])[1:-1])
        overlap_str+=str(model.overlapping_clusters[comms])[1:-1]
    st.download_button('Download Overlapping Clusters', overlap_str,file_name='overlapping_clusters_'+file_name)
    st.write("....... Finding disjoint clusters")
    model.get_non_overlapping_clusters(new_method=1)
    st.write("##### "+str(len(model.non_overlapping_clusters))+" disjoint clusters found")
    disjoint_str=''
    for comms in range(len(model.non_overlapping_clusters)):
        if comms>0:
            disjoint_str+='\n'
        disjoint_str+='Cluster '+str(comms+1)+','
        with st.expander("Cluster "+str(comms+1)):
            st.write(str(model.non_overlapping_clusters[comms])[1:-1])
            disjoint_str+=str(model.non_overlapping_clusters[comms])[1:-1]
    st.download_button('Download Disjoint Clusters', overlap_str,file_name='disjoint_clusters_'+file_name)

    #st.write("These are overlapping clusters")
    #st.write(model.overlapping_clusters)
    #st.write("These are non-overlapping clusters")
    #st.write(model.non_overlapping_clusters)


        #    run=st.form_submit_button(label="Run Another Analysis")
