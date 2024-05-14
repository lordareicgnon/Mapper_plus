import streamlit as st
import pandas as  pd
import numpy as np
import sklearn
from mapper_plus import mapper_plus
from download_button_file import download_button
from io import StringIO
import pandas as pd
run=0
st.write("""
# MapperPlus Ⓒ \n
2022 Esha Datta, Aditya Ballal, Javier Lopez & Leighton T. Izu
""")

#more_information = st.checkbox("More Information", False)

with st.expander("More Information"):
    st.write("""
    The Mapper Plus data pipeline is an extension of the algorithm Mapper which is based on topogical data analysis (TDA). Mapper Plus uses a community detection algorithm WLCF, first introduced by one is us, to get cluster assignments from a Mapper graph. We use the existing Mapper implementation provided by KeplerMapper in order to get a Mapper graph and then apply WLCF to get overlapping and disjoint clusters.
    \n
    More information on KeplerMapper can be found here: https://kepler-mapper.scikit-tda.org/en/latest/index.html
    \n
    More information on WLCF can be found here:
    \n
    - arxviv link: https://arxiv.org/abs/2202.11171
    \n
    - github code: https://github.com/lordareicgnon/Walk_likelihood

    """)

#with st.sidebar:
#    with st.expander("More information on parameters"):
#        st.write("This is more information on parameters")

#ranonce=0
#runmapperplus = st.checkbox("Run MapperPlus", False)
runmapperplus=1
#with st.expander("Run MapperPlus"):
if runmapperplus:
    st.markdown("## Upload data for Clustering")
    see_results=0
    uploaded_file='False'
    with st.expander("ℹ️ More information"):
        st.write("Upload data in CSV format.")
    #with st.sidebar:
    Wine_data = st.checkbox(
        "Use Sample Data", False, help="Wine Dataset")


    if not Wine_data:


        uploaded_file = st.file_uploader("Upload CSV", type=".csv")
        #st.write(uploaded_file)
        if uploaded_file:
            #csvfile=StringIO(uploaded_file)
            #data = np.loadtxt(uploaded_file, delimiter=',')
            datacols = st.columns((1, 1, 1))
            transpose=datacols[0].checkbox("Transpose Data", False)
            head=datacols[1].checkbox("Contains headers", False)
            ids=datacols[2].checkbox("Contains indices", False)
            
            

            #if head:
            #    df=pd.read_csv(uploaded_file)
            #else:
            #    df=pd.read_csv(uploaded_file,header=None)
            df=pd.read_csv(uploaded_file,header=None)

            if transpose:
                df=df.T
            if head:
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            if ids:
                df=df.set_index(df.columns.tolist()[0])

                
            st.write('### Data Uploaded')

            st.write(df)
            data=np.array(df)
            file_name=uploaded_file.name

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
        st.markdown("## MapperPlus Parameters")

        st.markdown("### Lens")
        with st.expander('ℹ️ More information'):
            st.write("Lens is any function that projects the dataset into a lower dimension. We provide option of PCA, Isolation Forest and l2 norm for lens. Multiple lenses can be chosen.")

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
        with st.expander('ℹ️ More information'):
            st.write("Clusterer is the clustering algorithm that MapperPlus applies within each cover. We provide the choice of 9 clustering algorithms:")
            st.write("1. K-Means \n 2. Affinity Propagation \n 3. Mean-shift \n 4. Spectral Clustering \n 5. Agglomerative clustering \n 6. DBSCAN \n 7. OPTICS \n 8. Gaussian mixtures \n 9. BIRCH")
        #st.help('Hoho')
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
        st.markdown("### Resolution and Gain")
        #st.info('Information about Gain and Resolution', icon="ℹ️")
        with st.expander("""ℹ️ More Information"""):
            st.write("MapperPlus takes two parameters as input: resolution and gain. The resolution determines the number of hypercubes used in the cover. The resolution can be viewd as “binning” the range of the lens. The gain determines the extent to which the hypercubes overlap , in turn determining the connectivity of the points in the dataset. The higher the gain, the more points will be shared by multiple pullback sets.")
        cols = st.columns((1, 1))
        resolution = cols[0].number_input('Resolution',min_value=1, max_value=100,step=1,value=8)
        gain = cols[1].number_input('Gain',min_value=0.0000001, max_value=0.9999999,value=0.6)
        #new_method=st.checkbox('Use new method',False)
        with st.form(key="my_form"):

            run=st.form_submit_button(label="Run Mapper Plus")
if run:
    #st.stop()
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
    #model=mapper_plus(new_method=new_method)
    model=mapper_plus()

    st.write("....... Running")
    st.write("....... Finding Mapper Graph")
    model.get_mapper_graph(lens,X,cover=cover,clusterer=clusterer,)
    st.write("....... Finding overlapping clusters")
    model.get_overlapping_clusters()
    st.write("##### "+str(len(model.overlapping_clusters))+" overlapping clusters found")
    model.get_non_overlapping_clusters(new_method=1)
    overlap_str=''
    for comms in range(len(model.overlapping_clusters)):
        if comms>0:
            overlap_str+='\n'
        overlap_str+='Cluster '+str(comms)+','
        with st.expander("Cluster "+str(comms)):
            st.write(str(model.overlapping_clusters[comms])[1:-1])
        overlap_str+=str(model.overlapping_clusters[comms])[1:-1]
    #st.download_button('Download Overlapping Clusters', overlap_str,file_name='overlapping_clusters_'+file_name)
    csvbutton=download_button(overlap_str, 'overlapping_clusters_'+file_name, 'Download Overlapping Clusters')
    #st.write(csvbutton)
    st.write("....... Finding disjoint clusters")
    st.write("##### "+str(len(model.non_overlapping_clusters))+" disjoint clusters found")
    disjoint_str=''
    for comms in range(len(model.non_overlapping_clusters)):
        if comms>0:
            disjoint_str+='\n'
        disjoint_str+='Cluster '+str(comms)+','
        with st.expander("Cluster "+str(comms)):
            st.write(str(model.non_overlapping_clusters[comms])[1:-1])
        disjoint_str+=str(model.non_overlapping_clusters[comms])[1:-1]
    download_button( disjoint_str,'disjoint_clusters_'+file_name,'Download Disjoint Clusters')
    outliers = np.array(range(len(model.comm_id)))[model.comm_id==-1]
    if len(outliers)>0:
        if len(outliers)==1:
            st.write("We found 1 outlier")
        else:    
            st.write("We found "+str(len(outliers))+" outliers")
        with st.expander("Outliers"):
            st.write(str(list(outliers))[1:-1])
