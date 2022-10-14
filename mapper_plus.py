import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
from sklearn.decomposition import PCA
import walk_likelihood as wl
import importlib
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
mapper = km.KeplerMapper(verbose=1)

class easy_dot():
    def __init__(self,a):
        self.a=a
        self.shape=[a.shape[0],a.shape[0]]

    def dot(self,b):
        return self.a.dot(self.a.T.dot(b))

class mapper_plus():
    def __init__(self,verbose=1):
        self.mapper_graph_found=0
        self.overlapping_clusters_found=0

    def get_mapper_graph(self,lens,data,**kepler_mapper_args):
        self.lens=lens
        self.mapper_graph = mapper.map(lens,data,**kepler_mapper_args)
        self.data=data
        self.N=len(self.data)
        self.Ng=len(self.mapper_graph['nodes'])
        i=0
        self.cl_list=[]
        self.hyper_cube_M=np.zeros((self.N,0))
        self.hyper_cube_elements=[]
        self.mapper_nodes=np.zeros((self.N,self.Ng))
        hpcs={}
        hpc_no=0
        for n1 in self.mapper_graph['nodes']:
            self.mapper_nodes[self.mapper_graph['nodes'][n1],i]=1
            i+=1
            self.cl_list.append(n1)
            s1=set(self.mapper_graph['nodes'][n1])

            hpc=n1.split('_')[0]
            if hpc not in hpcs:
                hpcs[hpc]=hpc_no
                hpc_no+=1
                self.hyper_cube_M=np.concatenate((self.hyper_cube_M,np.zeros((self.N,1))),axis=1)
                self.hyper_cube_elements.append([])

            self.hyper_cube_elements[hpcs[hpc]]+=list(s1)
            self.hyper_cube_M[list(s1),hpcs[hpc]]=1

        self.hyper_cube_M=self.hyper_cube_M/np.sqrt(sum(self.hyper_cube_M))
        self.mapper_graph_found=1

    def get_overlapping_clusters(self,**WLCF_args):
        if self.mapper_graph_found==1:
            self.A=self.mapper_nodes.T.dot(self.mapper_nodes)
            model=wl.walk_likelihood(self.A)
            model.WLM(**WLCF_args)
            self.U=self.mapper_nodes.dot(model.U)
            self.overlapping_clusters=[list(np.where(self.U[:,i]>0)[0]) for i in range(model.m)]
            self.overlapping_clusters_found=1
            self.m=model.m
            if verbose:
                print('We found '+str(self.m)+' overlapping clusters \n',end='\r')
            return model
        else:
            print('First compute the mapper_graph')

    def get_non_overlapping_clusters(self,new_method=1,**WLA_args):
        if self.overlapping_clusters_found==1:
            diff=np.where(np.sum(self.U,axis=1)==0)[0]

            Xa=self.mapper_nodes
            if (len(diff)>0):
                Xb=np.zeros(self.N)
                Xb[diff]=1
                Xa=np.concatenate((Xa,Xb[:,None]),axis=1)
                self.U=np.concatenate((self.U,Xb[:,None]),axis=1)
            if new_method==1:
                Xa=np.concatenate((Xa,self.hyper_cube_M),axis=1)
            self.X=easy_dot(Xa)

            model=wl.walk_likelihood(self.X)
            model.WLA(U=self.U,**WLA_args)
            self.non_overlapping_clusters=[]
            index=np.array(range(model.N))
            self.comm_id=model.comm_id
            self.m=model.m-(len(diff)>0)
            self.comm_id[self.comm_id==self.m]=-1
            for i in range(self.m):

                self.non_overlapping_clusters.append(list(index[model.comm_id==i]))
            if verbose:
                print('We found '+str(self.m)+' non-overlapping clusters \n',end='\r')
            if len(diff)>0:
                self.outliers=list(diff)
                print('We found '+str(len(diff))+' outliers',end='\r')
            return model
        else:
            if verbose:
                print('First find overlapping clusters')
