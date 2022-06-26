import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
from sklearn.decomposition import PCA
import walk_likelihood as wl
import importlib
mapper = km.KeplerMapper(verbose=1)

class mapper_plus():
    def __init__(self):
        self.mapper_graph_found=0
        self.overlapping_clusters_found=0

    def get_mapper_graph(self,lens,data,**kepler_mapper_args):
        self.mapper_graph = mapper.map(lens,data,**kepler_mapper_args)
        self.data=data
        self.N=len(self.data)
        self.Ng=len(self.mapper_graph['nodes'])
        self.A=np.zeros((self.Ng,self.Ng))
        i=-1
        self.cl_list=[]
        for n1 in self.mapper_graph['nodes']:
            self.cl_list.append(n1)
            i+=1
            s1=set(self.mapper_graph['nodes'][n1])
            j=-1
            for n2 in self.mapper_graph['nodes']:
                j+=1
                s2=set(self.mapper_graph['nodes'][n2])
                ins=len(s1.intersection(s2))
                self.A[i,j]=ins
        self.mapper_graph_found=1

    def get_overlapping_clusters(self,**WLCF_args):
        if self.mapper_graph_found==1:
            model=wl.walk_likelihood(self.A)
            model.WLM(**WLCF_args)
            ket=model.U
            cluster_members=[]
            Ng=len(self.mapper_graph['nodes'])
            for i in range(model.m):
                cl_i=[]
                for j in range(Ng):
                    if ket[j,i]==1:
                        cl_i.append(self.cl_list[j])
                cluster_members.append(cl_i)
            self.overlapping_clusters=[]
            self.m=model.m
            for cm in cluster_members:
                cl_p=set()
                for i in cm:
                    cl_p=cl_p.union(self.mapper_graph['nodes'][i])
                self.overlapping_clusters.append(list(cl_p))
            self.overlapping_clusters_found=1
            print('We found '+str(self.m)+' overlapping clusters')
            return model
        else:
            print('First compute the mapper_graph')

    def get_non_overlapping_clusters(self,**WLA_args):
        if self.overlapping_clusters_found==1:
            self.X=np.zeros((self.N,self.N))
            a=set(range(self.N))
            b=set()
            for n in self.mapper_graph['nodes']:
                for i in self.mapper_graph['nodes'][n]:
                    b.add(i)
                #print(i)
                    for j in self.mapper_graph['nodes'][n]:
                        self.X[i,j]+=1
            self.X[list(a-b),list(a-b)]=1

            U=np.zeros((self.N,self.m+len(a-b)))
            if (len(a-b)>0):
                U[list(a-b),self.m+np.array(range(len(a-b)))]=1
            mn=0
            for cm in self.overlapping_clusters:
                for i in cm:
                    U[i,mn]=1
                mn+=1
            model=wl.walk_likelihood(self.X)
            model.WLA(U=U,**WLA_args)
            self.non_overlapping_clusters=[]
            index=np.array(range(model.N))
            self.comm_id=model.comm_id
            for i in range(self.m):
                self.non_overlapping_clusters.append(list(index[model.comm_id==i]))
            print('We found '+str(self.m)+' non-overlapping clusters')
            return model
        else:
            print('First find overlapping clusters')
