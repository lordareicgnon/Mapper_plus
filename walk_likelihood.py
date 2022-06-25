import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.decomposition import NMF

class walk_likelihood:
    def __init__(self, X):
        self.X=X
        self.N=X.shape[0]                   #size of the network
        self.w=X.dot(np.ones(self.N))       #outward rate of each node

    def WLA(self,U=None,init='NMF',m=None,l_max=8,max_iter_WLA=100,thr_WLA=0.99,eps=0.00000001,WLCF_call=0):
        if U is None:                       #initializing U
            self.initialize_WLA(init,m)
        else:
            self.U=U
            self.comm_id=np.argmax(U,axis=1)
            self.m=U.shape[1]
        comm_id_prev=self.comm_id

        for iter in range(max_iter_WLA):
            nz_values=sum(self.U)>0                 #removes a coommunity if it contains no nodes
            if (np.prod(nz_values)==0):
                self.U=self.U[:,nz_values]
                self.m=len(self.U[0,:])
                #print(self.m)
            dV=self.X.dot(self.U).astype(float)     #Step 1 of pseudocode
            V=dV
            for i in range(l_max-1):
                dV=self.X.dot(dV/self.w[:,None])
                V=V+dV
            Q=V.T.dot(self.U)/self.w.dot(self.U)    #Step 2 of pseudocode
            g=1/np.diagonal(Q)                      #Step 3 of pseudocode
            g[:]=1
            log_Q=np.log(Q+eps*(Q==0))
            F=np.dot(V,log_Q*g[:,None])-np.outer(self.w,sum(Q*g[:,None]))
            self.comm_id=np.argmax(F,axis=1)        #Step 4 of pseudocode
            #self.comm_id=np.argmin(F,axis=1)        #Step 4 of pseudocode
            self.U=np.zeros((self.N,self.m))
            self.U[range(self.N),self.comm_id]=1
            if nmi(self.comm_id,comm_id_prev)>thr_WLA:  #Step 5 of pseudocode (convergence critera)
                break
            comm_id_prev=self.comm_id               #Step 6 of pseudocode

        nz_values=sum(self.U)>0
        if (np.prod(nz_values)==0):
            self.U=self.U[:,nz_values]
            self.m=U.shape[1]
        self.find_modularity()
        #print(self.modularity)
        #if not WLCF_call:
        #    self.find_communities()

    def WLCF(self,max_iter_WLCF=20,U=None,thr_WLCF=0.99,bifuraction_type='random',modularity_tolerance=0.1,**WLA_params):
        if U is None:
            self.U=np.ones((self.N,1)) #initializing U with the whole as a single community
            self.m=1
        else:
            self.U=U
            self.m=U.shape[1]
        self.comm_id=np.argmax(self.U,axis=1)
        self.active_comms=np.array(range(self.m))
        self.inactive_comms=[]
        self.modularity=-1
        for iter in range(max_iter_WLCF):
            self.U_prev=self.U.copy()
            self.comm_id_prev=self.comm_id.copy()
            self.m_prev=self.m
            self.modularity_prev=self.modularity
            self.bifuraction(bifuraction_type) #bifuracting each community into two (Fig 1 I)
            merge=1
            while(merge):
                self.WLA(U=self.U, WLCF_call=1,**WLA_params) #Fig 1 II
                merge=self.merge_communities()  #Fig 1 III
            self.find_active_comms()            #Elimination of spurious bifurcation-merge cycles
            if ((nmi(self.comm_id_prev,self.comm_id)>thr_WLCF) and self.m_prev==self.m) or (len(self.active_comms)==0) or ((self.modularity_prev-self.modularity)>modularity_tolerance):
                break                           #convergence criterea
        self.find_communities()

    def WLM(self,max_iter_WLCF=20,U=None,thr_WLCF=0.99,bifuraction_type='random',modularity_tolerance=0.1,**WLA_params):
        if U is None:
            self.U=np.zeros((self.N,self.N)) #initializing U with the whole as a single community
            np.fill_diagonal(self.U,1)
            self.m=self.N
        else:
            self.U=U
            self.m=U.shape[1]
        self.comm_id=np.argmax(self.U,axis=1)
        merge=1
        while(merge):
            self.WLA(U=self.U, WLCF_call=1,**WLA_params)
            merge=self.merge_communities()
        self.find_communities()

    def find_communities(self):
        self.communities={}
        Ud=(self.U==1)*np.array(range(self.N))[:,None]
        for i in range(self.m):
            self.communities['Community '+str(i)]=list(Ud[:,i][self.U[:,i]==1])

    def initialize_WLA(self,init,m):
        if init=='random':
            self.comm_id=np.random.randint(m,size=self.N)
        elif init=='NMF':
            model = NMF( n_components=m,init='nndsvda',solver='mu')
            self.comm_id=np.argmax(model.fit_transform(self.X),axis=1)
        self.U=np.zeros((self.N,m))
        self.m=m
        self.U[range(self.N),self.comm_id]=1

    def find_modularity(self):
        Wtot=np.sum(self.w)
        wtots=np.dot(np.transpose(self.w),self.U)
        e_ii=np.sum(self.U*self.X.dot(self.U))/Wtot
        a_ii=wtots/Wtot
        self.modularity= (np.sum(e_ii)-np.sum(a_ii*a_ii))


    def merge_communities(self):
        W=np.dot(self.w.T,self.U)
        M=np.dot(self.U.T,self.X.dot(self.U))-np.outer(W,W)/sum(W)
        np.fill_diagonal(M,0)
        merge=(M>0).any()
        #merge=(M<0).any()
        self.m=self.U.shape[1]
        if merge:
            amx=np.argmax(M)
            #amx=np.argmin(M)
            i=int(amx/self.m)
            j=amx-i*self.m
            self.U[:,i]+=self.U[:,j]
            self.m=self.U.shape[1]
            self.U=self.U[:,np.array(list(range(j))+list(range(j+1,self.m)))]
        self.m=self.U.shape[1]
        return merge

    def bifuraction(self,bifuraction_type):
        if bifuraction_type=='random':
            U2=self.U[:,self.active_comms]*np.random.randint(2,size=self.N)[:,None]
        elif bifuraction_type=='NMF':
            U2=np.zeros((self.N,len(self.active_comms)))
            i=0
            for c in self.active_comms:
                X2=self.X[self.U[:,c]==1,:]
                model = NMF( n_components=2,init='nndsvda',solver='mu')
                U2[self.U[:,c]==1,i]=np.argmax(model.fit_transform(X2),axis=1)
                i+=1
        self.U[:,self.active_comms]=self.U[:,self.active_comms]-(U2)
        self.U=np.concatenate((self.U,U2),axis=1)

    def find_active_comms(self, thr=0.01):
        Ks=2*np.dot(self.U_prev.T,self.U)/(np.outer(sum(self.U_prev),np.ones(self.m))+np.outer(np.ones(self.m_prev),sum(self.U)))
        nac=sum(Ks>(1-thr))==0
        self.active_comms = np.array(range(self.m))[nac]
        self.inactive_comms=np.array(range(self.m))[nac==0]
