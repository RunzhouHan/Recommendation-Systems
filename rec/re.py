import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from itertools import combinations
import math
from numpy import linalg as la
import copy
import scipy.io as sio
import heapq
import random
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
a = []
b = []

def ExtractFullMat(num=1100):
	ratings=pd.read_table('ratings.dat',sep='::',header=None,engine='python')
	rnames=['user_id','movie_id','rating','timestamps']
	ratings.columns=rnames
	#data=pd.merge(pd.merge(ratings,users),movies)

	data=np.array(ratings)
	data=data[:,0:3]

	row=data[:,0].T
	col=data[:,1].T
	rate=data[:,2].T
	mat=coo_matrix((rate,(row,col)),shape=(6041,3953))
	matr=mat.todense()

	uid=[]
	for i in range(6041):
		if (np.nonzero(matr[i]))[0].shape[0]>num:
			#print(i)
			uid.append(i)

	#print(len(uid))

	tp=matr.tolist()

	tpi=[]
	for i in uid:
		tpi.append(tp[i])
	#print(len(tpi))
	ldata=np.array(tpi)
	fi=[]

	mid=[]

	for i in range(3953):
		hang=ldata[:,i].T
		if (np.nonzero(hang))[0].shape[0]==len(uid):
			#print(i)
			mid.append(i)
			fi.append(hang.tolist())

	yu=np.array(fi,dtype=float).T
	return yu


def Scosine(A,B):
	'''return the cosine similarity of the WHOLE vectors'''
	if la.norm(A)*la.norm(B)==0:
		if la.norm(A)==0 and la.norm(B)==0:
			co=1
		else:
			co=0

	if la.norm(A)*la.norm(B)!=0:
		co=np.dot(A,B)/(la.norm(A)*la.norm(B))
	return co

def Snorm(A,B):
	pass

def tong(A,B):
	fi=0
	if A>=0:
		if B>=0:
			fi=1
	if A<=0:
		if B<=0:
			fi=1
	return fi


class Data():
	def __init__(self,mat):
		self.mat=mat
		self.meann=[]
		self.NNindex=[]
		self.Smat=[]
		self.CF=False
		self.SVD=False
		self.PMF=False
		self.CFmat=[]
		self.SVDmat=[]
		self.PMFmat=[]
		self.original=copy.deepcopy(mat)
		self.SVDcomp=[0,-10]
		self.PMFcomp=[0,-10]
		self.PMFcomp2=[0,1000]


	def NullIndexGenerator(self,lack,unif=True):
		'''return null INDEX given full percentage'''
		nullnum=int(self.mat.shape[0]*self.mat.shape[1]*lack)
		random.seed(22)
		ran=[]
		while 1:
			ran.append((int(self.mat.shape[0]*random.random()),int(self.mat.shape[1]*random.random())))
			ran=list(set(ran))
			if len(ran)==nullnum:
				break
		self.nullind=ran
		for tu in self.nullind:
			self.mat[tu[0],tu[1]]=0


	def Normalize(self):
		'''centralize'''
		mat=copy.deepcopy(self.mat)
		rmt=copy.deepcopy(self.original)
		meann=[]
		stdd=[]

		for i in range(mat.shape[0]):
			rmt[i,:]=rmt[i,:]-np.mean(rmt[i,:])
			tp=0
			tn=0
			for j in range(mat.shape[1]):
				if mat[i,j]!=0:
					tp=tp+mat[i,j]
					tn=tn+1
			if tn==0:
				meann.append(0)
			else:
				meann.append(tp/tn)
		for i in range(mat.shape[0]):
			for j in range(mat.shape[1]):
				#rmt[i,j]=rmt[i,j]-meann[i]
				if (i,j) not in self.nullind:
					mat[i,j]=mat[i,j]-meann[i]
			#mat[i,:]=mat[i,:]/np.std(mat[i,:])

		self.Noriginal=rmt
		self.meann=meann
		self.mat=mat


	def Nindex(self,k=3,typee='cosine'):
		'''typee=scosine,snorm,sabs,...'''
		mat=copy.deepcopy(self.mat)
		num=mat.shape[0]

		Usernullind={}
		for tu in self.nullind:
			Usernullind[tu[0]]=[]
		for tu in self.nullind:
			Usernullind[tu[0]].append(tu)

		self.UserNullInd=Usernullind
		#print(num)
		Dmat=np.zeros([num,num])
		for i in range(num):
			for j in range(num):
				if i==j:
					Dmat[i,j]=-999999
				else:
					if typee=='cosine':
						self.simtype='cosine'
						Dmat[i,j]=Scosine(mat[i,:],mat[j,:])
					if typee=='distance':
						self.simtype='distance'
						delta=0
						sumd=0
						hang1=[]
						hang2=[]
						for lie in range(self.mat.shape[1]):
							try:
								if (i,lie) not in Usernullind[i]:
									if (j,lie) not in Usernullind[j]:
										hang1.append(mat[i,lie])
										hang2.append(mat[j,lie])
										delta=delta+1
							except KeyError:
								continue
									#sumd=sumd+(mat[i,lie]-mat[j,lie])**2
						#sumd=sumd**0.5
						hang1=np.array(hang1)
						hang2=np.array(hang2)
						if delta==0:
							Dmat[i,j]=-999999
						else:
							#dist=sumd/delta
							Dmat[i,j]=Scosine(hang1,hang2)

		#print(Dmat)
		ind=[]
		for i in range(num):
			tp=[]
			hang={}
			for j in range(num):
				hang[j]=Dmat[i][j]
			while 1:
				aa=max(hang, key=hang.get)
				tp.append(aa)
				del hang[aa]
				if len(tp)==k:
					break
			ind.append(tp)
		self.NNindex=ind
		self.Smat=Dmat


	def GetCFresult(self,LeastSim=0):
		'''give CF recovery mat to attribute'''
		#if self.simtype=='distance':
			#LeastSim=-1000
		tpmat=copy.deepcopy(self.mat)
		dic={}
		for tu in self.nullind:
			j=tu[1]
			tp=[]
			for i in range(self.mat.shape[0]):
				if i in self.NNindex[tu[0]]:
					if (i,j) not in self.nullind:
						if self.Smat[tu[0],i]>=LeastSim:
							tp.append((i,j))
			dic[tu]=tp
		for tu in self.nullind:
			if len(dic[tu])!=0:
				summ=0
				sunn=0
				for pt in dic[tu]:
					summ=summ+self.mat[pt[0],pt[1]]*self.Smat[pt[0],tu[0]]
					sunn=sunn+self.Smat[pt[0],tu[0]]
				pred=summ/sunn
				#print('Predict for '+str(tu)+':'+str(pred))
				#print('Real for '+str(tu)+':'+str(self.Noriginal[tu[0],tu[1]]))
				tpmat[tu[0],tu[1]]=pred
		self.dict=dic # record those been rated
		self.CFmat=tpmat
		#print(tpmat)
		self.CF=True
		#print(self.CFmat)


	def GetSVDresult(self,Num=5,e=0.001):
		A=copy.deepcopy(self.mat)

		while 1:

			del(self.SVDcomp[0])
			self.SVDcomp.append(A)

			ast=np.fabs(self.SVDcomp[0]-self.SVDcomp[1])<=e
			#print(ast)
			if ast.all()==True:
				break

			#print(A[7,40])
			U,sigma,VT=la.svd(A)
			tp=list(sigma)
			while 1:
				if len(tp)==self.mat.shape[1]:
					break
				tp.append(0)
			sigma=np.array(tp)

			sigma=np.diag(sigma)
			sigma=sigma[0:self.mat.shape[0],:]
			for i in range(Num,self.mat.shape[0]):
				sigma[i,i]=0
			A=U@sigma@VT

			for i in range(self.mat.shape[0]):
				for j in range(self.mat.shape[1]):
					if (i,j) not in self.nullind:
						A[i,j]=self.mat[i,j]
					if A[i,j]>4:
						A[i,j]=4
					if A[i,j]<-4:
						A[i,j]=-4
		#print(sigma)
		self.SVDmat=A
		#print(A)
		self.SVD=True


	def GetPMFresult(self,k=1,e=0.00001,alpha=0.05,c1=1e-17,c2=1e-17):
		'''icremental gradient method'''
		m=self.mat.shape[0]
		n=self.mat.shape[1]
		self.time=0

		indlist=[]
		for i in range(m):
			for j in range(n):
				if (i,j) not in self.nullind:
					indlist.append((i,j))
		#p=np.ones([m,k])*0.1
		#q=np.ones([k,n])*(-0.1)
		np.random.seed(5)
		u=np.random.rand(m,k)-0.5
		v=np.random.rand(n,k)-0.5

		t = 0
		# for r_t in range(800):
		while 1:
			#print(tt)
			t = t+1
			self.time+=1
			del(self.PMFcomp[0])
			del(self.PMFcomp2[0])
			self.PMFcomp.append(u@v.T)
			self.PMFcomp2.append(la.norm(self.PMFcomp[0]-self.PMFcomp[1],"fro"))
			# print(la.norm(self.PMFcomp[0]-self.PMFcomp[1],"fro"))
			ast=np.fabs(self.PMFcomp[0]-self.PMFcomp[1])<=e
			# print(ast)
			if ast.all()==True:
				# print(t)
				break
			#if self.PMFcomp2[1]>self.PMFcomp2[0]:
			#	break
			#if self.time>1000:
			#	break
			# tpp=u@v.T
			# cost = 0
			# for tu in indlist:
			# 	cost=cost+(self.mat[tu[0],tu[1]]-tpp[tu[0],tu[1]])**2
			# a.append(cost)
			# print(cost)

			for i in range(m):
				for j in range(n):
					if (i,j) not in self.nullind:
						err=self.mat[i,j]-np.dot(u[i],v[j])
						for r in range(k):
							gu=err*v[j][r]-c1*u[i][r]
							gv=err*u[i][r]-c2*v[j][r]
							u[i][r]+=alpha*gu
							v[j][r]+=alpha*gv

		fi=u@v.T
		#print(fi)
		for i in range(m):
			for j in range(n):
				if (i,j) not in self.nullind:
					fi[i,j]=self.mat[i,j]
				if fi[i,j]>4:
					fi[i,j]=4
				if fi[i,j]<-4:
					fi[i,j]=-4
		self.PMFmat=fi
		self.PMF=True
		#print(fi)

	# def GetPMFresult_(self,k=1,e=0.00001,alpha=0.05,c1=1e-17,c2=1e-17):
	# 	'''icremental gradient method'''
	# 	m=self.mat.shape[0]
	# 	n=self.mat.shape[1]
	# 	self.time=0

	# 	indlist=[]
	# 	for i in range(m):
	# 		for j in range(n):
	# 			if (i,j) not in self.nullind:
	# 				indlist.append((i,j))
	# 	#p=np.ones([m,k])*0.1
	# 	#q=np.ones([k,n])*(-0.1)
	# 	np.random.seed(5)
	# 	u=np.random.rand(m,k)-0.5
	# 	v=np.random.rand(n,k)-0.5

	# 	t = 0
	# 	for r_t in range(800):
	# 	# while 1:
	# 		#print(tt)
	# 		t = t+1
	# 		self.time+=1
	# 		del(self.PMFcomp[0])
	# 		del(self.PMFcomp2[0])
	# 		self.PMFcomp.append(u@v.T)
	# 		self.PMFcomp2.append(la.norm(self.PMFcomp[0]-self.PMFcomp[1],"fro"))
	# 		# print(la.norm(self.PMFcomp[0]-self.PMFcomp[1],"fro"))
	# 		ast=np.fabs(self.PMFcomp[0]-self.PMFcomp[1])<=e
	# 		# print(ast)
	# 		# if ast.all()==True:
	# 		# 	print(t)
	# 		# 	break
	# 		#if self.PMFcomp2[1]>self.PMFcomp2[0]:
	# 		#	break
	# 		#if self.time>1000:
	# 		#	break
	# 		tpp=u@v.T
	# 		cost = 0
	# 		for tu in indlist:
	# 			cost=cost+(self.mat[tu[0],tu[1]]-tpp[tu[0],tu[1]])**2
	# 		b.append(cost)
	# 		# print(cost)

	# 		for i in range(m):
	# 			for j in range(n):
	# 				if (i,j) not in self.nullind:
	# 					err=self.mat[i,j]-np.dot(u[i],v[j])
	# 					for r in range(k):
	# 						gu=err*v[j][r]-c1*u[i][r]
	# 						gv=err*u[i][r]-c2*v[j][r]
	# 						u[i][r]+=alpha*gu
	# 						v[j][r]+=alpha*gv

	# 	fi=u@v.T
	# 	#print(fi)
	# 	for i in range(m):
	# 		for j in range(n):
	# 			if (i,j) not in self.nullind:
	# 				fi[i,j]=self.mat[i,j]
	# 			if fi[i,j]>4:
	# 				fi[i,j]=4
	# 			if fi[i,j]<-4:
	# 				fi[i,j]=-4
	# 	self.PMFmat=fi
	# 	self.PMF=True
	# 	#print(fi)
		

	def GetRe(self,ReNum=3):

		# 1.SVD
		if self.SVD==True:
			SVDre={}
			SVDtp={}
			SVDlist=[]
			for tu in self.nullind:
				SVDre[tu[0]]=[]
				SVDtp[tu[0]]=[]
			for tu in self.nullind:
				SVDtp[tu[0]].append(tu)
			for user in SVDtp.keys():
				tp1={}
				for tup in SVDtp[user]:
					tp1[tup]=self.SVDmat[tup[0],tup[1]]
				while 1:
					if len(tp1.keys())==0:
						break
					aa=max(tp1, key=tp1.get)
					if self.SVDmat[aa[0],aa[1]]>0:
						SVDre[user].append(aa)
						del tp1[aa]
						if len(SVDre[user])==ReNum:
							break
					if self.SVDmat[aa[0],aa[1]]<=0:
						break
			for i in SVDre.keys():
				for tuu in SVDre[i]:
					SVDlist.append(tuu)
			#print(SVDlist)
			self.SVDreList=SVDlist

		# 2.PMF
		if self.PMF==True:
			PMFre={}
			PMFtp={}
			PMFlist=[]
			for tu in self.nullind:
				PMFre[tu[0]]=[]
				PMFtp[tu[0]]=[]
			for tu in self.nullind:
				PMFtp[tu[0]].append(tu)
			for user in PMFtp.keys():
				tp2={}
				for tup in PMFtp[user]:
					tp2[tup]=self.PMFmat[tup[0],tup[1]]
				while 1:
					if len(tp2.keys())==0:
						break
					aa=max(tp2, key=tp2.get)
					if self.PMFmat[aa[0],aa[1]]>0:
						PMFre[user].append(aa)
						del tp2[aa]
						if len(PMFre[user])==ReNum:
							break
					if self.PMFmat[aa[0],aa[1]]<=0:
						break
			for i in PMFre.keys():
				for tuu in PMFre[i]:
					PMFlist.append(tuu)
			#print(PMFlist)
			self.PMFreList=PMFlist

		# 3.CF
		if self.CF==True:
			CFre={}
			CFtp={}
			CFlist=[]
			for tu in self.dict.keys():
				CFre[tu[0]]=[]
				CFtp[tu[0]]=[]
			for tu in self.dict.keys():
				CFtp[tu[0]].append(tu)
			for user in CFtp.keys():
				tp3={}
				for tup in CFtp[user]:
					tp3[tup]=self.CFmat[tup[0],tup[1]]
				while 1:
					if len(tp3.keys())==0:
						break
					aa=max(tp3, key=tp3.get)
					if self.CFmat[aa[0],aa[1]]>0:
						CFre[user].append(aa)
						del tp3[aa]
						if len(CFre[user])==ReNum:
							break
					if self.CFmat[aa[0],aa[1]]<=0:
						break
			for i in CFre.keys():
				for tuu in CFre[i]:
					CFlist.append(tuu)
			#print(CFlist)
			self.CFreList=CFlist
			self.RE=True







	def MAE(self):
		MAE_=[]
		if self.CF==True:
			CFm=0
			CFn=0
			for tu in self.nullind:
				if len(self.dict[tu])!=0:
					CFm=CFm+math.fabs(self.Noriginal[tu[0],tu[1]]-self.CFmat[tu[0],tu[1]])**2
					CFn=CFn+1
			MAE_.append((CFm/CFn)**0.5)

		if self.SVD==True:
			SVDm=0
			SVDn=0
			for tu in self.nullind:
				SVDm=SVDm+math.fabs(self.Noriginal[tu[0],tu[1]]-self.SVDmat[tu[0],tu[1]])**2
				SVDn=SVDn+1
			MAE_.append((SVDm/SVDn)**0.5)


		if self.PMF==True:
			PMFm=0
			PMFn=0
			for tu in self.nullind:
				PMFm=PMFm+math.fabs(self.Noriginal[tu[0],tu[1]]-self.PMFmat[tu[0],tu[1]])**2
				PMFn=PMFn+1
			MAE_.append((PMFm/PMFn)**0.5)
					
		AVGm=0
		AVGn=0
		for tu in self.nullind:
			AVGm=AVGm+math.fabs(self.Noriginal[tu[0],tu[1]]-self.mat[tu[0],tu[1]])**2
			AVGn=AVGn+1
		MAE_.append((AVGm/AVGn)**0.5)


		self.MAE_=MAE_
		# print(MAE_)
		return MAE_


	def FSC(self):
		FSC={'FSC':'','CF':[],'SVD':[],'PMF':[]}
		if self.CF==True:
			CFm=0
			CFn=0
			for tu in self.nullind:
				if len(self.dict[tu])!=0:
				#print(self.CFmat[tu[0],tu[1]])
					if tong(self.Noriginal[tu[0],tu[1]],self.CFmat[tu[0],tu[1]])==1:
						CFm=CFm+1
					CFn=CFn+1
			FSC['CF']=CFm/CFn
		if self.SVD==True:
			SVDm=0
			SVDn=0
			for tu in self.SVDreList:
				if tong(self.Noriginal[tu[0],tu[1]],self.SVDmat[tu[0],tu[1]])==1:
					SVDm=SVDm+1
				SVDn=SVDn+1
			FSC['SVD']=SVDm/SVDn
		self.FSC=FSC
		print(FSC)


	def CVG(self):
		CVG={'CVG':'','CF':[],'SVD':[],'PMF':[]}
		total=self.mat.shape[1]
		if self.CF==True:
			tp=[]
			for tu in self.CFreList:
				tp.append(tu[1])
			part=list(set(tp))
			CVG['CF']=len(part)/total

		if self.SVD==True:
			tp=[]
			for tu in self.SVDreList:
				tp.append(tu[1])
			part=list(set(tp))
			CVG['SVD']=len(part)/total

		if self.PMF==True:
			tp=[]
			for tu in self.PMFreList:
				tp.append(tu[1])
			part=list(set(tp))
			CVG['PMF']=len(part)/total
		self.CVG=CVG
		print(CVG)


	#ran=list(set(ran))
nullrate=[]
MAE=[]
MAE1=[]

'''for k in range(1,10):
	
	s=0.01*k
	print(s)'''

data = sio.loadmat('yu.mat')
yu=data['yuan']
#yu=ExtractFullMat()
#print(yu.shape)
#yu=yu.T
ob=Data(yu)
misrate = 0.01
_MAE_ = []
while misrate<=0.55:
	ob.NullIndexGenerator(misrate)
	ob.Normalize()
	ob.Nindex(3,'cosine')
#print(ob.nullind)

	ob.GetCFresult(0)
	ob.GetSVDresult(2,1)
	ob.GetPMFresult(2,0.0001,0.01) # k e alpha
	# ob.GetPMFresult_(1,0.0001,0.01)
	ob.GetRe(1)
	performance = ob.MAE()
	performance.append(misrate)
	_MAE_.append(performance)
	print(performance)
	misrate = misrate+0.01

#ob.CVG()

#ob.FSC()

'''	nullrate.append(k)
	MAE.append(ob.MAE['SVD'])
	MAE1.append(ob.MAE['PMF'])'''


#nullrate.append(s)



#FSC.append(ob.FSC['CF'])


'''MAE2=[]
FSC2=[]

for k in range(5,90):
	
	s=0.01*k
	print(s)
	data = sio.loadmat('yu.mat')
	yu=data['yuan']
	#print(yu.shape)
	#yu=yu.T
	ob=Data(yu)
	ob.NullIndexGenerator(s)

	ob.Normalize()
	ob.Nindex(5,'distance')

	ob.GetCFresult(0)
	#ob.GetSVDresult()
	#ob.GetPMFresult()
fsfd
	ob.GetRe(2)
	ob.MAE()

	#ob.CVG()
	ob.FSC()
	MAE2.append(ob.MAE['CF'])
	FSC2.append(ob.FSC['CF'])'''

CF = []
SVD = []
PMF = []
AVG = []
for i in _MAE_:
	CF.append(i[0])
	SVD.append(i[1])
	PMF.append(i[2])
	AVG.append(i[3])

# print("CF:",len(CF))
# print("SVD:",len(SVD))
# print("PMF:",len(PMF))
# print("AVG:",len(AVG))

nullrate = np.arange(0,0.54,0.01)
# nullrate_ = np.arange(0,184,1)
plt.figure() 
# plt.plot(nullrate,a,color='blue',label='k=2')
# plt.plot(nullrate,b,color='green',label='k=1')
plt.plot(nullrate,CF,color='blue',label='CF')
plt.plot(nullrate,SVD,color='green',label='SVD')
plt.plot(nullrate,PMF,color='black',label='PMF')
plt.plot(nullrate,AVG,color='red',linestyle='-.',label='AVG')  

# plt.xlabel("Iteration Time")  
# plt.ylabel("Training cost")  
# plt.title("")
plt.legend()
plt.grid()
plt.show()


# print(a)


	


