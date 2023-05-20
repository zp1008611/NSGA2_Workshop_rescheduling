import numpy as np
import random
import pandas as pd
import time

class nsga_II():
	def __init__(self,flag,generation,popsize,to,oh,work,order_num):
		self.flag = flag                    # 如果为1就是重调度，为0就是普通调度
		self.generation=generation                  #迭代次数
		self.popsize = popsize                      # 种群规模
		self.to=to
		self.oh=oh
		self.work=work
		self.order_num=order_num
	def mac_cross(self,WMT1,WMT2,gennow):  #机器均匀交叉
		ccount = np.zeros((1,self.order_num),dtype=np.int)
		varp = 1-(gennow/self.generation)
		WMT2lis = []
		for job in range(self.order_num):
			WMT2lis.append(WMT2[WMT2['W2']==job])
		varindex = []
		# 确定哪些位点需要变异
		if(varp>0.5):
			varindex = random.sample(range(WMT1.shape[0]),int(WMT1.shape[0]*0.5*varp))
		for i in WMT1.index:
			rannum = np.random.randint(0,2,1)[0]
			w = int(WMT1.loc[i,'W1'])
			# 交叉
			if(rannum==1):
				m = WMT1.loc[i,'M1']
				t = WMT1.loc[i,'T1']
				# 两个个体的同一订单的同一道工序的机器进行交换
				crossindex = WMT2lis[w].iloc[[ccount[0][w]]].index[0]
				WMT1.loc[i,'M1'] = WMT2.loc[crossindex,'M2']
				WMT1.loc[i,'T1'] = WMT2.loc[crossindex,'T2']
				WMT2.loc[crossindex,'M2'] = m
				WMT2.loc[crossindex,'T2'] = t
			# 变异
			if( i in varindex):
				if np.random.rand()>0.5:     			#选取最小加工时间机器     
					minM=min(self.to.info[w][ccount[0][w]].items(), key=lambda x: x[1])[0]
					WMT1.loc[i,'M1'] = minM
					WMT1.loc[i,'T1'] = self.to.info[w][ccount[0,w]][minM]
				else:										#否则随机挑选机器								 
					n_machine = list(self.to.info[w][ccount[0,w]].keys())
					n_time = list(self.to.info[w][ccount[0,w]].values())
					index=np.random.randint(0,len(n_machine),1)
					WMT1.loc[i,'M1'] = n_machine[index[0]]
					WMT1.loc[i,'T1'] = n_time[index[0]]

			ccount[0,w] = ccount[0][w] + 1
		W1,m1,t1,W2,m2,t2=np.array(WMT1['W1']).reshape(1,-1),np.array(WMT1['M1']).reshape(1,-1),np.array(WMT1['T1']).reshape(1,-1),np.array(WMT2['W2']).reshape(1,-1),np.array(WMT2['M2']).reshape(1,-1),np.array(WMT2['T2']).reshape(1,-1)
		return W1,m1,t1,W2,m2,t2

	def job_cross(self,WMT1,WMT2):
		num=list(range(self.order_num))
		np.random.shuffle(num)
		index=np.random.randint(0,len(num),1)[0]
		jpb_set1=num[:index+1]                  #固定不变的工件
		jpb_set2=num[index+1:]                  #按顺序读取的工件
		p1InS1 = WMT1[WMT1["W1"].isin(jpb_set1)]
		p2InS2 = WMT2[WMT2["W2"].isin(jpb_set2)]
		p2InS1 = WMT2[WMT2["W2"].isin(jpb_set1)]
		p1InS2 = WMT1[WMT1["W1"].isin(jpb_set2)]
		C1,C2 = [],[]
		for i in range(WMT1.shape[0]):
			if(i in p1InS1.index):
				if(p1InS1.shape[0]>0):
					C1.append(np.array(p1InS1.loc[i,:]))
			else:
				if(p2InS2.shape[0]>0):
					C1.append(np.array(p2InS2.iloc[0,:]))
					p2InS2 = p2InS2.iloc[1:,:]
			if(i in p2InS1.index):
				if(p2InS1.shape[0]>0):
					C2.append(np.array(p2InS1.loc[i,:]))
			else:
				if(p1InS2.shape[0]>0):
					C2.append(np.array(p1InS2.iloc[0,:]))
					p1InS2 = p1InS2.iloc[1:,:]
		C1 = pd.DataFrame(C1)
		C1.columns=['W1','M1','T1']
		C2 = pd.DataFrame(C2)
		C2.columns=['W2','M2','T2']
		return C1,C2

	def cross_var(self,WMT1,WMT2,gennow):
		startime = time.time()
		C1,C2 = self.job_cross(WMT1,WMT2)
		endtime = time.time()
		# print('工序交叉用时：',endtime-startime)
		startime = time.time()
		W1,m1,t1,W2,m2,t2 = self.mac_cross(C1,C2,gennow)
		endtime = time.time()
		# print('机器交叉用时：',endtime-startime)
		return W1,m1,t1,W2,m2,t2

	def nsga_total(self):
		answer=[]
		fit_every=[[],[],[],[]]
		# self.popsize = 100, self.work = 55
		# work_job1:[100,55],work_M1:[100,55],work_T1:[100,55]
		work_job1,work_M1,work_T1=np.zeros((self.popsize,len(self.work))),np.zeros((self.popsize,len(self.work))),np.zeros((self.popsize,len(self.work)))
		# work_job:[100,55],work_M:[100,55],work_T:[100,55]
		work_job,work_M,work_T=np.zeros((self.popsize,len(self.work))),np.zeros((self.popsize,len(self.work))),np.zeros((self.popsize,len(self.work)))
		startime = time.time()
		for gen in range(self.generation): # self.generation=50
			if(gen<1):                      #第一代生成多个可行的工序编码，机器编码，时间编码
				for i in range(self.popsize):
					# job:[1,55],machine:[1,55],machine_time:[1,55]
					job,machine,machine_time=self.to.encoding()
					
					if(self.flag==0):
						C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding1(job,machine,machine_time)
						answer.append([C_finish,Twork,E_all])
					else:
						C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding2(job,machine,machine_time)
						answer.append([C_finish,Twork,E_all])
					work_job[i],work_M[i],work_T[i]=job[0],machine[0],machine_time[0]
				
				endtime = time.time()
				# print('编码用时：',endtime-startime)
				_,crowder=self.oh.dis(answer)    #计算分层，拥挤度，种群排序结果
				
			# 建立精英库
			index_sort=crowder
			work_job,work_M,work_T=work_job[index_sort][0:self.popsize],work_M[index_sort][0:self.popsize],work_T[index_sort][0:self.popsize] # (100,55)
			answer=np.array(answer)[index_sort][0:self.popsize].tolist() # len(answer)=100
			
			answer1=[]
			startime = time.time()
			for i in range(0,self.popsize,2):   
				W1,M1,T1=work_job[i:i+1],work_M[i:i+1],work_T[i:i+1]
				W2,M2,T2=work_job[i+1:i+2],work_M[i+1:i+2],work_T[i+1:i+2]
				WMT1 = pd.DataFrame([W1[0],M1[0],T1[0]]).T # 父代1
				WMT1.columns=['W1','M1','T1']
				WMT2 = pd.DataFrame([W2[0],M2[0],T2[0]]).T # 父代2
				WMT2.columns=['W2','M2','T2']
				
				C1,m1,t1,C2,m2,t2=self.cross_var(WMT1,WMT2,gen) # 交叉与变异
				
				if(self.flag==0):
					C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding1(C1,m1,t1)
					answer1.append([C_finish,Twork,E_all])
					C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding1(C2,m2,t2)
					answer1.append([C_finish,Twork,E_all])
				else:
					C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding2(C1,m1,t1)
					answer1.append([C_finish,Twork,E_all])
					C_finish,Twork,E_all,_,_,_,_,_=self.to.decoding2(C2,m2,t2)
					answer1.append([C_finish,Twork,E_all])
				work_job1[i], work_M1[i], work_T1[i] = C1[0], m1[0], t1[0]
				work_job1[i+1], work_M1[i+1], work_T1[i+1] = C2[0], m2[0], t2[0]
				
			endtime = time.time()
			# print('交叉变异：',endtime-startime)
			work_job,work_M,work_T=np.vstack((work_job,work_job1)),np.vstack((work_M,work_M1)),np.vstack((work_T,work_T1)) # (200,55)
			answer = answer + answer1
			startime = time.time()
			front,crowder=self.oh.dis(answer)
			endtime = time.time()
			# print('选择：',endtime-startime)
			

			signal=front[0]
			pareto=np.array(answer)[signal]
			x=[pareto[i][0] for i in range(len(pareto))] # len(x) = b
			y=[pareto[i][1] for i in range(len(pareto))] # len(y) = b
			z=[pareto[i][2] for i in range(len(pareto))] # len(z) = b
			fit_every[3].append(gen)
			fit_every[0].append([min(x),sum(x)/len(x),max(x)])
			fit_every[1].append([min(y),sum(y)/len(y),max(y)])
			fit_every[2].append([min(z),sum(z)/len(z),max(z)])
			
			# 本文选择的最优解是pareto解中距离远点最近的点，也可以取某个目标达到最值的点
			minpareto = np.argmin((pareto*pareto).sum(axis=1))
			pareto = pareto[minpareto]
			bestindex = [signal[minpareto]]
			best_job,best_machine,best_time=work_job[bestindex],work_M[bestindex],work_T[bestindex]
			
			print('算法迭代到了第%.0f次'%(gen+1))
			
		return best_job,best_machine,best_time,fit_every  


