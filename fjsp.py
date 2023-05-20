import numpy as np
import pandas as pd
from data_solve import data_deal
import random 
import matplotlib.pyplot as plt 
#plt.rcParams['font.sans-serif'] = ['STSong'] 
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class FJSP():
	def __init__(self,flag,filename,order_num,machine_num,pi,parm_data,jobing=None, insertpoint=None, jobdic=None, nojob=None):
		self.flag = flag                    # 如果为1就是重调度，为0就是普通调度		
		self.order_num=order_num     			#订单数
		self.machine_num=machine_num		#机器数
		self.pi=pi  				#随机挑选机器的概率
		self.info,self.delivery_time,self.work=parm_data[0],parm_data[1],parm_data[2]
		self.jobing = jobing
		self.insertpoint = insertpoint
		self.jobdic = jobdic
		self.nojob = nojob
		data = pd.read_excel(filename,index_col=0)
		self.p1 = data.loc['负载'].values.tolist()
		self.p2 = data.loc['功率'].values.tolist()
		
	def axis(self):
		index=['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12',
		'M13','M14','M15','M16','M17','M18','M19','M20']
		scale_ls,index_ls=[],[]   
		for i in range(self.machine_num):
			scale_ls.append(i+1)
			index_ls.append(index[i])
		return index_ls,scale_ls  #返回坐标轴信息，按照工件数返回，最多画20个机器，需要在后面添加
	def encoding(self):
		job=np.copy(self.work) # 工序编码，[55,]

		np.random.shuffle(job) # 打乱顺序
		# [8 0 4 6 4 4 2 5 2 0 7 9 1 5 5 3 6 5 1 8 2 9 7 3 8 9 2 0 4 1 3 6 4 9 6 6 0 7 5 8 3 8 0 2 9 1 7 5 8 0 1 3 9 4 7]
		job=np.array(job).reshape(1,len(self.work)) # 维度：[1,55] 
		ccount=np.zeros((1,self.order_num),dtype=np.int) # [1,10]
		machine=np.ones((1,job.shape[1])) # [1,55]
		machine_time=np.ones((1,job.shape[1]))    #初始化矩阵 
		for i in range(job.shape[1]): 
			oper=int(job[0,i])  
			if np.random.rand()>self.pi:     			#选取最小加工时间机器    
				minM=min(self.info[oper][ccount[0,oper]].items(), key=lambda x: x[1])[0]
				machine[0,i] = minM
				machine_time[0,i] = self.info[oper][ccount[0,oper]][minM]
			else:										#否则随机挑选机器								 
				n_machine = list(self.info[oper][ccount[0,oper]].keys())
				n_time = list(self.info[oper][ccount[0,oper]].values())
				index=np.random.randint(0,len(n_machine),1)
				machine[0,i]=n_machine[index[0]]
				machine_time[0,i]=n_time[index[0]]
			ccount[0,oper]+=1
		return job,machine,machine_time
		
	def decoding1(self,job,machine,machine_time):
		jobtime=np.zeros((1,self.order_num))  # (1,10)      
		tmm=np.zeros((1,self.machine_num))  # (1,6) 			
		tmmw=np.zeros((1,self.machine_num))	# (1,6)		
		startime=0
		# list_M存放机器编号，list_S存放工序开始时间，list_W存放对应工序在对应机器上的加工时间
		list_M,list_S,list_W=[],[],[]
		for i in range(job.shape[1]): 
			svg,sig=int(job[0,i]),int(machine[0,i])-1  
			if(jobtime[0,svg]>0): # 如果不是订单的第一道工序								
				startime=max(jobtime[0,svg],tmm[0,sig])   	
				tmm[0,sig]=startime+machine_time[0,i]
				jobtime[0,svg]=startime+machine_time[0,i]
			if(jobtime[0,svg]==0): # 如果是订单的第一道工序							
				startime=tmm[0,sig] 
				tmm[0,sig]=startime+machine_time[0,i]
				jobtime[0,svg]=startime+machine_time[0,i] 

			tmmw[0,sig]+=machine_time[0,i] 
			list_M.append(machine[0,i]) 
			list_S.append(startime)
			list_W.append(machine_time[0,i])      
		tmax=np.argmax(tmm[0])+1		#结束最晚的机器
		C_finish = 0
		for i in range(len(jobtime[0])):
			if(jobtime[0,i]<=self.delivery_time[int(job[0,i])]):
				C_finish += 0
			else:
				C_finish += abs(jobtime[0,i]-self.delivery_time[int(job[0,i])])
		T_finish=max(tmm[0])			#最晚完工时间
		C_finish += T_finish
		trest=tmm-tmmw					#空闲时间
		E_all=sum((tmmw*self.p1)[0])+sum((trest*self.p2)[0])			#能耗计算
		Twork=sum(tmmw[0])                    #机器负荷
		# list_M,list_S,list_W,tmax是为了方便画图
		return C_finish,Twork,E_all,list_M,list_S,list_W,tmax,T_finish

	def decoding2(self,job,machine,machine_time):
		"""
		job:(1,55)
		machine:(1,55)
		machine_time:(1,55)
		"""
		jobtime=np.zeros((1,self.order_num))  # (1,10) # 订单的完工时间
		tmm=np.zeros((1,self.machine_num))  # (1,6) # 某工序机器的完工时间			
		tmmw=np.zeros((1,self.machine_num))	# (1,6) 

		jobingdic = dict(zip([item[0] for item in self.jobdic.items()],self.jobdic.keys()))
		for u in self.jobing:
			if(u[0] in jobingdic.keys()):
				jobtime[0,int(jobingdic[int(u[0])])-1] = u[2]
			tmm[0,int(u[1]-1)] = u[2]
					
		startime=0
		# list_M存放机器编号，list_S存放工序开始时间，list_W存放对应工序在对应机器上的加工时间
		list_M,list_S,list_W=[],[],[]
		for i in range(job.shape[1]): 
			svg,sig=int(job[0,i]),int(machine[0,i])-1  
			if(jobtime[0,svg]>0): # 如果不是订单的第一道工序								
				startime=max(jobtime[0,svg],tmm[0,sig])   	
				tmm[0,sig]=startime+machine_time[0,i]
				jobtime[0,svg]=startime+machine_time[0,i]
			if(jobtime[0,svg]==0): # 如果是订单的第一道工序							
				startime=tmm[0,sig] 
				tmm[0,sig]=startime+machine_time[0,i]
				jobtime[0,svg]=startime+machine_time[0,i] 

			tmmw[0,sig]+=machine_time[0,i] 
			list_M.append(machine[0,i]) 
			list_S.append(startime)
			list_W.append(machine_time[0,i])      
		tmax=np.argmax(tmm[0])+1		#结束最晚的机器
		C_finish = 0
		for i in range(len(jobtime[0])):
			if(jobtime[0][i]<=(self.delivery_time[i]-self.insertpoint)):
				C_finish += 0
			else:
				C_finish += abs(jobtime[0][i]-(self.delivery_time[i]-self.insertpoint))
		T_finish=max(tmm[0]) + self.insertpoint		#最晚完工时间
		C_finish += max(tmm[0])
		trest=tmm-tmmw					#空闲时间
		E_all=sum((tmmw*self.p1)[0])+sum((trest*self.p2)[0])			#能耗计算
		Twork=sum(tmmw[0])                    #机器负荷
		# list_M,list_S,list_W,tmax是为了方便画图
		return C_finish,Twork,E_all,list_M,list_S,list_W,tmax,T_finish


	def draw(self,filename,A,B,C,tmax,T_finish,colorfile):
		dd = pd.read_excel(filename)
		job = dd['订单编号'].values.reshape(1,-1)	#(1,55)
		self.order_num = job.max()
		list_M = dd['机器编号'].tolist()
		list_S = dd['工序开始时间'].tolist()
		list_W = dd['工序加工时间'].tolist()
		job[0] = job[0] - 1
		
		figure,ax=plt.subplots(figsize=(25,10))
		count=np.zeros((1,self.order_num))
		ecolor = np.array(pd.read_excel(colorfile,index_col='订单'))
		for i in range(job.shape[1]):  #每一道工序画一个小框
			count[0,int(job[0,i])]+=1
			icolor = (int(ecolor[int(job[0,i]),0])/255,int(ecolor[int(job[0,i]),1])/255,int(ecolor[int(job[0,i]),2])/255,0.8)
			ax.bar(x=list_S[i], bottom=list_M[i], height=0.5, width=list_W[i], orientation="horizontal",color=icolor,edgecolor='black')
			ax.text(list_S[i]+list_W[i]/32,list_M[i]+0.125, f'J{int(job[0,i]+1)}',color='black',fontsize=10,weight='bold')#12是矩形框里字体的大小，可修改
			ax.text(list_S[i]+list_W[i]/32,list_M[i], '{}'.format(int(count[0,int(job[0,i])])),color='black',fontsize=10,weight='bold')#12是矩形框里字体的大小，可修改

		if(self.insertpoint):
			ax.plot([self.insertpoint,self.insertpoint],[0,6],c='black',linestyle='-.')	
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='完工时间=%.1f'% (T_finish))#用虚线画出最晚完工时间	
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='重调度部分时间成本=%.1f'% (A))
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='重调度部分机器负荷=%.1f'% (B))
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='重调度部分能耗=%.1f'% (C))#可以选择显示几个目标，其他的用#号屏蔽
		else:
			ax.plot([self.insertpoint,self.insertpoint],[0,6],c='black',linestyle='-.')	
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='完工时间=%.1f'% (T_finish))#用虚线画出最晚完工时间	
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='时间成本=%.1f'% (A))
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='机器负荷=%.1f'% (B))
			ax.plot([T_finish,T_finish],[0,tmax],c='black',linestyle='-.',label='能耗=%.1f'% (C))#可以选择显示几个目标，其他的用#号屏蔽

		font1={'weight':'bold','size':22}#汉字字体大小，可以修改
		ax.set_xlabel("加工时间",font1)
		ax.set_title("甘特图",font1)
		ax.set_ylabel("机器",font1)

		scale_ls,index_ls=self.axis()
		plt.yticks(index_ls,scale_ls)
		plt.axis([0,T_finish*1.1,0,self.machine_num+1])
		plt.tick_params(labelsize = 22)#坐标轴刻度字体大小，可以修改
		labels=ax.get_xticklabels()
		[label.set_fontname('Times New Roman')for label in labels]
		plt.legend(prop={'family' : ['SimHei'], 'size'   : 16})#标签字体大小，可以修改
		plt.xlabel("加工时间",font1)
		plt.savefig(filename.split('.')[0]+'.png')
		plt.show()

	def get_result(self,job,machine,machine_time,colorfile,resultsfile):#画图
		if(self.flag==0):
			C_finish,Twork,E_all,list_M,list_S,list_W,tmax,T_finish=self.decoding1(job,machine,machine_time) 
			procedure = [] 
			count = np.zeros((1,self.order_num))
			for i in range(job.shape[1]):  #每一道工序画一个小框
				count[0,int(job[0,i])] += 1
				procedure.append(count[0,int(job[0,i])])
			result = pd.DataFrame({'订单编号':job[0]+1,'订单的工序号':procedure,'机器编号':list_M,'工序开始时间':list_S,'工序加工时间':list_W})
			result['工序完成时间'] = result['工序开始时间'] + result['工序加工时间']
			result = result.sort_values(by=['工序开始时间','工序完成时间']).to_excel(resultsfile,index=False)   
			self.draw(resultsfile,C_finish,Twork,E_all,tmax,T_finish,colorfile)
		else:
			C_finish,Twork,E_all,list_M,list_S,list_W,tmax,T_finish=self.decoding2(job,machine,machine_time) 
			procedure = [] 
			count = np.zeros((1,self.order_num))
			for i in range(job.shape[1]):  #每一道工序画一个小框
				count[0,int(job[0,i])] += 1
				procedure.append(count[0,int(job[0,i])])
			result = pd.DataFrame({'订单编号':job[0]+1,'订单的工序号':procedure,'机器编号':list_M,'工序开始时间':list_S,'工序加工时间':list_W})
			for u in result['订单编号'].unique(): 
				tmp = self.jobdic[int(u)][1]
				result.loc[result['订单编号']==u,'订单的工序号'] = result.loc[result['订单编号']==u,'订单的工序号'] + tmp
			result['订单编号'] = result['订单编号'].apply(lambda x:self.jobdic[int(x)][0])
			result['工序开始时间'] = result['工序开始时间'].apply(lambda x:x+self.insertpoint)
			result['工序完成时间'] = result['工序开始时间'] + result['工序加工时间']
			result = pd.concat([self.nojob,result])
			result = result.sort_values(by=['工序开始时间','工序完成时间']).to_excel(resultsfile,index=False)   
			self.draw(resultsfile,C_finish,Twork,E_all,tmax,T_finish,colorfile)
