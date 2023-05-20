import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['STSong'] 
from mpl_toolkits.mplot3d import axes3d
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

class mul_op():
	def divide(self,answer):
		S=[[] for i in range(len(answer))] # len(S)=100
		front = [[]]
		n=[0 for i in range(len(answer))] # 记录下比该点厉害的点的数量
		for p in range(len(answer)): # 构建每个点的支配集
			for q in range(len(answer)):
				# 如果p支配q
				if (np.array(answer[p])<=np.array(answer[q])).all() and (answer[p]!=answer[q]): 
					if q not in S[p]:
						S[p].append(q)  # 同时如果q不属于sp将其添加到sp中
				# 如果q支配p
				elif (np.array(answer[p])>=np.array(answer[q])).all() and (answer[p]!=answer[q]):
					n[p] = n[p] + 1  # 记录下比该点厉害的点的数量
			if n[p]==0:
				if p not in front[0]:
					front[0].append(p)
		i = 0
		while(front[i] != []): 
			Q=[]
			for p in front[i]:
				for q in S[p]:
					n[q] =n[q] - 1  
					if( n[q]==0):   # 如果n[q]==0，说明没有点比q点厉害
						if q not in Q:
							Q.append(q)
			i = i+1
			front.append(Q)
		del front[len(front)-1] # 最后一个是空数组删掉
		return front

	def dis(self,answer):
		crowder = []
		front=self.divide(answer)
		answerdf = pd.DataFrame(answer)
		answerdf.columns = ['x','y','z']
		for i in range(len(front)):
			if(len(front[i])>1):
				tmp = answerdf.loc[front[i]]
				x = tmp.sort_values(by=['x'],ascending=True)['x'] #由小到大排序
				y = tmp.sort_values(by=['y'],ascending=True)['y']
				z = tmp.sort_values(by=['z'],ascending=True)['z']
				tmpx = x.tolist()
				tmpy = y.tolist()
				tmpz = z.tolist()
				# 计算三个维度的拥挤度
				for i in range(1,(tmp.shape[0]-1)):
					if(tmpx[-1]==tmpx[0]):
						x.iloc[i] = 0	
					else:
						x.iloc[i] = (tmpx[i+1]-tmpx[i-1])/(tmpx[-1]-tmpx[0])
					if(tmpy[-1]==tmpy[0]):
						y.iloc[i] = 0
					else:
						y.iloc[i] = (tmpy[i+1]-tmpy[i-1])/(tmpy[-1]-tmpy[0])
					if(tmpz[-1]==tmpz[0]):
						z.iloc[i] = 0
					else:
						z.iloc[i] = (tmpz[i+1]-tmpz[i-1])/(tmpz[-1]-tmpz[0])
				x.iloc[0] ,y.iloc[0],z.iloc[0] = 100000,100000,100000 # 最左边的点
				x.iloc[-1] ,y.iloc[-1],z.iloc[-1] = 100001,100001,100001 # 最右边的点
				f = (x+y+z).sort_values(ascending=False) # 拥挤度由大到小排序
				crowder = crowder + list(f.index)
			else:
				crowder.append(front[i][0])
		return front,crowder

	def draw_change(self,fit_every):
		plt.figure(figsize=(25,10))
		font1={'weight':'normal','size':22}
		legend=[['最小完工时间','平均完工时间','最大完工时间'],['最小负荷','平均负荷','最大负荷'],['最小能耗','平均能耗','最大能耗']]
		title=['完工时间变化图','机器负荷变化图','能耗变化图']
		for i in range(3):
			plt.subplot(1,3,i+1)
			x=[fit_every[i][j][0] for j in range(len(fit_every[i]))]
			y=[fit_every[i][j][1] for j in range(len(fit_every[i]))]
			z=[fit_every[i][j][2] for j in range(len(fit_every[i]))]
			plt.plot(fit_every[3],x,c='black',linestyle='-')                                                  #作h1函数图像
			plt.plot(fit_every[3],y,c='black',linestyle='--')    
			plt.plot(fit_every[3],z,c='black',linestyle='-.')    
			plt.xlabel('迭代次数',font1)
			plt.title(title[i],font1)
			plt.legend(legend[i],fontsize=18)
			plt.tick_params(labelsize = 18)
		plt.show()

	# def divide(self,answer):
	# 	front=[]
	# 	setdf = pd.DataFrame(answer)
	# 	while(setdf.shape[0]):
	# 		f = []
	# 		for i in setdf.index:
	# 			Dominating_set = setdf[setdf<=setdf.loc[i,:]].dropna()
	# 			same_index = setdf[setdf==setdf.loc[i,:]].dropna().index
	# 			Dominating_set = Dominating_set.drop(same_index,axis=0)
	# 			if(Dominating_set.shape[0]==0):
	# 				f.append(i)
	# 		front.append(f)
	# 		setdf = setdf.drop(f,axis=0)
	# 	return front

	# def dis(self,answer):
	#     crowder,crowd=[],[]
	#     front=self.divide(answer) # front下标越小的，越优秀
	#     for i in range(len(front)):
	#         x=[answer[front[i][j]][0] for j in range(len(front[i]))] #取三个目标函数的各个目标
	#         y=[answer[front[i][j]][1] for j in range(len(front[i]))]
	#         z=[answer[front[i][j]][2] for j in range(len(front[i]))]
	#         sig=front[i]
	#         clo=[[] for j in range(len(front[i]))]
	#         if(len(sig)>1):    #每层的个体大于1个做拥挤度计算
	#             x_index,y_index,z_index=np.array(x).argsort(),np.array(y).argsort(),np.array(z).argsort()
	#             x.sort(),y.sort();z.sort()
	#             dis1,dis2,dis3=[],[],[]
	#             dis1.append(100000);dis2.append(100000);dis3.append(100000) # 该维度上左边最远的点
	#             if(len(sig)>2):    #大于2个做中间个体的拥挤度计算
	#                 for k in range(1,len(sig)-1):
	#                     distance1,distance2,distance3=(x[k+1]-x[k-1])/(x[-1]-x[0]),(y[k+1]-y[k-1])/(y[-1]-y[0]),(z[k+1]-z[k-1])/(z[-1]-z[0])
	#                     dis1.append(distance1);dis2.append(distance2);dis3.append(distance3)
	#             dis1.append(100001);dis2.append(100001);dis3.append(100001) # 该维度上右边最远的点
	#             crow=[]
	#             x_index=x_index.tolist()
	#             y_index=y_index.tolist()
	#             z_index=z_index.tolist()
	#             for m in range(len(sig)):
	#                 index1,index2,index3=x_index.index(m),y_index.index(m),z_index.index(m)
	#                 cro=dis1[index1]+dis2[index2]+dis3[index3]
	#                 crow.append(cro)
	#             crowd.append(crow)
	#             index=np.array(crow).argsort()
	#             for n in range(len(index)):     #拥挤度排列并取出
	#                 clo[n]=sig[index[n]]
	#             for o in range(len(clo)-1,-1,-1):
	#                 crowder.append(clo[o]) # 按拥挤度由大到小顺序填入

	#         else:
	#             crowder.append(front[i][0])
	#             crowd.append([1])
	#     return front,crowder # len(front)==len(crowd),len(crowder)=100