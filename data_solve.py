import numpy as np 
import pandas as pd

class data_deal():
	def __init__(self,order_num=0,machine_num=0):
		self.order_num = order_num
		self.machine_num=machine_num
	def get_om(self):
		return self.order_num,self.machine_num
	def read(self,data):
		delivery_time = []
		info = dict()
		work = []
		for i in range(self.order_num):
			info[i] = dict()
			jobdf = data[data['order']==(i+1)]
			delivery_time.append(jobdf.iloc[0,-1])
			jobdf = jobdf.iloc[:,2:-1]
			for j in range(jobdf.shape[0]):
				work.append(i)
				info[i][j]=dict()
				for k in range(0,self.machine_num):
					machine_time = jobdf.iloc[j,k]
					if(machine_time!='-'):
						info[i][j][k+1] = machine_time
		work = np.array(work)
		return info,delivery_time,work

	def read_first(self,filename):
		data = pd.read_excel(filename,skiprows=1)
		data = data.fillna(method="ffill")
		self.order_num = len(data.iloc[:,0].unique())
		self.machine_num = len(data.columns)-3
		data.columns = ['order','procedure'] + ['M'+str(i) for i in range(1,self.machine_num+1)] + ['delivery_time']
		info,delivery_time,work = self.read(data)
		return info,delivery_time,work

	def read_Rescheduling(self,machinefilename,schedulefilename,Insertfilename):
		data = pd.read_excel(schedulefilename)
		data['工序完成时间'] = data['工序开始时间'] + data['工序加工时间']
		insertdf = pd.read_excel(Insertfilename,skiprows=1)
		insertdf.columns = list(insertdf.columns[:-2])+['插入时间','交货时间']
		insertdf = insertdf.fillna(method='ffill')
		insertpoint = insertdf.iloc[0,-2]
		eing = data[(data['工序开始时间']<=insertpoint) & (data['工序完成时间']>insertpoint)].reset_index(drop=True)
		eing['剩余加工时间'] = eing['工序完成时间'] - insertpoint
		jobing = np.array(eing[['订单编号','机器编号','剩余加工时间']])
		nojob = data[(data['工序开始时间']<=insertpoint)].reset_index(drop=True)
		rejob = data[(data['工序开始时间']>insertpoint)].reset_index(drop=True)
		machineinfo = pd.read_excel(machinefilename,skiprows=1)
		machineinfo.columns = list(machineinfo.columns)[:-1] + ['交货时间']
		machineinfo = machineinfo.fillna(method="ffill")
		rejobmachine = pd.DataFrame(columns=machineinfo.columns)
		for i in rejob.index:
			rejobmachine = pd.concat([rejobmachine,machineinfo[(machineinfo['订单']==rejob.loc[i,'订单编号']) & (machineinfo['工序']==rejob.loc[i,'订单的工序号'])]])
		rejobmachine = rejobmachine.sort_values(by='订单').reset_index(drop=True)
		rejobmachine = pd.concat([rejobmachine,insertdf.drop('插入时间',axis=1)]).reset_index(drop=True)
		self.order_num = len(rejobmachine['订单'].unique())
		self.machine_num = len(rejobmachine.columns)-3
		jobdic = dict()
		neworder = 1
		orderlis = rejobmachine['订单'].unique()
		for r in orderlis:
			jobdf = rejobmachine[rejobmachine['订单']==r]
			jobdic[neworder] = (r,jobdf['工序'].min())
			rejobmachine['订单'] = rejobmachine['订单'].replace(r,neworder)
			neworder = neworder + 1
		rejobmachine.columns = ['order','procedure'] + ['M'+str(i) for i in range(1,self.machine_num+1)] + ['delivery_time']
		reinfo,delivery_time,work = self.read(rejobmachine)
		return reinfo, delivery_time,work, jobdic,jobing, insertpoint,nojob


