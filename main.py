from pickle import FALSE

from soupsieve import select
from data_solve import data_deal
from fjsp import FJSP
from multi_opt import mul_op
from nsga_2 import nsga_II
import numpy as np

flag = 1 # 如果需要重调度，写1
machinefile = 'MK01.xlsx'
powerfile = '设备负载与功率.xlsx'
selectpi = 0.5
iterationsNum = 50
individualsNum = 100
colorfile = '订单甘特图颜色.xlsx'
if(flag==0): 
    # 数据读取
    oj=data_deal()               
    # 数据处理 data_solve.py
    info,delivery_time,work = oj.read_first(machinefile)
    order_num, machine_num = oj.get_om() # mk01例子：order_num = 10, machine_num = 6
    parm_data=[info,delivery_time,work]

    # 数据初始化 fjsp.py,nsga_2.py,multi_opt.py
    to=FJSP(flag,powerfile,order_num,machine_num,selectpi,parm_data)      #工件数，机器数，选择最短机器的概率和mk01的数据
    oh=mul_op()
    ho=nsga_II(flag,iterationsNum,individualsNum,to,oh,work,order_num)     #数50,100,10分别代表迭代的次数、种群的规模、工件数
    #to是柔性车间模块，oh是多目标模块

    # 求解 nsga_2.py
    job,machine,machine_time,fit_every=ho.nsga_total()  #最后一次迭代的最优解
    oh.draw_change(fit_every)        #每次迭代过程中pareto解中3个目标的变化
    resultsfile = '第一次调度结果.xlsx'
    to.get_result(job,machine,machine_time,colorfile,resultsfile)#画pareto解的第一个解的甘特图
else:
    # 数据读取
    oj=data_deal()               
    # 数据处理
    firstResultFile = '第一次调度结果.xlsx'
    insertOrderFile = '插入订单.xlsx'
    info, delivery_time,work, jobdic,jobing, insertpoint,nojob = oj.read_Rescheduling(machinefile,firstResultFile,insertOrderFile)
    order_num, machine_num = oj.get_om() # mk01例子：order_num = 10, machine_num = 6
    parm_data=[info,delivery_time,work]

    # 数据初始化
    to=FJSP(flag,powerfile,order_num,machine_num,selectpi,parm_data,jobing, insertpoint, jobdic, nojob)      
    oh=mul_op()
    ho=nsga_II(flag,iterationsNum,individualsNum,to,oh,work,order_num)     #数50,100,10分别代表迭代的次数、种群的规模、工件数
    #to是柔性车间模块，oh是多目标模块

    # 求解 nsga_2.py
    job,machine,machine_time,fit_every=ho.nsga_total()  #最后一次迭代的最优解
    oh.draw_change(fit_every)        #每次迭代过程中pareto解中3个目标的变化
    resultsfile = '重调度结果.xlsx'
    to.get_result(job,machine,machine_time,colorfile,resultsfile)#画pareto解的第一个解的甘特图