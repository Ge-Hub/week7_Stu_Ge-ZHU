# Target of project -> buy or not buy -> purchasing probability of customers -> binar classification task(0 / 1) 
    # Step1: data loading 数据加载可小样本先行
    # Step2: data exploring 观察并排除异常数据。异常数据会影响model准确性
    # Step3: data loading 特征工程决定模型的上限
    # Step4: training  选择分类模型 (LR, RF, GBDT, GBDT+LR, XGBoost, LightGBM)
    # Step5: testing 
    # Step6: Evaluating modeling  


import pandas as pd

df = pd.read_csv('tianchi_fresh_comp_train_user.csv')
print(df.head())
#print(df.shape[0])
#print(df.shape[1])

print('-'*118)

# CVR calculation  购买占所有行为的比率
print(df['behavior_type'].value_counts())
count_all,count_4 = 0,0
count_user = df['behavior_type'].value_counts()
count_all = count_user[1] + count_user[2] + count_user[3] + count_user[4]
count_4 += count_user[4] #  += Equivalent to A = A + B (A=10, A+=5, A=15)
cvr = count_4 / count_all
print('CVR: {}%'.format(cvr*100))  #format() https://blog.csdn.net/rongDang/article/details/79771614

print('-'*118)

# Data exploring1: time 将time字段设置为pandas中datetime类型
df['time'] = pd.to_datetime(df['time']) # 数据列对应的类型是“object”，这样没法对时间数据处理，可以用过pd.to_datetime将该列数据转换为时间类型，即datetime
df.index = df['time'] # df.index = df'time' -> time列保留 / df.set_index('time') -> time列删除
print(df.head())

from collections import defaultdict
from datetime import datetime,timedelta

"""
The datetime classes in Python are categorized into main 5 classes:
1. date – Manipulate just date ( Month, day, year)
2. time – Time independent of the day (Hour, minute, second, microsecond)
3. datetime – Combination of time and date (Month, day, year, hour, second, microsecond)
4. timedelta— A duration of time used for manipulating dates
tzinfo— An abstract class for dealing with time zones
datetime加减：对日期和时间进行加减实际上就是把datetime往后或往前计算，得到新的datetime。加减可以用+和-运算符，不过需要导入timedalta这个类
注意timedelta 中只支持 days、seconds、microseconds
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
"""

def show_count_day(df):
    count_day = defaultdict(int)  #2014-11-18 to 遍历 2014-12-18
    str1 = '2014-11-17' 
    temp_date = datetime.strptime(str1,'%Y-%m-%d') # str转换为datetime：用户输入的日期和时间是字符串，要处理日期和时间，首先必须把str转换为datetime。转换方法是通过datetime.strptime()实现，需要一个日期和时间的格式化字符串。https://www.cnblogs.com/pingqiang/p/7812137.html
    delta = timedelta(days=1)
    for i in range(31):
        temp_date = temp_date + delta 
        temp_str= temp_date.strftime('%Y-%m-%d') #时间转化字符串类型
        count_day[temp_str] += df[df['time'] == temp_str].shape[0]
        #count_day[temp+str] += df[temp_str].shape[0] #df的index为时间
        print(temp_date)
    print(count_day)

    import matplotlib.pyplot as plt
    df_count_day = pd.DataFrame.from_dict(count_day, orient = 'index', columns = ['count'])
    df_count_day['count'].plot.bar(color ='black') # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

show_count_day(df)


# Data exploring2: 商品子集P的操作次数
df_p = pd.read_csv('tianchi_fresh_comp_train_item.csv')
df = pd.merge(df,df_p,on = ['item_id']) # set/reset https://www.cnblogs.com/Allen-rg/p/9694979.html
print(df.shape)
print(df.head())

show_count_day(df)

import matplotlib.pyplot as plt
def show_count_hour(date1):
    count_hour = {}
    for i in range(24):
        time_str = date1 + ' %0.2d' % i
        #print(time_str)
        count_hour[time_str] = [0,0,0,0]
        temp = df[df['time']==time_str]['behavior_type'].value_counts()
        for j in range(len(temp)):
            count_hour[time_str][temp.index[j]-1] += temp[temp.index[j]]
    print(count_hour)
    df_count_hour = pd.DataFrame.from_dict(count_hour,orient = 'index')
    df_count_hour.plot(kind='bar')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
show_count_hour('2014-12-12')