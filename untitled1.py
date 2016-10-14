# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:28:11 2016

@author: X93
"""

import urllib
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def getHuangli(date_str):
    print date_str
    res=[]
    try:
        soup = BeautifulSoup(urllib.urlopen("http://www.meiguoshenpo.com/huangli/"+ date_str + ".html")) 
        
        nianwuxing = soup.find('th',text='年五行')
        nian =  nianwuxing.find_next_sibling().text
        yuewuxing = nianwuxing.find_parent().find_next_sibling().find('th',text='月五行')
        yue =  yuewuxing.find_next_sibling().text
        riwuxing = yuewuxing.find_parent().find_next_sibling().find('th',text='日五行')
        ri =  riwuxing.find_next_sibling().text
        res = [nian,yue,ri]
    finally:
        return res
    
#print getHuangli('2009-10-01')
"""
year = '2015'
history = pd.read_csv(year+'.txt','\s+', header=None, names=['nums','date'], usecols = [1,2])
#history[[1,2,3,4,5,6,7]]= history['nums'].str.split(',|')
history[['1','2','3','4','5','6']] = pd.DataFrame([ x.split(',') for x in history['nums'].tolist() ])
history[['6','7']] = pd.DataFrame([ x.split('|') for x in history['6'].tolist() ])
history.drop('nums', axis=1, inplace=True)
history[['y','m','d']] = pd.DataFrame([ getHuangli(x) for x in history['date'].tolist() ])
print 'ok'
"""

def makeTable():
    chunks = []
    for year in range(2003,2016):
        history = pd.read_csv(str(year)+'.txt','\s+', header=None, names=['nums','date'], usecols = [1,2])
        chunks.append(history)
    return pd.concat(chunks, ignore_index=True)

def fillWithHuangli(history):
    history[['1','2','3','4','5','6']] = pd.DataFrame([ x.split(',') for x in history['nums'].tolist() ])
    history[['6','7']] = pd.DataFrame([ x.split('|') for x in history['6'].tolist() ])
    history.drop('nums', axis=1, inplace=True)
    history[['y','m','d']] = pd.DataFrame([ getHuangli(x) for x in history['date'].tolist() ])
    print 'ok'
    return history

def prepare(history):
    dummies_df = pd.get_dummies(history[['y','m','d']])
    #dummies_df = dummies_df.rename(columns=lambda x:'WX_'+str(x))
    df = pd.concat([history,dummies_df], axis=1)
    

    print df
    return df
    
def get2016():
    soup = BeautifulSoup(urllib.urlopen("20162016.html")) 
    target2016 = []
    date2016 = []
    list_nums = soup.find_all('tr',{'class' : 't_tr1'})
    for x in list_nums:
        #位置1~7正好是7个球的号码，位置15为开奖日期
        target2016.append( x.find_all('td')[7].text )
        date2016.append( x.find_all('td')[15].text )
    target2016 = pd.DataFrame(target2016)
    target2016.columns=['7']
    date2016 = pd.DataFrame(date2016)
    date2016.columns=['date']
    
    df2016 = pd.concat([target2016, date2016], axis=1)
    
    df2016[['y','m','d']] = pd.DataFrame([ getHuangli(x) for x in df2016['date'].tolist() ])
    df2016 = prepare(df2016)
    
    return df2016

#history = fillWithHuangli( makeTable() )
#dummed = prepare(history) 
#dummed.to_csv("2003~2015.txt", encoding='gb2312', index=False)

#df = pd.read_csv("2003~2015.txt",header=None)
df = pd.read_csv("2003~2015.txt",encoding='gb2312') #03~15年的所有中奖号码和五行数据（包括原始的和吖编码之后的）
history = pd.read_csv("history.txt",header=None) #03~15年的所有中奖号码 第0列为日期
print "history:"
print history
patt = df.head(1)
print 'patt:'
print patt
#df2016 = get2016()
#df2016.to_csv("2016.txt",encoding='gb2312', index=False)
df2016 = pd.read_csv("2016.txt",encoding='gb2312')
print df2016

#f2016 = pd.concat([patt,df2016], axis=0) #按正确格式排过的df2016
f2016 = patt.merge(df2016,how='outer')
print 'f2016:\n'
print f2016

targets = [[],[],[],[],[],[],[],[]]
for x in range(1,8):
    targets[x] = history[x]
features = df.drop(['date','1','2','3','4','5','6','7','y','m','d'], axis=1)
features2016 = f2016.fillna(0).drop(['date','1','2','3','4','5','6','7','y','m','d',u'y_山下火'], axis=1)
target2016 = f2016['7'].fillna(0)
#features = df
#lr = LogisticRegression()

"""
print 'LR predict precision:'
lrs = [[],[],[],[],[],[],[],[]]
for x in range(1,8):
    lrs[x] = lr.fit(features, targets[x])
    #lrs.append(lr.fit(features, targets[x]))
    print lrs[x].score(features, targets[x])
   

print "RF precision:"
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 600, random_state = 1)
fs = [[],[],[],[],[],[],[],[]]
for x in range(1,8):
    fs[x] = forest.fit(features, targets[x])
    #lrs.append(lr.fit(features, targets[x]))
    print fs[x].score(features, targets[x])
"""

#f2016.to_csv("f2016.txt", index=False, encoding='gb2312')
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 1000, random_state = 1)
forest.fit(features, history[7])
print "RF precision old:"
print forest.score(features, history[7])
print "RF precision 2016:"
print forest.score(features2016, target2016)



#df2016 = get2016()
#print df2016
#test_features = df2016.drop(['y','m','d'], axis=1).values
#fs[7].score()