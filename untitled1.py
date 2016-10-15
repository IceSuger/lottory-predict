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
from sklearn import svm 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split  
from sklearn.grid_search import GridSearchCV  


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

def splitNums(history):
    history[['1','2','3','4','5','6']] = pd.DataFrame([ x.split(',') for x in history['nums'].tolist() ])
    history[['6','7']] = pd.DataFrame([ x.split('|') for x in history['6'].tolist() ])
    history.drop('nums', axis=1, inplace=True)
    return history

def fillWithHuangli(history):
    print history
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
    target2016 = [[],[],[],[],[],[],[],[]]
    date2016 = []
    list_nums = soup.find_all('tr',{'class' : 't_tr1'})
    for x in list_nums:
        #位置1~7正好是7个球的号码，位置15为开奖日期
        for i in range(1,8):
            target2016[i].append( x.find_all('td')[i].text )
        #target2016.append( x.find_all('td')[7].text )
        date2016.append( x.find_all('td')[15].text )
    target2016 = pd.DataFrame(target2016).T.drop([0], axis=1)
    print target2016
    target2016.columns=['1','2','3','4','5','6','7']
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
#print history
patt = df.head(1)
print 'patt:'
#print patt
#df2016 = get2016()
#df2016.to_csv("2016.txt",encoding='gb2312', index=False)
df2016 = pd.read_csv("2016.txt",encoding='gb2312')
#print df2016

#f2016 = pd.concat([patt,df2016], axis=0) #按正确格式排过的df2016
f2016 = patt.merge(df2016,how='outer').drop(0).fillna(0)
print 'f2016:\n'
#print f2016

targets = [[],[],[],[],[],[],[],[]]

features = df.drop(['date','1','2','3','4','5','6','7','y','m','d'], axis=1)
features[u'y_山下火'] = 0.0
#features2016 = f2016.fillna(0).drop(['date','1','2','3','4','5','6','7','y','m','d',u'y_山下火'], axis=1)
features2016 = f2016.fillna(0).drop(['date','1','2','3','4','5','6','7','y','m','d'], axis=1)
target2016 = f2016['7']

for x in range(1,8):
    targets[x] = pd.concat( [ history[x], f2016[str(x)] ], ignore_index = True)
#features = df
newfeat = pd.concat([features,features2016 ],ignore_index = True)







#下面尝试用grid_search自动调参
X = newfeat
y = targets[7]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.01, random_state=0)  

tuned_parameters_rf = {'max_depth':[9,10,11,12,13,14], 'n_estimators':range(600,1800,100), 'random_state':[1,2]} 

tuned_parameters_ab = {'n_estimators':range(100,1500,50)}

rf = GridSearchCV(RandomForestClassifier(), tuned_parameters_rf)  
rf.fit(X, y)
print("RandomForest---Best parameters set found on development set:")  
print()  
print(rf.best_params_)  
print()  
print("Grid scores on development set:")  
print()  
for params, mean_score, scores in rf.grid_scores_:  
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))  
print()

ab = GridSearchCV(AdaBoostClassifier(), tuned_parameters_ab)  
ab.fit(X, y)
print("AdaBoost---Best parameters set found on development set:")  
print()  
print(ab.best_params_)  
print()  
print("Grid scores on development set:")  
print()  
for params, mean_score, scores in ab.grid_scores_:  
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))  
print()












"""
#下面加上2016年数据一块训练，并预测要买的号码
fs = [] #Random Forest
fs.append('')
lr = [] #Logistic Regression
lr.append('')
svms = [] #SVM
svms.append('')
ab = [] #Ada boost
ab.append('')
newfeat = pd.concat([features,features2016 ],ignore_index = True)

    
    

#用最后十组 测试准确率
ntargets = [[],[],[],[],[],[],[],[]]
print 'shape: ',newfeat.shape
newf = newfeat.drop(range(2009,2019))
for i in range(1,8):
    ntargets[i] = targets[i].drop(range(2009,2019))
    #fs
    fs.append( RandomForestClassifier(max_depth = 9, min_samples_split=2, n_estimators = 1300, random_state = 1))#自动化调参选出来的相对好的参数
    fs[i].fit(newf,ntargets[i])
    #lr
    lr.append( LogisticRegression() )   
    lr[i].fit(newf,ntargets[i])
    #svm
    svms.append( svm.SVC() )   
    svms[i].fit(newf,ntargets[i])
    #ada boost
    ab.append( AdaBoostClassifier(n_estimators=1150) )#自动化调参选出来的相对好的参数
    ab[i].fit(newf,ntargets[i])

new_test_feat = newfeat[2009:2019]
#print 'fea:'
#print new_test_feat

print 'random forest precision:'
for i in range(1,8):
    print fs[i].score(new_test_feat, targets[i][2009:2019])
print 'LR precision:'
for i in range(1,8):
    print lr[i].score(new_test_feat, targets[i][2009:2019])
print 'SVM precision:'
for i in range(1,8):
    print svms[i].score(new_test_feat, targets[i][2009:2019])
print 'Ada boost precision:'
for i in range(1,8):
    print ab[i].score(new_test_feat, targets[i][2009:2019])
    


#预测最新号码
#先用当前全部数据训练模型
newf = newfeat
for i in range(1,8):
    ntargets[i] = targets[i]
    #fs
    fs.append( RandomForestClassifier(max_depth = 12, min_samples_split=2, n_estimators = 1500, random_state = 1))
    fs[i].fit(newf,ntargets[i])
    #lr
    lr.append( LogisticRegression() )   
    lr[i].fit(newf,ntargets[i])
    #svm
    svms.append( svm.SVC() )   
    svms[i].fit(newf,ntargets[i])
    #ada boost
    ab.append( AdaBoostClassifier(n_estimators=1000) )
    ab[i].fit(newf,ntargets[i])
#然后把最新一期的开奖日期的特征搞出来
a = [{'date':'2016-10-16'}]
pred_df = pd.DataFrame(a)
print pred_df
pred_df = prepare(fillWithHuangli(pred_df))
pred_df = patt.merge(pred_df,how='outer').drop(0).fillna(0)
print pred_df
predict_feat = pred_df.fillna(0).drop(['date','1','2','3','4','5','6','7','y','m','d'], axis=1)
#然后预测
print 'Random Forest:'
for i in range(1,8):
    print fs[i].predict(predict_feat)
print 'AdaBoost:'
for i in range(1,8):
    print ab[i].predict(predict_feat)
"""

