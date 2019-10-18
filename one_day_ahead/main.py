'''
Created on Oct. 11, 2019

@author: dmh
'''
from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from sklearn import ensemble
from sklearn import linear_model

def get_data(currency):
    df = web.DataReader(currency, data_source='fred' ,start='2017-10-01',\
                        end='2019-10-01')
    #df=df[['Open','High','Low','Close','Volume']]
    return df

def interpolate_NaN(df):
    df = df.interpolate()
    #print('NaN values have been interpolated.')
    return df

def plot_data(df):
    df.plot()
    plt.show()

def print_NaN(df):
    df1 = df[df.isna().any(axis=1)]
    print(df1)

def count_NaN(df):
    df_NaN = df[df.isna().any(axis=1)]
    N_missing = len(df_NaN.index)
    #print(list(N_missing))
    if N_missing > 0:
        print('There are ' + str(N_missing) + ' NaN values, or ' + \
                str(100*N_missing/len(df.index)) + '% of all entries.')

def add_lags(currency,df, n_lags):
    df = df.join(pd.DataFrame({'lag_' + str(lag): df[currency].\
                            shift(lag) for lag in range(1,n_lags+1)}))
    return df.iloc[n_lags:]

def add_output(currency,df):
    df['rise'] = np.where(df[currency]>= df['lag_1'], 1, 0)
    return df

def prep_data(currency,df):
    df = interpolate_NaN(df)
    count_NaN(df)
    df = add_lags(currency,df, 10) 
    df  = add_output(currency, df)
    return df
    
def run_log_reg(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, \
                        y, test_size=0.4, random_state=9)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test, y_test)
    print(score)

def function_output(X,y,a,b):
    svm = SVC(kernel='rbf', random_state=0, gamma=a, C=b)
    x_train, x_test, y_train, y_test = train_test_split(X, \
                        y, test_size=0.4, random_state=9)
    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    return (score)

def test_grad_desc(X,y):
    gamma_list = [0.1, 0.2, 0.4, 0.5, 0.55, 0.59, 0.6, 0.61, 0.65, 0.7, 0.8,\
                   1, 2, 4, 8, 10]
    C_list = [0.5, 1, 2, 4, 8, 10, 20, 40, 50, 60, 80, 90, 100, 130, \
        145, 147,148, 149, 150, 154.2, 154.3, 154.4,154.5,154.6, 155.5, 160, 200]
    high_score = 0
    high_C = 0
    high_gamma =0
    for g in gamma_list:
        for C in C_list:
            score = function_output(X,y,g,C)
            if score > high_score:
                high_score = score
                high_C = C
                high_gamma = g
    print(high_score)
    print(high_C)
    print(high_gamma)


def run_SVM(X,y):
    svm = SVC(kernel='rbf', random_state=0, gamma=0.59, C=154.3)
    x_train, x_test, y_train, y_test = train_test_split(X, \
                        y, test_size=0.4, random_state=9)
    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    print(score)
    
def tree_output(X,y,e,depth,learn):
    params = set_params(e,depth,learn,'friedman_mse')
    grad_booster = ensemble.GradientBoostingClassifier(**params)
    x_train, x_test, y_train, y_test = train_test_split(X, \
                        y, test_size=0.4, random_state=9)
    grad_booster.fit(x_train,y_train)
    score = grad_booster.score(x_test, y_test)
    return (score)    
    
def find_tree_params(X,y):
    est_list = [20,21,22,30,40,60,80,100,150]
    depth_list = [3,4,5]
    learning_list = [0.3,0.4,0.5,0.59,0.6,0.61]
    high_score = 0
    high_est = 0
    high_depth =0
    high_learn =0
    for e in est_list:
        for depth in depth_list:
            for learn in learning_list:
                sum = 0
                for n in range(10):
                    score = tree_output(X,y,e,depth,learn)
                    sum += score
                score = sum / 10.0
                print(score)
                if score > high_score:
                    high_score = score
                    high_est = e
                    high_depth = depth
                    high_learn = learn
    print(high_score)
    print(high_est)
    print(high_depth)
    print(high_learn)
    
def run_boosted_tree(X,y):
    params = set_params(20,4,1.0,'friedman_mse') #100,2,0.25/20,4,1.0
    grad_booster = ensemble.GradientBoostingClassifier(**params)
    x_train, x_test, y_train, y_test = train_test_split(X, \
                        y, test_size=0.4, random_state=9)
    grad_booster.fit(x_train,y_train)
    score = grad_booster.score(x_test, y_test)
    print(score)
    
def set_params(n_est,max_d,learn_rate,criterion):
    params = {
        'n_estimators': n_est,
        'max_depth': max_d,
        'learning_rate': learn_rate,
        'criterion': criterion
    }
    return params    

if __name__ == '__main__':
    currency = 'DEXUSUK'
    df = get_data(currency)
    df = prep_data(currency, df) 
    X = df.drop(columns=['rise',currency])
    X = np.asarray(X)
    y = df[['rise']]
    y = np.asarray(y)
    y = np.ravel(y)  
    find_tree_params(X,y)  
    #test_grad_desc(X,y)
    #run_SVM(X,y)
    #run_log_reg(X,y)
    #run_boosted_tree(X,y)
    
    