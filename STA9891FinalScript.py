import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
import time
import seaborn as sns


"""Returns the classification error of a given X and y set, and a given fitted model"""
def calc_error(X, y, model):
    '''returns in-sample error for already fit model.'''
    return 1 - model.score(X, y)



"""Read and process dataset"""
df = pd.read_csv('/Users/jonathandriscoll/Dropbox/Degrom Pitches 2018 Cleaned.csv')


y = df.loc[:,'pitch_type_code']

continuous_columns = ['release_speed','release_pos_x','release_pos_z','pfx_x',
                     'pfx_z','plate_x','plate_z','hc_x','hc_y','vx0','vy0','vz0','ax','ay','az','sz_top','sz_bot','hit_distance_sc',
                     'launch_speed','launch_angle','effective_speed','release_spin_rate','release_extension',
                     'release_pos_y']


X = df.drop(['Unnamed: 0','pitch_type','Batter L or R','type','on_3b','on_2b','on_1b','pitch_type_code'], axis = 1)
X[continuous_columns] = StandardScaler().fit_transform(X[continuous_columns]) #scales continuous predictors

X.to_csv('X.csv')
y.to_csv('y.csv')


def supportvec(X,y,inc=101):
    gamma = [0.1000000, 0.2154435, 0.4641589, 1.0000000, 2.1544347, 4.6415888, 10.0000000, 21.5443469, 46.4158883, 100.0000000]
    i = 0
    j=0
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])
    train_errors = []
    val_errors = []
    test_errors = []
    alphaiter = []
    train = []
    iter = []
    model = []
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))

    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)
        for alpha in gamma:

            for train_index, val_index in kf.split(x_2p_train, y_2p_train):
                xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
                ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
                clf = SVC(kernel='rbf', gamma = alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                iter.append(i)
                model.append('Radial SVM')
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                    format(i, alpha, size,
                           round(np.mean(train_errors), 4),
                           round(np.mean(val_errors), 4),
                           round(np.mean(test_errors), 4)))

        i += 1

    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)
        for alpha in gamma:

            for train_index, val_index in kf.split(x_10p_train, y_10p_train):
                xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
                ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
                clf = SVC(kernel='rbf', gamma=alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                iter.append(j)
                model.append('Radial SVM')
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                        format(j, alpha, size,
                               round(np.mean(train_errors), 4),
                               round(np.mean(val_errors), 4),
                               round(np.mean(test_errors), 4)))
        j += 1
    df['Gamma'] = alphaiter
    df['Model'] = model
    df['Train Error'] = train_errors
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Iteration'] = iter
    df['Train Size'] = train
    df.to_csv('SVMErrors.csv')
    return df

def ridgeclf(X,y, inc=101):
    alphas = [ 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    i = 0
    j=0
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])
    train_errors = []
    val_errors = []
    model = []
    test_errors = []
    alphaiter = []
    train = []
    iter = []
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))

    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_2p_train, y_2p_train):
                xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
                ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
                clf = LogisticRegression(C=alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
                alphaiter.append(alpha)
                model.append('Ridge')
                train.append(size)
                iter.append(i)
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                    format(i, alpha, size,
                           round(np.mean(train_errors), 4),
                           round(np.mean(val_errors), 4),
                           round(np.mean(test_errors), 4)))

        i += 1

    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_10p_train, y_10p_train):
                xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
                ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
                clf = LogisticRegression(C = alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                iter.append(j)
                model.append('Ridge')
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                        format(j, alpha, size,
                               round(np.mean(train_errors), 4),
                               round(np.mean(val_errors), 4),
                               round(np.mean(test_errors), 4)))
        j += 1
    df['Gamma'] = alphaiter
    df['Model'] = model
    df['Train Error'] = train_errors
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Train Size'] = train
    df['Iteration'] = iter
    df.to_csv('RidgeErrors.csv')
    return df

def lassoclf(X,y, inc=101):
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    i = 0
    j=0
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])
    train_errors = []
    model = []
    val_errors = []
    test_errors = []
    alphaiter = []
    train = []
    iter = []
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))

    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_2p_train, y_2p_train):
                xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
                ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
                clf = LogisticRegression(penalty = 'l1', C=alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                model.append('Lasso')
                iter.append(i)
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                    format(i, alpha, size,
                           round(np.mean(train_errors), 4),
                           round(np.mean(val_errors), 4),
                           round(np.mean(test_errors), 4)))

        i += 1

    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_10p_train, y_10p_train):
                xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
                ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
                clf = LogisticRegression(penalty = 'l1', C = alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                iter.append(j)
                model.append('Lasso')
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                        format(j, alpha, size,
                               round(np.mean(train_errors), 4),
                               round(np.mean(val_errors), 4),
                               round(np.mean(test_errors), 4)))
        j += 1
    df['Gamma'] = alphaiter
    df['Model'] = model
    df['Train Error'] = train_errors
    df['Train Size'] = train
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Iteration'] = iter
    df.to_csv('LassoErrors.csv')
    return df


def elasticclf(X,y, inc=101):
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    i = 0
    j=0
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])
    train_errors = []
    model = []
    val_errors = []
    test_errors = []
    alphaiter = []
    train = []
    iter = []
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))

    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_2p_train, y_2p_train):
                xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
                ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
                clf = SGDClassifier(penalty = 'elasticnet',alpha =alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                model.append('Elastic Net')
                iter.append(i)
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                    format(i, alpha, size,
                           round(np.mean(train_errors), 4),
                           round(np.mean(val_errors), 4),
                           round(np.mean(test_errors), 4)))

        i += 1

    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)
        for alpha in alphas:

            for train_index, val_index in kf.split(x_10p_train, y_10p_train):
                xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
                ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
                clf = SGDClassifier(penalty = 'elasticnet', alpha = alpha).fit(xkf_train, ykf_train)
                train_errors.append(calc_error(xkf_train, ykf_train, clf))
                val_errors.append(calc_error(xkf_val, ykf_val, clf))
                test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
                alphaiter.append(alpha)
                train.append(size)
                model.append('Elastic Net')
                iter.append(j)
                print(
                    'iteration:  {:3} | gamma: {:6} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                        format(j, alpha, size,
                               round(np.mean(train_errors), 4),
                               round(np.mean(val_errors), 4),
                               round(np.mean(test_errors), 4)))
        j += 1
    df['Gamma'] = alphaiter
    df['Train Size'] = train
    df['Model'] = model
    df['Train Error'] = train_errors
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Iteration'] = iter
    df.to_csv('ElasticErrors.csv')
    return df

def rfclf(X,y, inc = 101):
    i = 0
    j=0
    train_errors = []
    val_errors = []
    test_errors = []
    iter = []
    model = []
    train = []
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration','Train Size','Model'])
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))
    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)

        for train_index, val_index in kf.split(x_2p_train, y_2p_train):
            xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
            ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
            clf = RandomForestClassifier().fit(xkf_train, ykf_train)
            train_errors.append(calc_error(xkf_train, ykf_train, clf))
            val_errors.append(calc_error(xkf_val, ykf_val, clf))
            model.append('Random Forest')
            test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
            iter.append(i)
            train.append(size)
        i += 1
    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)

        for train_index, val_index in kf.split(x_10p_train, y_10p_train):
            xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
            ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
            clf = RandomForestClassifier().fit(xkf_train, ykf_train)
            train_errors.append(calc_error(xkf_train, ykf_train, clf))
            val_errors.append(calc_error(xkf_val, ykf_val, clf))
            test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
            iter.append(i)
            model.append('Random Forest')
            train.append(size)
        j += 1
    df['Train Error'] = train_errors
    df['Model'] = model
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Train Size'] = train
    df['Iteration'] = iter
    df.to_csv('RFErrors.csv')
    return df

def logisticclf(X, y, inc=101):
    i = 0
    j=0
    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])
    train_errors = []
    model = []
    val_errors = []
    test_errors = []
    alphaiter = []
    train = []
    iter = []
    x_2p_train, x_2p_test, y_2p_train, y_2p_test = train_test_split(X, y, train_size= 2 * len(X.columns))
    x_10p_train, x_10p_test, y_10p_train, y_10p_test = train_test_split(X,y, train_size = 10*len(X.columns))

    while i < inc:
        size = '2p'
        kf = KFold(n_splits=10)

        for train_index, val_index in kf.split(x_2p_train, y_2p_train):
            xkf_train, xkf_val = x_2p_train.iloc[train_index], x_2p_train.iloc[val_index]
            ykf_train, ykf_val = y_2p_train.iloc[train_index], y_2p_train.iloc[val_index]
            clf = SGDClassifier(penalty = 'none').fit(xkf_train, ykf_train)
            train_errors.append(calc_error(xkf_train, ykf_train, clf))
            val_errors.append(calc_error(xkf_val, ykf_val, clf))
            model.append('Logistic Regression')
            test_errors.append(calc_error(x_2p_test, y_2p_test, clf))
            train.append(size)
            iter.append(i)
            print(
                'iteration:  {:3} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                format(i, size,
                       round(np.mean(train_errors), 4),
                       round(np.mean(val_errors), 4),
                       round(np.mean(test_errors), 4)))

        i += 1

    while j < inc:
        size = '10p'
        kf = KFold(n_splits=10)

        for train_index, val_index in kf.split(x_10p_train, y_10p_train):
            xkf_train, xkf_val = x_10p_train.iloc[train_index], x_10p_train.iloc[val_index]
            ykf_train, ykf_val = y_10p_train.iloc[train_index], y_10p_train.iloc[val_index]
            clf = SGDClassifier(penalty = 'none').fit(xkf_train, ykf_train)
            train_errors.append(calc_error(xkf_train, ykf_train, clf))
            val_errors.append(calc_error(xkf_val, ykf_val, clf))
            test_errors.append(calc_error(x_10p_test, y_10p_test, clf))
            train.append(size)
            model.append('Logistic Regression')
            iter.append(j)
            print(
                'iteration:  {:3} | Train Size: {} | mean(train_error): {:7} | mean(val_error): {} | mean(test_error): {:6}'.
                    format(j, size,
                           round(np.mean(train_errors), 4),
                           round(np.mean(val_errors), 4),
                           round(np.mean(test_errors), 4)))
        j += 1
    df['Train Error'] = train_errors
    df['Model'] = model
    df['Train Size'] = train
    df['Validation Error'] = val_errors
    df['Test Error'] = test_errors
    df['Iteration'] = iter
    #df.to_csv('LogisticErrors.csv')
    return df





def dfappend():

    df = pd.DataFrame(columns=['Gamma', 'Train Error', 'Validation Error', 'Test Error', 'Iteration', 'Train Size','Model'])

    start_time = time.time()
    lr = logisticclf(X,y,inc=99)
    print("logistic regression took", start_time - time.time(),"seconds to run")
    lrtime = start_time - time.time()
    df = df.append(lr)
    df.to_csv('AllErrors.csv')
    ridge = ridgeclf(X,y, inc=99)
    print("Ridge Took:" , start_time - time.time(), " Seconds to run")
    ridgetime = start_time - time.time()
    df = df.append(ridge, ignore_index = True)
    df.to_csv('AllErrors.csv')
    lasso = lassoclf(X,y, inc=99)
    print("Lasso Took:" , start_time - time.time()," Seconds to run")
    lassotime = start_time - time.time()
    df = df.append(lasso, ignore_index = True)
    df.to_csv('AllErrors.csv')
    elastic = elasticclf(X,y, inc=99)
    elastictime = start_time - time.time()
    print("Elastic Net took:",start_time - time.time(),"seconds to run")
    df = df.append(elastic, ignore_index = True)
    df.to_csv('AllErrors.csv')
    forest = rfclf(X,y,inc=99)
    rftime = start_time - time.time()
    print("Random Forest took:", start_time - time.time(),"seconds to run")
    df = df.append(forest, ignore_index = True)
    df.to_csv('AllErrors.csv')
    svm = supportvec(X,y, inc=99)
    svmtime = start_time - time.time()
    print("SVM Took",start_time - time.time()," Seconds to run")
    df = df.append(svm, ignore_index = True)
    df.to_csv('AllErrors.csv')
    elapsed_time = time.time() - start_time
    print("overall time: ", elapsed_time, "seconds")
    print ("Logistic regression took {} Seconds, Ridge Regression took {} seconds, lasso took {} seconds, elastic took {} seconds, random forest took {} seconds, SVM took {} seconds".format(
        lrtime, ridgetime, lassotime, elastictime, rftime, svmtime)
    )





def charts():
    df = pd.read_csv('/Users/jonathandriscoll/Dropbox/PDF/STA9891/Final Project/AllErrors.csv')
    train2p = df[df['Train Size']=='2p']
    train10p = df[df['Train Size']=='10p']
    iter502p = df[(df['Iteration']==1) & (df['Train Size']== '2p') & ((df['Model'] != ('Random Forest'))| (df['Model']!= 'Logistic Regression'))]
    iter5010p = df[(df['Iteration'] == 1) & (df['Train Size'] == '10p')]


    box2ptrain = sns.boxplot(x = 'Model', y = 'Train Error', data=train2p)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Model')
    plt.ylabel('Classification Error Rate')
    plt.title('2p Train Error Rate vs. Model Type, 10-Fold CV')
    plt.show()

    box2ptest = sns.boxplot(x = 'Model', y = 'Test Error', data=train2p)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Model')
    plt.ylabel('Classification Error Rate')
    plt.title('2p Test Error Rate vs. Model Type, 10-Fold CV')
    plt.show()

    box10ptrain = sns.boxplot(x = 'Model', y = 'Train Error', data=train10p)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Model')
    plt.ylabel('Classification Error Rate')
    plt.title('10p Train Error Rate vs. Model Type, 10-Fold CV')
    plt.show()

    box10ptest = sns.boxplot(x = 'Model', y = 'Test Error', data=train10p)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Model')
    plt.ylabel('Classification Error Rate')
    plt.title('10p Test Error Rate vs. Model Type, 10-Fold CV')
    plt.show()


    errors = sns.scatterplot(x = 'Train Error', y = 'Validation Error', hue = 'Model' , style = 'Model', legend = 'brief', data = iter502p )
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Train Error')
    plt.ylabel('Validation Error Error Rate')
    plt.title('2p Train Error vs. Validation Error')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.show()

    errors10p = sns.scatterplot(x = 'Train Error', y = 'Validation Error', hue = 'Model' , style = 'Model', legend  = 'brief', data = iter5010p )
    plt.gcf().subplots_adjust(bottom=0.31)
    plt.xlabel('Train Error')
    plt.ylabel('Validation Error Error Rate')
    plt.title('2p Train Error vs. Validation Error')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.show()


if __name__ == "__main__":
    dfappend()
    charts()