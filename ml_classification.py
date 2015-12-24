import scipy.io
import numpy as np
import time
from sklearn import svm
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics


LGR_PARAMS = [
    {
        'C': [0.0001, 0.0003, 0.001, 0.003, 0.1, 0.3, 1, 3, 10, 30, 100],
    }
]

def machine_learning(method='regression', num_iteration=1):
    input_rows = 50000
    X, y = train[:input_rows], np.ravel(train_label)[:input_rows]
    X = np.array(X, dtype=np.float64) # convert to float
    X_scaled = preprocessing.scale(X) # normalize
   # Logistic Regression.
    if method == 'regression':
        print "\nInvoking Logistic Regression on train_label data"
        logReg = linear_model.LogisticRegression()
        clf = GridSearchCV(logReg, LGR_PARAMS, cv=10)
        clf.fit(X_scaled, y)
        print 'Best Parameter that fit the model :', clf.best_params_
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.10, random_state=111)
        clf = linear_model.LogisticRegression(C=clf.best_params_.get('C'))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print '******** Average accuracy score in LogisticRegression: ***********', metrics.accuracy_score(y_test, predicted)
        return clf
    #SVM procedure.
    if method == 'svm':
        print "\nInvoking SVM on train_label data."
        clf = svm.SVC(
            C=40,cache_size=200,class_weight=None,coef0=0.0,degree=3,gamma=0.09,kernel='linear',  # linear / rbf / sigmoid / poly
            max_iter=-1,probability=False,random_state=None,shrinking=True,tol=0.001,verbose=False)
        average_accuracy_score = 0
        for ni in xrange(num_iteration):
            seed = np.random.randint(10000)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.10, random_state=seed)
            clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            accuracy_score = metrics.accuracy_score(y_test, predicted)
            # print '[%d/%d] accuracy score:%f\tseed:%d' % (ni + 1, num_iteration, accuracy_score, seed)
            average_accuracy_score += accuracy_score
        print '******** Average accuracy score in SVM : %f ***********' % (average_accuracy_score / num_iteration)
        return clf


def test_label_data():
    choice = input("Plese input 1 for LogisticRegression 2 for SVM Approach:\n")
    start_time = time.time()
    if(choice == 1):
        clf = machine_learning(method='regression', num_iteration=1)
    else:
        clf = machine_learning(method='svm', num_iteration=1)
    test_X = np.array(test, dtype=np.float64)
    test_X_scaled = preprocessing.scale(test_X)
    test_predicted = clf.predict(test_X_scaled)
    print "\nWriting data to Test_Labels file."
    with open('test_labels.txt', 'w') as f:
        for i,j in zip(test_predicted, test):
            f.write(str(i) + '\n')
    f.close()
    print("------- %s Time Taken in Seconds ------" % (time.time() - start_time))


mat = scipy.io.loadmat('cs6923Project.mat')
train_label = mat.get('train_label')
train = mat.get('train')
test = mat.get('test')

if __name__ == '__main__':
    test_label_data()

