#configure testing
feature_num = 1000
test_times = 10
test_DP = 1
epsilon = 0.1


from sklearn.datasets import fetch_20newsgroups
categories = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware'];

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,categories=categories)


#print(len(twenty_train.target_names),len(twenty_train.data),len(twenty_train.filenames),len(twenty_test.data))
from time import time
t0 = time()
def AddLapNoise(realNum, scal):
    from scipy.stats import laplace
    x = realNum + laplace.rvs(scale=scal,size=1)
    #print x
    #because we can post-processing differentially private data
    return x

def AddLapNoise2(realNum, scal):
    from scipy.stats import laplace
    x = realNum + laplace.rvs(scale=scal,size=1)
    #print x
    #because we can post-processing differentially private data
    if x < 0 :
        re = 0
    else:
        re = round(x)
    return re

def AddLapNoise4DenseMatrix(X, scal):
    # cal m n
    from scipy.sparse import csr_matrix
    m = csr_matrix(X).shape[0]
    n = csr_matrix(X).shape[1]
    #generate laplace noise array
    from scipy.stats import laplace
    data = laplace.rvs(scale=scal, size=m * n)
    #generate lalace noise matrix
    col = range(n) * m
    row = []
    #tt = time()
    for i in range(m):
        row = row + [i] * n
    A = csr_matrix((data,(row,col)),shape=(m,n))
    #print 'time1:',time()-tt
    #add noise to real data
    return Smooth(A+X)

def AddLapNoise4DenseMatrix2(X, scal):
    from scipy.sparse import csr_matrix
    m = csr_matrix(X).shape[0]
    n = csr_matrix(X).shape[1]
    A = csr_matrix((m,n)).todense()
    for i in range(m):
        for j in range(n):
            A[i, j] = AddLapNoise(0, scal)
    return Smooth(csr_matrix(A)+X)

def Smooth(X):
    #tt = time()
    m = csr_matrix(X).shape[0]
    n = csr_matrix(X).shape[1]
    A = csr_matrix((m, n)).todense()
    for i in range(m):
        for j in range(n):
            if X[i,j] < 0:
                A[i,j] = 0
            else:
                A[i,j] = round(X[i, j])
    #print 'time2:', time() - tt
    return A

def AddLapNoise2SparseMatrix(X, scal):
# just test, only add noise to non-zero bits
    from scipy.sparse import coo_matrix
    cx = coo_matrix(X)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        X[i, j] = AddLapNoise(X[i, j], scal)
    return X

def AddLapNoise2SparseMatrix_TEST(X, scal):
# just test, only add noise to non-zero bits
    print X[1,1]
    for i in range(1000):
        X[0, i+75210] = 10
        X[10, i] = 200
        X[100, i] = 100
    return X

def AddLapNoise2SparseMatrix_Zero(X, scal):
# just test, only add noise to non-zero bits
    from scipy.sparse import coo_matrix
    cx = coo_matrix(X)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        X[i, j] = 0;
    return X


def mean_absolute_relative_error(X, Y,r):
    from numpy import mean, abs
    m = csr_matrix(X).shape[0]
    n = csr_matrix(X).shape[1]
    A = csr_matrix((m, n)).todense()
    for i in range(m):
        for j in range(n):
                A[i, j] = abs((X[i,j] - Y[i,j])/max(float(r),float((X[i,j]))))
    return mean(A)


# -- Calculate the frequency of words (features)
#from sklearn.feature_extraction.text import HashingVectorizer
#count_vect = HashingVectorizer(stop_words = 'english',non_negative = True, n_features = 10000)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english",decode_error='ignore',max_features=feature_num)

X_train_counts = count_vect.fit_transform(twenty_train.data)
print 'TF is prepared, done in time ', (time() - t0)

from scipy.sparse import coo_matrix
print 'TF matrix size:', coo_matrix(X_train_counts).shape
#print X_train_counts


#print A.toarray()

t1 = time()


# -- Print info about features
#print count_vect.get_feature_names()
#for word,index in sorted(count_vect.vocabulary_.items(), key  = lambda x: x[1]): print "%s\t%d" % (word, index)


#X_train_counts.shape

# -- Calculate TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
from scipy.sparse import coo_matrix
print 'TF-IDF matrix size:', coo_matrix(X_train_tfidf).shape
#print X_train_tfidf

# -- Print info about features
#print count_vect.get_feature_names()
#for word,index in sorted(count_vect.vocabulary_.items(), key  = lambda x: x[1]): print "%s\t%d" % (word, index)


from scipy.sparse import csr_matrix
from numpy import savetxt,asarray
#savetxt("X_train_tfidf.csv", csr_matrix(X_train_tfidf).toarray(), fmt='%.5f', delimiter=",")
#savetxt("count_vect.csv", asarray(count_vect.vocabulary_.keys()), fmt='%s', delimiter=",")



# ===== Using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)



# -- Test: predict the topic of a given document
#docs_new = ['God is love','OpenGL on the GPU is fast']
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#predicted = clf.predict(X_new_tfidf)
#for doc,category in zip(docs_new,predicted): print("%r => %s") %(doc,twenty_train.target_names[category])


docs_new = twenty_test.data
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


# -- Evaluate accuracy of MultinomialNB
import numpy as np
predicted = clf.predict(X_new_tfidf)
print np.mean(predicted == twenty_test.target)
# - detailed evaluation
from sklearn import metrics
print(metrics.classification_report(twenty_test.target,predicted,
                                   target_names = twenty_test.target_names))


# ===== Using SVM
print "using SVM:"
# -- Evaluate accuracy of  SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
clf_svm = SGDClassifier(loss = 'hinge',penalty = 'l2',
                                          alpha = 1e-3,n_iter = 5, random_state = 42).fit(X_train_tfidf,twenty_train.target)

predicted_svm = clf_svm.predict(X_new_tfidf)
print np.mean(predicted_svm == twenty_test.target)
# - detailed evaluation
print(metrics.classification_report(twenty_test.target,predicted_svm,
                                   target_names = twenty_test.target_names))


t_dp = time()
# -- DP version
if test_DP == 1:

    X_train_counts_DP_all = []
    X_train_counts_all = []
    predicted_DP_nb_all = []
    predicted_DP_svm_all = []
    target_all = np.tile(twenty_test.target, test_times)

    for test_idx in range(test_times):
        # -- Calculate differentially privaye TF
        X_train_counts_DP = AddLapNoise4DenseMatrix(X_train_counts, 1 / epsilon)
        # X_train_counts_DP = AddLapNoise2SparseMatrix(X_train_counts, epsilon)

        #X_train_counts_DP_all = np.append(X_train_counts_DP_all, csr_matrix(X_train_counts_DP).toarray())
        #X_train_counts_all = np.append(X_train_counts_all, csr_matrix(X_train_counts).toarray())

        print 'noised TF is prepared, done in time ', (time() - t_dp)
        # print X_train_counts_DP
        from scipy.sparse import coo_matrix

        #from numpy import savetxt, asarray
        #savetxt("X_train_counts_DP.csv", csr_matrix(X_train_counts_DP).toarray(), delimiter=",")
        #savetxt("X_train_counts.csv", csr_matrix(X_train_counts).toarray(), delimiter=",")

        # -- Calculate DP version TF-IDE
        tfidf_transformer_DP = TfidfTransformer()
        X_train_tfidf_DP = tfidf_transformer_DP.fit_transform(X_train_counts_DP)
        # print X_train_tfidf_DP

        print 'noised TF-IDF matrix size:', coo_matrix(X_train_tfidf_DP).shape

        # -- Train DP version Naive Bayes
        print '-- using noised Naive Bayes, test_idx=', test_idx
        clf_DP_NB = MultinomialNB().fit(X_train_tfidf_DP, twenty_train.target)

        # -- predict using trained model
        predicted_DP_NB = clf_DP_NB.predict(X_new_tfidf)
        print np.mean(predicted_DP_NB == twenty_test.target)
        predicted_DP_nb_all = np.append(predicted_DP_nb_all, predicted_DP_NB)

        # -- Train DP version SVM
        print '-- using noised SVM, test_idx=', test_idx
        clf_DP_svm = SGDClassifier(loss='hinge', penalty='l2',
                               alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf_DP, twenty_train.target)
        predicted_DP_svm = clf_DP_svm.predict(X_new_tfidf)
        print np.mean(predicted_DP_svm == twenty_test.target)
        predicted_DP_svm_all = np.append(predicted_DP_svm_all, predicted_DP_svm)


    # -- ERRORs
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(csr_matrix(X_train_counts).toarray(), csr_matrix(X_train_counts_DP).toarray())
    mae = mean_absolute_error(csr_matrix(X_train_counts).toarray(), csr_matrix(X_train_counts_DP).toarray())
    mre = mean_absolute_relative_error(csr_matrix(X_train_counts), csr_matrix(X_train_counts_DP), 1.0)
    print 'mse=', mse
    print 'mae=', mae
    print 'mre=', mre

    # -- evaluation
    from sklearn import metrics
    print '# average error of Naive Bayes:'
    print np.mean(predicted_DP_nb_all == target_all)
    print(metrics.classification_report(target_all, predicted_DP_nb_all, target_names=twenty_test.target_names))

    print '# average error of SVM:'
    print np.mean(predicted_DP_svm_all == target_all)
    print(metrics.classification_report(target_all, predicted_DP_svm_all, target_names=twenty_test.target_names))

print 'test is finished, done in time', (time() - t0)