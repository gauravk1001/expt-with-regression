import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

import math
from itertools import groupby
from operator import itemgetter
from numpy import linalg
from numpy import matlib
import matplotlib.lines as mlines
#matplotlib inline

global_mean = None

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD

    mean_mat = np.ndarray(shape = (5,2))
    mean_mat.fill(0)
    #print 'mean_mat', mean_mat, mean_mat.shape
    sum_mat = np.ndarray(shape = (5,2))
    #sum_mat.fill(None)
    
    meanval = np.mean(X, axis=0)
    #print 'meanval', meanval
    diff_vals = X - meanval

    for i in range(1,6):
        label_rows = np.where(y==i)
        label_rows = np.array(label_rows)        
        rows = X[label_rows]
        sum1 = np.sum(rows[0],axis=0)
        sum_mat[i-1] = sum1
        
        label_rows = np.array(label_rows)
        avg1 = sum1[:]/label_rows.shape[1]
        mean_mat[i-1] = avg1

    means = mean_mat.T
    global global_mean
    global_mean = means
    covmat = np.cov(diff_vals, rowvar=0)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    qda_list = []
    qda_cov = None

    mean_mat = np.ndarray(shape = (5,2))
    mean_mat.fill(0)
    #print 'mean_mat : ', mean_mat, mean_mat.shape
    mean_val = np.mean(X, axis = 0)
    
    for i in range(1,6):
        label_rows = np.where(y==i)
        label_rows = np.array(label_rows)
        rows = X[label_rows]

        diff_val = rows[0]-mean_val
        qda_cov = np.cov(diff_val.T)
        qda_list.append(qda_cov)

    means = global_mean
    covmats = qda_list
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar acc
    # IMPLEMENT THIS METHODuracy value    
    means_t = means.T
    pred_vals = np.ndarray(shape = (5))

    acc = 0

    for i in range (0, Xtest.shape[0]):
        for j in range(0, 5):
            temp = np.dot((Xtest[i] - means_t[j]), linalg.inv(covmat))
            power = np.dot(temp ,((Xtest[i] - means_t[j]).T)) / (-2)
            pred = ((math.exp(power)) / (2 * math.pi * math.sqrt(linalg.det(covmat))))
            pred_vals[j] = pred

        pred_idx  = np.argmax(pred_vals)
        if (pred_idx+1) == ytest[i]:
            acc = acc + 1

    acc = acc * 100.0/ (ytest.shape[0])
    #print 'LDA acc=', acc

    #discriminator plot
    x = np.sort(Xtest[:,0])
    y = np.sort(Xtest[:,1])
    X, Y = np.meshgrid(x,y)
    n = x.shape[0]*y.shape[0]
    D = np.zeros((n,2))
    D[:,0] = X.ravel();
    D[:,1] = Y.ravel();
    prediction_list1 = np.ndarray(shape = (D.shape[0],1))

    for i in range (0, D.shape[0]):
        for j in range(0, 5):
            temp = np.dot((D[i] - means_t[j]), linalg.inv(covmat))
            power = np.dot(temp ,((D[i] - means_t[j]).T)) / (-2)
            pred = ((math.exp(power)) / (2 * math.pi * math.sqrt(linalg.det(covmat))))
            pred_vals[j] = pred

        pred_idx1  = np.argmax(pred_vals)
        prediction_list1[i] = pred_idx1

    labels = prediction_list1.reshape(x.shape[0], y.shape[0]);
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.set_title('LDA Discriminating boundary')
    plt.contourf(x,y,labels)
    plt.show()
    #print 'D', D, D.shape
    #print  'X shape', X.shape, 'Y shape', Y.shape, 'prediction label shape', prediction_list1.shape

    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    #print means.shape, Xtest.shape, linalg.inv(covmats).shape
    #print Xtest, ytest
    pred_vals = np.ndarray(shape = (5))
    means_t = means.T

    acc = 0

    for i in range (0, Xtest.shape[0]):
        for j in range (0, 5):
            temp = np.dot((Xtest[i] - means_t[j]), linalg.inv(covmats[j]))
            power = np.dot(temp ,((Xtest[i] - means_t[j]).T)) / (-2)
            pred = ((math.exp(power)) / (2 * math.pi * math.sqrt(linalg.det(covmats[j]))))
            pred_vals[j] = pred
        
        pred_idx  = np.argmax(pred_vals)
        if (pred_idx+1) == ytest[i]:
            acc = acc + 1

    acc = acc * 100.0/ (ytest.shape[0])
    #print 'QDA acc=', acc

    #discriminator plot
    x = np.sort(Xtest[:,0])
    y = np.sort(Xtest[:,1])
    X, Y = np.meshgrid(x,y)
    n = x.shape[0]*y.shape[0]
    D = np.zeros((n,2))
    D[:,0] = X.ravel();
    D[:,1] = Y.ravel();
    prediction_list1 = np.ndarray(shape = (D.shape[0],1))

    for i in range (0, D.shape[0]):
        for j in range(0, 5):
            temp = np.dot((D[i] - means_t[j]), linalg.inv(covmats[j]))
            power = np.dot(temp ,((D[i] - means_t[j]).T)) / (-2)
            pred = ((math.exp(power)) / (2 * math.pi * math.sqrt(linalg.det(covmats[j]))))
            pred_vals[j] = pred

        pred_idx1  = np.argmax(pred_vals)
        prediction_list1[i] = pred_idx1

    labels = prediction_list1.reshape(x.shape[0], y.shape[0]);
    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    ax.set_title('QDA Discriminating boundary')    
    plt.contourf(x,y,labels)
    plt.show()
    #print 'D', D, D.shape
    #print  'X shape', X.shape, 'Y shape', Y.shape, 'prediction label shape', prediction_list1.shape

    
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD

    w1 = np.dot(linalg.inv(np.dot((X.T),X)),X.T)
    w=np.dot(w1,y)

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    b = X.shape[0]
    N = X.shape[1]
    
    temp = (lambd * b * np.identity(N) + np.dot(X.T, X))
    v=np.dot (X, (np.linalg.inv(temp)))
    w = np.dot(v.T, y)
    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    n = Xtest.shape[0]
    error = 0
    w_trans = w.T
    
    for i in range(0, n):
        error = error + ((ytest[i] - np.dot(w_trans, Xtest[i]))**2)
    J = (math.sqrt(error)) / n
    
    rmse = J
    
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD      
    N = len(X)
    w = np.mat(w).T

    part1 = (y - np.dot(X, w))
    part2 = ((lambd * np.dot(w.T,w))/2)

    J = (np.dot(part1.T, part1)) / (2*N) + part2
    
    delJ = ((-(np.dot(y.T, X)) + np.dot(w.T, np.dot(X.T, X))) / N) + (lambd * w.T)
    
    error=J
    delJ = delJ.T
    error_grad = np.array(delJ)
    error_grad = np.ndarray.flatten(error_grad)

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    N = x.shape[0]
    #print N
    Xd = np.ones((N,1))

    for i in range (1,p+1):
        abc = np.power(x, i)
        Xd = np.column_stack((Xd,abc))

    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mletrain = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mletrain_i = testOLERegression(w_i,X_i,y)

print 'weights from linear regression', w.shape
print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

print('RMSE without intercept on train data '+str(mletrain))
print('RMSE with intercept on train data'+str(mletrain_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1

#rmse plot
fig = plt.figure(1)
blue_line = mlines.Line2D([], [], color='blue', marker='o', label='RMSE3')
plt.legend(handles=[blue_line])
ax = fig.add_subplot(111)
ax.set_title('Ridge Regression')
ax.set_xlabel('lambda')
ax.set_ylabel('rmse3')
plt.text(lambdas[np.argmin(rmses3)],rmses3[np.argmin(rmses3)] - 0.05,'lambda opt='+ str(lambdas[np.argmin(rmses3)]))
plt.show(plt.plot(lambdas,rmses3, marker='o', markersize=3))
print 'Minimum rmse 3 is ', rmses3[np.argmin(rmses3)]
#print 'rmses3', rmses3

#train data errors
fig = plt.figure(9)
blue_line = mlines.Line2D([], [], color='blue', marker='o', label='RMSE3(Test data)')
green_line = mlines.Line2D([], [], color='green', marker='o', label='RMSE3(Train data)')
plt.legend(handles=[blue_line, green_line])
ax = fig.add_subplot(111)
ax.set_title('Ridge Regression (Train and test data)')
ax.set_xlabel('lambda')
ax.set_ylabel('rmse3train and rmse3')
plt.text(lambdas[np.argmin(rmses3train)],rmses3train[np.argmin(rmses3train)] - 0.05,'lambda opt='+ str(lambdas[np.argmin(rmses3train)]))
plt.show(plt.plot(lambdas,rmses3, marker='o', markersize=3))
plt.show(plt.plot(lambdas,rmses3train, marker='o', markersize=3))
#print 'rmse3train', rmses3train

#weights plot
fig = plt.figure(6)
indices = np.linspace(1, 65, 65)
blue_line = mlines.Line2D([], [], color='blue', marker='o', label='Linear Regression weights')
red_line = mlines.Line2D([], [], color='red', marker='o', label='Ridge Regression weights')
plt.legend(handles=[blue_line,red_line])
#plt.legend(handles=[red_line])
ax = fig.add_subplot(111)
ax.set_title('Weight Comparison')
ax.set_xlabel('indices')
ax.set_ylabel('weights')
plt.plot(indices, w_i, marker='o', color='blue', markersize=4, label='Linear Regression weights')
plt.plot(indices, w_l, marker='o', color='red', markersize=4, label='Ridge Regression weights')
plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4train = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4train[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1

#rmse4 plot
fig = plt.figure(2)
blue_line = mlines.Line2D([], [], color='blue',marker='o', label='RMSE4')
plt.legend(handles=[blue_line])
ax = fig.add_subplot(111)
ax.set_title('Ridge Regression using Gradient Descent')
ax.set_xlabel('lambda')
ax.set_ylabel('rmse4')
plt.text(lambdas[np.argmin(rmses4)] + 0.0003, rmses4[np.argmin(rmses4)],'lambda opt='+ str(lambdas[np.argmin(rmses4)]))
plt.show(plt.plot(lambdas,rmses4, marker='o', markersize=3))
#print 'weights from ridge regression with gradient descent', w_l_1.shape
print 'Minimum rmse 4 is ', rmses4[np.argmin(rmses4)]

#train data errors
fig = plt.figure(10)
blue_line = mlines.Line2D([], [], color='blue',marker='o', label='RMSE4(Train data)')
green_line = mlines.Line2D([], [], color='green', marker='o', label='RMSE4(Test data)')
plt.legend(handles=[blue_line, green_line])
ax = fig.add_subplot(111)
ax.set_title('Ridge Regression using Gradient Descent (Train and test data)')
ax.set_xlabel('lambda')
ax.set_ylabel('rmse4train and rmse4')
plt.text(lambdas[np.argmin(rmses4train)] + 0.0003, rmses4train[np.argmin(rmses4train)],'lambda opt='+ str(lambdas[np.argmin(rmses4train)]))
plt.show(plt.plot(lambdas,rmses4train, marker='o', markersize=3))
plt.show(plt.plot(lambdas,rmses4, marker='o', markersize=3))

#errors comparison plot
fig = plt.figure(7)
blue_line = mlines.Line2D([], [], color='blue', marker='o', label='RMSE3')
red_line = mlines.Line2D([], [], color='red', marker='o', label='RMSE4')
plt.legend(handles=[blue_line,red_line])
ax = fig.add_subplot(111)
ax.set_title('RMSE Comparison')
ax.set_xlabel('lambds')
ax.set_ylabel('RMSES')
plt.plot(lambdas, rmses3, color='blue', marker='o', markersize=1, label='RMSE3')
plt.plot(lambdas, rmses4, color='red', marker='o', markersize=1, label='RMSE4')
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
rmses5train = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    rmses5train[p,0] = testOLERegression(w_d1,Xd,y)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    rmses5train[p,1] = testOLERegression(w_d2,Xd,y)

fig = plt.figure(3)
blue_line = mlines.Line2D([], [], color='blue')
plt.legend(handles=[blue_line])
ax = fig.add_subplot(111)
ax.set_title('Non-linear Regression')
ax.set_xlabel('range(pmax)')
ax.set_ylabel('rmse5')
plt.show(plt.plot(range(pmax),rmses5, marker='o', markersize=6))
plt.legend(('No Regularization','Regularization'))

#train data errors
fig = plt.figure(8)
blue_line = mlines.Line2D([], [], color='blue')
plt.legend(handles=[blue_line])
ax = fig.add_subplot(111)
ax.set_title('Non-linear Regression(Train data)')
ax.set_xlabel('range(pmax)')
ax.set_ylabel('rmse5train')
plt.show(plt.plot(range(pmax),rmses5train, marker='o', markersize=6))
plt.legend(('No Regularization','Regularization'))
