import random
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()
#fit_Wh is the equivalent of the least square regression
def fit_Wh(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
#in cheb and chebx I am aproximating the coefficients of an equation of the given order
def cheb(xs, c):
    coefs  = c * [0] + [1]
    return np.polynomial.chebyshev.chebval(xs, coefs)

def chebx(x, order):
    xs=cheb(x, 0)
    for c in range(order-1):
        xs = np.vstack([xs, cheb(x, c+1)])
    return xs.T
#I created a class of points to make an equivalence between x and y elements
class Cooord:
    x = np.array([])
    y = np.array([])

def KFold(data, k=5):
    #shuffle the data
    arr = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    random.shuffle(arr)
    random_data = Cooord()
    a = np.array([])
    b = np.array([])
    for i in range(20):
        a = np.append(a, data.x[arr[i]])
        b = np.append(b, data.y[arr[i]])
    random_data.x = a
    random_data.y = b
    
    train_f_x= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test_f_x = np.array([0,0,0,0,0])
    train_f_y= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    test_f_y = np.array([0,0,0,0,0])
    
    fold_x = random_data.x[0:5]
    fold_x = np.vstack([fold_x, random_data.x[5:10]])
    fold_x = np.vstack([fold_x, random_data.x[10:15]])
    fold_x = np.vstack([fold_x, random_data.x[15:]])

    fold_y = random_data.y[0:5]
    fold_y = np.vstack([fold_y, random_data.y[5:10]])
    fold_y = np.vstack([fold_y, random_data.y[10:15]])
    fold_y = np.vstack([fold_y, random_data.y[15:]])

    for i in range(4):
        train_x = np.array([])
        test_x = np.array([])
        train_y = np.array([])
        test_y = np.array([])
        test_x = fold_x[i]
        test_y = fold_y[i]
        for j in range(4):
            if( j != i):
                train_x = np.append(train_x, fold_x[j])
                train_y = np.append(train_y, fold_y[j])
        train_f_x = np.vstack([train_f_x, train_x])
        train_f_y = np.vstack([train_f_y, train_y])
        test_f_x = np.vstack([test_f_x, test_x])
        test_f_y = np.vstack([test_f_y, test_y])
    return train_f_x, train_f_y, test_f_x, test_f_y

arg_list = sys.argv
xs, ys = load_points_from_file(str(arg_list[1]))
s_error = 0
nr_segm = len(xs) // 20
fig, ax = plt.subplots()

ax.scatter(xs, ys)
for i in range(nr_segm):
    #x_current and y_current are the arrays of 20 elements.
    x_current = np.array([])
    y_current = np.array([])
    for j in range(20):
        x_current = np.append(x_current, xs[20 * i + j])
        y_current = np.append(y_current, ys[20 * i + j])

    order = 2 #begins at 2 to be able to multiply the matrixes
    error = np.array([]) # the cross-validation error
    arr = Cooord()
    arr.x = x_current
    arr.y = y_current

    #train_x and test_x combine to obtain x_current
    #train_y and test_y combine to obtain y_current
    train_x, train_y, test_x, test_y = KFold(arr)
    x_train = train_x
    x_test = test_x
    y_train = train_y
    y_test = test_y

    #we can choose any value of x_train, x_test, y_test and y_train 
    # as long is the same  for all of them. There are 4 options( 0 - 3)
    x_train = x_train[2]
    x_test = x_test[2]
    y_test = y_test[2]
    y_train = y_train[2]
    
#calculating the cross-error for every order
    while(order < 20):
        weight = fit_Wh(chebx(x_train, order), y_train)
        yf = chebx(x_test, order).dot(weight)
        cross_error = ((y_test - yf) ** 2).mean()
        error = np.append(error, cross_error)
        order += 1

    #the cross-error for the sinusoidal function
    yf = chebx(np.sin(x_test), 2).dot(fit_Wh(chebx(np.sin(x_train), 2), y_train))
    cross_error = ((y_test - yf) ** 2).mean()
    e1 = error[0]
    i = 1
    while(error[i] < e1):
        e1 = error[i]
        i += 1
    #i is the order of the function
    
    #verifyng if the function is a polynomial  or a sinusoidal
    if(e1 < cross_error):
        weight = fit_Wh(chebx(x_train, i+1), y_train)
        yh = chebx(x_current, i+1).dot(weight)
        s_error = s_error + np.sum((y_current - yh) ** 2)
        ax.plot(x_current, yh , 'r', label="fitted")
    else:
        weight = fit_Wh(chebx(np.sin(x_train), 2), y_train)
        yh = chebx(np.sin(x_current), 2).dot(weight)
        s_error = s_error + np.sum((y_current - yh) ** 2)
        ax.plot(x_current, yh, 'r', label="fitted")

print(s_error)
if(len(arg_list) == 3):
    if(str(arg_list[2]) == "--plot"):
        plt.show()











