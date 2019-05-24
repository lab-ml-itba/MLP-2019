from sklearn.linear_model import LogisticRegression
import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
from scipy.stats import norm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal

def plot_gaussians(X, mu1, sigma1, mu2, sigma2, prior1=0.5, prior2=0.5, N = 100, alpha = 0.5):
    X1 = np.linspace(X[:,0].min(), X[:,0].max(), N)
    X2 = np.linspace(X[:,1].min(), X[:,1].max(), N)
    X1, X2 = np.meshgrid(X1, X2)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X1.shape + (2,))
    pos[:, :, 0] = X1
    pos[:, :, 1] = X2

    # The distribution on the variables X, Y packed into pos.
    Z_1 = multivariate_normal.pdf(pos, mu1, sigma1) * prior1
    Z_2 = multivariate_normal.pdf(pos, mu2, sigma2) * prior2
    fig = plt.figure(figsize=(15,10))
    ax = fig.gca(projection='3d')
    cm1 = plt.cm.Reds
    cm2 = plt.cm.Blues
    ax.contourf(X1, X2, Z_1, 256, cmap = cm1)
    ax.contourf(X1, X2, Z_2, 256, alpha = alpha, cmap = cm2)
    plt.show()

def get_weights_array(ws):
    weights_norm = []
    for weights in ws:
        weights_norm.append(np.linalg.norm(weights[0]))
    return weights_norm

def params_vs_pol_order(N, k = 2):
    k = 2
    return int(np.math.factorial(N+2)/(np.math.factorial(N+2-k)*np.math.factorial(k)))

def get_curves(alturas_pol, pesos, al_min, al_max, mean, std, order = 3, N=20, lamb = 1, ptos = 100):
    WMLs, WRRs = get_MLE_MAP_weights(alturas_pol, pesos, order = order, lamb = lamb, N = N)
    al = np.linspace(al_min, al_max,ptos)
    al_lin_pol = get_lin_reg_pol(al, order, normalize=False, mean=mean, std=std)
    curv_MLE = np.zeros([len(WMLs), ptos])
    curv_MAP = np.zeros([len(WRRs), ptos])
    for i, w in enumerate(WMLs):
        curv_MLE[i] = al_lin_pol.dot(w)
    for i, w in enumerate(WRRs):
        curv_MAP[i] = al_lin_pol.dot(w)
    return curv_MLE, curv_MAP

def plt_lin_reg_gauss(alturas, pesos, WML, sigma, Xmin, Xmax, Ymin, Ymax, order, mean, std, points = 100, ax=None):
    X = np.linspace(Xmin, Xmax, points)
    Y = np.linspace(Ymin, Ymax, points)
    X, Y = np.meshgrid(X, Y)
    Xr = X.reshape(-1)
    Yr = Y.reshape(-1)
    Y_est = get_lin_reg_pol(Xr, order, normalize=False, mean=mean, std=std).dot(WML)
    Z = norm.pdf(Yr - Y_est, 0, sigma).reshape(points, points)
    Z_points = norm.pdf(pesos - get_lin_reg_pol(alturas, order, normalize=False, mean=mean, std=std).dot(WML),0 , sigma)
    if ax is None:
        fig = plt.figure(figsize=(20,10))
        ax = fig.gca(projection='3d')
    ax.contour3D(X, Y, Z, 512)
    ax.scatter3D(alturas, pesos, Z_points, color='r', marker='o')
    ax.view_init(65,-120)
    plt.show()

def get_MLE_MAP_weights(alturas_pol, pesos, order = 3, lamb = 0.1, N = 20):
    # Devuelve dos arrays con los pesos de MAP y MLE
    ident = np.matrix(np.identity(order+1))
    WMLs = []
    WRRs = []
    for i in range(int(np.floor(len(pesos)/N))):
        X = np.matrix(alturas_pol[i*N:(i+1)*N])
        y = np.matrix(pesos[i*N:(i+1)*N]).T
        wML = ((X.T.dot(X))**-1*X.T)*y
        WMLs = WMLs + [wML.tolist()]
        wRR = (lamb*ident + X.T*X)**-1*X.T*y
        WRRs = WRRs + [wRR.tolist()]
        #print(i, len(y))
    WMLs = np.array(WMLs).reshape(-1,order+1)
    WRRs = np.array(WRRs).reshape(-1,order+1)
    return WMLs, WRRs

def get_ridge_weights(alturas, pesos, lamb = 0.1):
    ident = np.matrix(np.identity(alturas.shape[1]))
    X = np.matrix(alturas)
    y = np.matrix(pesos).T
    wRR = (lamb*ident + X.T*X)**-1*X.T*y
    return wRR

def get_lin_reg_pol(data, order=1, normalize=True, mean = 0, std = 1):
    data_rep = np.repeat(data.reshape(-1,1), order+1, axis=1)
    exps = [i for i in range(order+1)]
    data_all = np.power(data_rep, exps)
    if normalize:
        mean = data_all.mean(axis=0)[1:]
        std = data_all.std(axis=0)[1:]
        data_all[:, 1:] = data_all[:, 1:] - data_all.mean(axis=0)[1:]
        data_all[:, 1:] = data_all[:, 1:]/data_all.std(axis=0)[1:]
        return data_all, mean, std
    else:
        data_all[:, 1:] = data_all[:, 1:] - mean
        data_all[:, 1:] = data_all[:, 1:]/std
        return data_all

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12, bias = True):
    # Recibe el dataset X de numero_de_muestras x features  y devuelve una matriz con todas las combinaciones 
    # De los productos del grado indicado en degree
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    if bias:
        return X_mat
    else:
        return X_mat[:,1:]

def plot_boundaries(X_train, y_train, score=None, probability_func=None, degree = None, n_colors = 100, mesh_res = 1000, ax = None):
    X = X_train #np.vstack((X_test, X_train))
    if len(y_train.shape) == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if probability_func is not None:
        if degree is not None:
            polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
            Z = probability_func(polynomial_set)[:, 1]
        else:
            Z_aux = probability_func(np.c_[xx.ravel(), yy.ravel()])
            Z = Z_aux[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
    
        cf = ax.contourf(xx, yy, Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
        plt.colorbar(cf, ax=ax)
        #plt.colorbar(Z,ax=ax)

        boundary_line = np.where(np.abs(Z-0.5)<0.001)

        ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.text(xx.max() - .3, yy.min() + .3, score,
                size=20, horizontalalignment='right')

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=40, marker='o')
    

def fit_and_get_regions(X_train, y_train, X_test, y_test, degree = 2, lambd = 0, plot_it = True, print_it = False):
    X_train_degree = get_polynimial_set(X_train, degree=degree)
    X_test_degree = get_polynimial_set(X_test, degree=degree)
    # Defino el modelo de clasificación como Regresion Logistica
    if lambd == 0:
        C1 = 10000000000
    else:
        C1 = 1/lambd 
    #C2 = 1
    clf_logist_pol = LogisticRegression(C=C1, fit_intercept=False)

    # Entreno el modelo con el dataset de entrenamiento
    clf_logist_pol.fit(X_train_degree, y_train)

    # Calculo el score (Exactitud) con el dataset de testeo
    score_test_logist_pol = clf_logist_pol.score(X_test_degree, y_test)

    # Calculo tambien el score del dataset de entrenamiento para comparar
    score_train_logist_pol = clf_logist_pol.score(X_train_degree, y_train)
    
    #loss_train = _logistic_loss(clf_logist_pol.coef_, X_train_degree, y_train, 1 / clf_logist_pol.C)
    #loss_test = _logistic_loss(clf_logist_pol.coef_, X_test_degree, y_test, 1 / clf_logist_pol.C)

    # print('Test Accuracy (Exactitud):',score_test_logist_pol)
    # print('Train Accuracy (Exactitud):',score_train_logist_pol)
    # print('coeficientes:', clf_logist_pol.coef_)
    # print('intercept:', clf_logist_pol.intercept_)
    if plot_it:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
        plot_boundaries(X_train, y_train, score_train_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax1)
        plot_boundaries(X_test, y_test, score_test_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax2)
        print('Regresion Logistica Polinomial de orden '+str(degree) +', con lamdba (regularización L2):' +  str(lambd))
        plt.show()
    if print_it:
        print('Train Accuracy (Exactitud):',score_train_logist_pol)
        print('Test Accuracy (Exactitud):',score_test_logist_pol)
    return score_train_logist_pol, score_test_logist_pol, clf_logist_pol.coef_ #, loss_train, loss_test

def plot_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, h = .02, ax = None, margin=0.5):
    X = X_train
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    print(Zaux.shape)
    
    if Zaux.shape[1] == 2:
        Z = Zaux[:, 1]
    else:
        Z = Zaux[:, 0]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, 50, cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if score is not None:
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=40, horizontalalignment='right')
    