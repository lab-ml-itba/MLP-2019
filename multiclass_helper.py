from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def get_custom_cmap(Ri=0, Gi=0, Bi=0, alpha=0.8):
    N_c = 256
    N_c_col = int(N_c/2)
    marg = 0
    R = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Ri, N_c_col+marg)])
    G = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Gi, N_c_col+marg)])
    B = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Bi, N_c_col+marg)])
    A = np.hstack([np.zeros(N_c_col-marg), alpha*np.ones(N_c_col+marg)])
    custom_map = ListedColormap(np.vstack([R,G,B,A]).T)
    return custom_map

def plot_MC_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, mesh_res = 300, ax = None, margin=0.5, color_index = 0, normalize = False):
    y_train_cat_aux = to_categorical(y_train)
    if (y_train_cat_aux.shape[1] > 2):
        y_train_cat = y_train_cat_aux
    else:
        y_train_cat = y_train
    X = X_train
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
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    
    # if Zaux.shape[1] == 2:
        # Es un polinomio
        # Z = Zaux[:, 1]
    # else:
        # No es un polinomio
        # Z = Zaux[:, 2]

    # Put the result into a color plot
    
    if normalize:
        Zaux = (Zaux.T/Zaux.sum(axis=1)).T
    
    cm_borders = ListedColormap(["#FFFFFFFF", "#000000"])
    my_colors = [[0,0,0.5], [0,0.5,0], [0.5,0,0], [0.5,0.5,0.5], [0,0.5,0.5]]
    # my_colors = [list(x) for x in list(mpl.colors.BASE_COLORS.values())]
    cat_order = len(y_train_cat.shape)
    if cat_order>1:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1], y_train_cat.shape[1])
        for i in range(Z_reshaped.shape[2]):
            my_cmap = get_custom_cmap(my_colors[i][0],my_colors[i][1],my_colors[i][2], alpha=0.5)
            Z = Z_reshaped[:,:,i]    

            cf = ax.contourf(xx, yy,
                             Z,
                             50, 
                             vmin = 0,
                             vmax = 1,
                             cmap=my_cmap, 
                            )
            ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               cmap=ListedColormap(my_colors),
               edgecolors='k', 
               s=100)
    else:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1])
        my_cmap = get_custom_cmap(my_colors[color_index][0],my_colors[color_index][1],my_colors[color_index][2], alpha=0.5)
        cf = ax.contourf(xx, yy,
                             Z_reshaped,
                             50, 
                             vmin = 0,
                             vmax = 1,
                             cmap=my_cmap, 
                            )
        ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               # cmap=ListedColormap(my_colors[color_index]),
               edgecolors='k', 
               s=100)
    thres = 0.5
    boundary_line_1 = np.where(((Z_reshaped[1:,:]>=thres)*(Z_reshaped[:-1,:]<=thres)))
    boundary_line_2 = np.where(((Z_reshaped[1:,:]<=thres)*(Z_reshaped[:-1,:]>=thres)))
    boundary_line_3 = np.where(((Z_reshaped[:,1:]<=thres)*(Z_reshaped[:,:-1]>=thres)))
    boundary_line_4 = np.where(((Z_reshaped[:,1:]>=thres)*(Z_reshaped[:,:-1]<=thres)))
    ax.scatter(x_domain[boundary_line_1[1]], y_domain[boundary_line_1[0]], color='k', alpha=0.5, s=1)
    ax.scatter(x_domain[boundary_line_2[1]], y_domain[boundary_line_2[0]], color='k', alpha=0.5, s=1)
    ax.scatter(x_domain[boundary_line_3[1]], y_domain[boundary_line_3[0]], color='k', alpha=0.5, s=1)
    ax.scatter(x_domain[boundary_line_4[1]], y_domain[boundary_line_4[0]], color='k', alpha=0.5, s=1)

    #boundary_line = np.where(np.abs(Z_reshaped-0.5)<0.001)
    
    #ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')
    #return Zaux

def generate_dataset(random_variables, random_seed=42):
    np.random.seed(random_seed)
    X = np.array([]).reshape(0, len(random_variables[0][0]))
    y = np.array([]).reshape(0, 1)
    for i, rv in enumerate(random_variables):
        X = np.vstack([X, np.random.multivariate_normal(rv[0], rv[1], rv[2])])
        y = np.vstack([y, np.ones(rv[2]).reshape(rv[2],1)*i]) 
    y = y.reshape(-1)
    return X, y

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, coefs_, intercepts_):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    ax.axis('off')
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =0.1, head_width=0.01, head_length=0.02)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                plt.text(left-0.125, layer_top - m*v_spacing, r'$X_{'+str(m+1)+'}$', fontsize=15)
            elif (n_layers == 3) & (n == 1):
                plt.text(n*h_spacing + left+0.00, layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing), r'$a_{'+str(m+1)+'}$', fontsize=15)
            elif n == n_layers -1:
                # plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, r' $\hat{p}_{'+str(m+1)+'}$=sigmoid($h_{'+str(m+1)+'}$)', fontsize=15)
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, r' $h_{'+str(m+1)+'}$', fontsize=15)
            ax.add_artist(circle)
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., color='w', ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), y_bias, r'$1$', fontsize=15)
            ax.add_artist(circle)   
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', lw=0.1)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                # print(n, m, o, str(coefs_[n][m, o]))
                plt.text( xm1, ym1,\
                         str(coefs_[n][m, o]),\
                         rotation = rot_mo_deg, \
                         fontsize = 12)
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], c='k', lw=0.1)
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
            plt.text( xo1, yo1,\
                 str(intercepts_[n][o]),\
                 rotation = rot_bo_deg, \
                 fontsize = 12)    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)