import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
import bayesian_optimizer as bo
from skopt.plots import plot_convergence, plot_gaussian_process, plot_objective_2D


# functions used to prepare dataset
def make_chessboard(N=1000,
                    xbins=(0.,0.5,1.),
                    ybins=(0.,0.5,1.)):
    """Chessboard pattern data"""
    X = np.random.uniform(size=(N,2))
    xcategory = np.digitize(X[:,0], xbins)%2
    ycategory = np.digitize(X[:,1], ybins)%2
    y = np.logical_xor(xcategory, ycategory)
    y = np.where(y, -1., 1.)
    return X,y

def plot_data(X,y, is_bumper=False):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    if is_bumper:
        ax.set_title("Bumper results")
    else:
        ax.set_title("Simple tree results")
    return (fig, ax)

def draw_decision_regions(X, y, estimator, resolution=0.01, is_bumper=False):
    """Draw samples and decision regions
    
    The true label is indicated by the colour of each
    marker. The decision tree's predicted label is
    shown by the colour of each region.
    """
    plot_step = resolution
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, axis = plot_data(X,y, is_bumper=is_bumper)
    axis.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
    plt.show()

# generate problem (data)
X, y = make_chessboard(N=1000, xbins=(0,0.25,.75,1), ybins=(0,0.33,.66,1))
p = plot_data(X, y)
plt.show()

df = pd.DataFrame(X, columns=["x", "y"])
df["Target"] = y

result = bo.optimize(df)
print("Optimal parameters (minimum): ", result.x)
print("Objective function at minimum: ", result.fun)

plot_convergence(result)
plt.show()

plot_objective_2D(result, "max_depth", "min_samples_split")
plt.show()





# #plot decision region
# with sns.axes_style('white'):
#     draw_decision_regions(X, y, simple_tree, is_bumper=False)
# #score simple tree
# score, _, _ = bumper.score_model(model=simple_tree, 
#                                 features=X, 
#                                 target=y)
# print("Simple tree score:\n", score)
