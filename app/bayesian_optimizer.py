from skopt import Space, gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy
import pandas

# global variables
param_grid = [  Integer(2,10, name="max_depth"),
                Integer(20, 250, name="min_samples_split"),
                Integer(10, 100, name="min_samples_leaf")
                ]
tree = DecisionTreeClassifier()
train_data = []
valid_data =  []
n_resamples = 50
n_steps = 20

# Bayesian optimization via Gaussian process
def optimize(data):
    global train_data, valid_data
    train_data = get_datasets(data, n_resamples)
    valid_data = get_datasets(data, n_resamples)
    result = gp_minimize(   
                    objective, 
                    param_grid,
                    acq_func="EI",
                    n_initial_points=5,
                    n_calls=n_steps, 
                    n_jobs=-1,
                    random_state=0,
                    verbose=False
                    )
    return result

# objective function - minimized for the best model
@use_named_args(param_grid)
def objective(**params):
    tree.set_params(**params)
    scores = score_models(tree, train_data, valid_data, metric="rec")
    return -numpy.median(scores)

# get Bootstrap resamples of the training set
def get_datasets(data, n_resamples):
    datasets = []
    for i in range(n_resamples):
        new_resample = resample(
                            data, 
                            replace=True,
                            stratify=data['Target']
                            )
        datasets.append(new_resample)
    return datasets

#score model on datasets
def score_models(tree, train, validation, metric="f1"):
    """
    Takes DecisionTreeClassifier, a list of training sets, a list of validation sets,
    and a scoring metric. Fits classifiers on training sets, and scores them on validation.
    Returns median score.
    Args:
        * tree: sklearn.DecisionTreeClassifier
        * train: list of pandas.Dataframes. Same length as validation.
        * validation: list of pandas.Dataframes. Same length as train.
        * metric: string. Can be "prec", "acc", "rec", "f1", or "auc" for 
        precision, accuracy, recall, f1-score, and area under curve, respectively.
    Returns:
        * median model score (over validation datasets)
    """
    scores = []
    metric_dict = { 'prec': precision_score,
                    'acc': accuracy_score,
                    'rec': recall_score,
                    'f1': f1_score
                    }
    try:
        n_resamples = len(train)
        for j in range(n_resamples):
            tree.fit(train[j].drop(columns="Target"),
                    train[j]["Target"]
            )
            if metric=="auc":
                predicted_proba = tree.predict_proba(validation[j].drop(columns="Target"))[:,1]
                fpr, tpr, _ = roc_curve(validation[j]["Target"], predicted_proba)
                score = auc(fpr,tpr)
            else:
                predicted_labels = tree.predict(train[j].drop(columns="Target"))
                score = metric_dict[metric](validation[j]["Target"], predicted_labels)
            if score==numpy.nan:
                scores.append(0)
            else:
                scores.append( numpy.round(100*score, 4) )
        return scores
    except:
       print("Error while attempting model validation.")

# build dataset
def get_data(ncols:int, nrows:int, train_ratio:int, tgt_ratio:int):
    '''
    Takes parameters of the dataset and builds synthetic data. 
    Returns two datasets: train and test.
    Args:
        * ncols: number of columns
        * nrows: number of rows
        * train_ratio: share of train rows for the train-test split
        * tgt_tatio: target share, percentage of target rows
    Returns:
        * train predictors and target, and test predictors and target: numpy arrays 
        Let train factor be trf=train_ratio/100, and test factor be tef=(100-train_ratio)/100.
        Shapes of resulting datasets are ncols x (nrows*trf), 1x(nrows*trf), ncols x (nrows*tef), and 1x(nrows*tef)
    '''
    # prepare distributions
    predictor_means = numpy.random.uniform(-1,1,ncols)
    error_means = numpy.full(ncols, 0)
    # covariances are symmetric and positive semi-definite
    random_matrix = numpy.random.rand(ncols, ncols)-1/2
    predictor_covariance = numpy.dot(random_matrix, random_matrix.transpose()) 
    random_matrix = numpy.random.rand(ncols, ncols)-1/2
    error_covariance = numpy.dot(random_matrix, random_matrix.transpose())
    # generate data
    tgt_length_goal = int(nrows*tgt_ratio/100)
    non_tgt_length_goal = int(nrows*(100-tgt_ratio)/100)
    data_length, tgt_length, non_tgt_length = 0, 0, 0
    predictors = [] # avoiding numpy.append(); using native predictors.append() instead
    target = numpy.array([])
    # generate target and non-target samples until enough of each is generated
    while data_length<nrows:
        X = numpy.random.multivariate_normal(mean=predictor_means, 
                                            cov=predictor_covariance, 
                                            size=1
                                            )[0]+\
                numpy.random.multivariate_normal(mean=error_means,
                                                cov=error_covariance,
                                                size=1)[0]
        y = numpy.sum(X)
        y = (y>0.5).astype(int)
        if (y>0.5) and tgt_length<tgt_length_goal:
            predictors.append(X)
            target = numpy.append(target, y)
            data_length+=1
            tgt_length+=1
        elif (y<=0.5) and non_tgt_length<non_tgt_length_goal:
            predictors.append(X)
            target = numpy.append(target, y)
            data_length+=1
            non_tgt_length+=1
    predictors = numpy.array(predictors)
    x_train, x_test, y_train, y_test = train_test_split(predictors, 
                                                        target, 
                                                        train_size=train_ratio/100,
                                                        stratify=target
                                                        )
    return x_train, x_test, y_train, y_test