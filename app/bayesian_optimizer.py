from skopt import Space, gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA
from time import perf_counter
import numpy
import plotly.express as px
import plotly

# global variables
# param_grid is set up in optimize(). instantiated here to give parameter names for objective().
param_grid = [Integer(0, 1, name="n_estimators"),
            Integer(0, 1, name="max_depth"),
            Integer(0, 1, name="min_samples_split"),
            Integer(0, 1, name="min_samples_leaf"),
            Integer(0, 1, name="max_features")
            ]
model = RandomForestClassifier()
# lists of resampled train and test datasets
train_x, train_y = [], []
valid_x, valid_y = [], []
n_resamples = 20
n_steps = 15

# Bayesian optimization via Gaussian process
def optimize(features, target, data_params):
    """
    Main access point to scikit-optimize Bayesian optimization functionality.
    Args:
        * features: numpy array of predictive features
        * target: numpy array of class labels
        * data_params: dictionary of dataset parameters
    Returns:
        * result: scikit-optimize OptimizeResult object
        * progress_data: 2d array of best (objective function) results until given step
    """
    global param_grid, train_x, train_y, valid_x, valid_y
    train_x, train_y = get_datasets(features, target, n_resamples)
    valid_x, valid_y = get_datasets(features, target, n_resamples)
    param_grid = [Integer(2,data_params['ncols'], name="n_estimators"),
                Integer(2,10, name="max_depth"),
                Integer(10, data_params['nrows']//2, name="min_samples_split"),
                Integer(5, data_params['nrows']//3, name="min_samples_leaf"),
                Integer(1, data_params['ncols']//3, name="max_features")
                ]
    time = perf_counter()
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
    print(f"Parameter search took {perf_counter()-time:.3f}s to complete.")
    progress_data = plot_opti_progress(result.func_vals)
    return result, progress_data

# objective function - minimized for the best model
@use_named_args(param_grid)
def objective(**params):
    """
    For given choice of parameters it trains multiple models on bootstrap
    resamples of the train set. It then evaluates them on other resamples
    (validation) and scores them. Average score on validation is taken as
    value of the objective function for the given parameter choice.
    """
    model.set_params(**params)
    scores = score_bumpers(model, train_x, train_y, valid_x, valid_y, metric="f1")
    return -numpy.median(scores)

def get_datasets(features, target, n_resamples):
    """
    Produces bootstrap resamples of the original dataset.
    Args:
        * features: original numpy array of predictive features
        * target: original numpy array of target labels
        * n_resamples: number of new resampled datasets to create
    Returns:
        * feat_resamples, tgt_resamples: lists of new features/target 
        resampled datasets
    """
    feat_resamples, tgt_resamples = [], []
    for i in range(n_resamples):
        feat_resample, tgt_resample = resample(
                                        features, target, 
                                        replace=True,
                                        stratify=target
                                        )
        feat_resamples.append(feat_resample)
        tgt_resamples.append(tgt_resample)
    return feat_resamples, tgt_resamples


def score_bumpers(model, train_x, train_y, valid_x, valid_y, metric="f1"):
    """
    Takes scikit-learn classifier, a list of training sets, a list of validation sets,
    and a scoring metric. Fits classifiers on the training sets, and scores them on validation.
    Returns a list of scores.
    Args:
        * model: scikit-learn classifier
        * train_x, train_y: lists of numpy arrays with training data. Same length as validation.
        * valid_x, valid_y: lists of numpy arrays with validation data. Same length as train.
        * metric: string. Can be "prec", "acc", "rec", "f1", or "auc" for 
        precision, accuracy, recall, f1-score, and area under curve, respectively.
    Returns:
        * scores: list of model scores on validation
    """
    scores = []
    metric_dict = { 'prec': precision_score,
                    'acc': accuracy_score,
                    'rec': recall_score,
                    'f1': f1_score
                    }
    try:
        n_resamples = len(train_x)
        for j in range(n_resamples):
            model.fit(train_x[j], train_y[j])
            if metric=="auc":
                predicted_proba = model.predict_proba(valid_x[j])[:,1]
                fpr, tpr, _ = roc_curve(valid_y[j], predicted_proba)
                score = auc(fpr,tpr)
            else:
                predicted_labels = model.predict(valid_x[j])
                score = metric_dict[metric](valid_y[j], predicted_labels)
            if score==numpy.nan:
                scores.append(0)
            else:
                scores.append( numpy.round(100*score, 4) )
        return scores
    except:
       print("Error while attempting model validation.")


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


def build_model(params, train_x, train_y, test_x, test_y):
    """
    Args:
        * params: a dictionary of Random Forest parameters
        * train_x, train_y, test_x, test_y: train and test set predictors (x) and targets (y)
    Returns:
        * scores: a dict of model scores on train and test sets
    """
    RF = RandomForestClassifier(
                        n_estimators=params['ntrees'], 
                        max_depth=params['max_depth'],
                        min_samples_split=params['min_samples_split'],
                        min_samples_leaf=params['min_samples_leaf'],
                        max_features=params['max_features']
                        )
    RF.fit(train_x, train_y)
    train_scores = get_scores(RF, train_x, train_y)
    test_scores = get_scores(RF, test_x, test_y)
    return train_scores, test_scores


def get_scores(model, features, target):
    """
    Scores classifier by comparing predictions from features and the target label.
    Args:
        * model: scikit-learn classifier
        * features: numpy array of predictive features
        * target: numpy array of corresponding true target labels
    Returns:
        * scores: dictionary of model scores on the provided dataset
    """
    metric_dict = { 'prec': precision_score,
                    'acc': accuracy_score,
                    'rec': recall_score,
                    'f1': f1_score
                    }
    scores = {}
    predicted_proba = model.predict_proba(features)[:,1]
    fpr, tpr, _ = roc_curve(target, predicted_proba)
    scores['auc'] = auc(fpr,tpr)
    predictions = model.predict(features)
    for metric in metric_dict.keys():
        score = metric_dict[metric](target, predictions)
        if score==numpy.nan:
            scores[metric]=0
        else:
            scores[metric]=numpy.round(100*score, 4)
    return scores

def plot_pca(data, labels):
    """
    Takes numpy array, performs PCA analysis, and plots two principal components.
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    train_pca = pca.transform(data)
    plot_data = [[pca[0], pca[1], int(label)] for pca, label in zip(train_pca, labels)]
    return plot_data

def plot_opti_progress(objective_values):
    """
    Takes list of objective function values. Returns list of best scores until given step.
    """
    optimum = 0
    result = []
    for n, el in enumerate(objective_values):
        if -el>optimum:
            result.append([n+1,-el])
            optimum = -el
        else:
            result.append([n+1,optimum])
    return result

if __name__=="__main__":

    ## Mock data parameters to use in testing
    data_params = {
        'switch': 'true',
        'ncols': 6,
        'nrows': 30,
        'train_ratio': 80,
        'tgt_ratio': 50,
        'ntrees': 5,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 1
    }
    ## Test dataset creation
    x_train, x_test,\
    y_train, y_test = get_data(ncols=data_params['ncols'], 
                            nrows=data_params['nrows'], 
                            train_ratio=data_params['train_ratio'], 
                            tgt_ratio=data_params['tgt_ratio'])
    # print("x_train:", x_train)
    # print("y_train:", y_train)

    ### Test dataset resampling
    # train_feats, traing_tgt = get_datasets(x_train, y_train, 4)
    # valid_feats, valid_tgt = get_datasets(x_train, y_train, 4)
    # print("train resamples:", train_feats, traing_tgt)
    
    ### Test model building
    # RF = RandomForestClassifier(n_estimators=5, max_features=1)
    # scores = score_bumpers(RF, train_feats, traing_tgt, valid_feats, valid_tgt)
    # print("Validation scores: ", scores)

    ## Test optimizer
    optimum = optimize(x_train, y_train, data_params)
    print("Optimal parameters (minimum): ", optimum.x)
    print("Objective function at minimum: ", optimum.fun)
    print("Function values for each iter:", optimum.func_vals)
    ## Test PCA plotting
    # fig = plot_pca(x_train, y_train)
    # print(fig)
    print("best score until given step: ", plot_opti_progress(optimum.func_vals))


    
