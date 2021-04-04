# Bayesian Optimization of Random Forests on Synthetic Datasets

## Overview
The goal of this document is too provide an overview of the app. You can try out the app [here](https://bayesian-optimization-mini-app.herokuapp.com/).

I'll begin by explaining what the app does, then I'll discuss the design, and finally the statistical methodology of its key components.

## Application workflow
The app consists of three parts stacked vertically.
The first "row" lets you prepare the data. The second one lets you prepare the model, and the third one gives you some further info on the dataset and the modeling.

In addition to this documentation, in every input form of the app you have access to further info through a help/info button.

### Data preparation
In the input part, you can tune a couple of dataset parameters:
* the number of predictive features
* the number of rows in the dataset
* the share of the target rows
* train-test split, i.e. the ratio in which the original dataset is split into training and test datasets

As you tune the input sliders the app will calculate and show the information about the dataset that will be generated.
All data is generated once you request the model to be built later on.

In addition to showing you the parameters that you've chosen, the app will calculate the number of target and non-target rows for each of the three datasets: train, test, and overall. You will also see the full size of the train and test sets (target and non-target rows together).

Once you've decided on the characteristics of the dataset, you move to the second part - modeling.

### Modeling
In the modeling part, you can choose between manual or automatic modeling. You do this with the "Automatic optimization" switch.

If the switch is turned on, you will not be able to tune any of the model parameters. In this case you may proceed to click "Build" and the app will take over the modeling.

If the switch is turned off, you will be able to tune the following parameters of the Random Forest model:
* *number of estimators*: this sets the number of scikit-learn decision tree classifiers in the random forest model.
* *max tree depth*: an upper bound on the allowed depth of each decision tree classifier. This is the first of the three parameters used to prevent overfitting.
* *min node size*: a lower bound on the allowed size of the nodes of decision tree classifiers. It is used to prevent the tree from creating child nodes which isolate outliers.
* *min leaf size*: similar to the previous parameter, this one applies to the leaf nodes. It prevents node splitting if it would result in leafs smaller than this bound.
* *max features used per tree*: each estimator has access to a random subset of the original set of features. This parameter controls the number of features to which each tree has access to. The smaller this number, the more uncorrelated estimators are going to be.

Note that input sliders minimum and maximum allowed value for some of the above parameters is going to depend on the number of features of the dataset. The application will automatically set up the limits as you modify the dataset.

Once you've clicked "Build" the application sends request to the backed which generates the dataset and builds the model. If you've provided the parameters manually, one model will be built. The model will be trained on the training set, and then it'll be evaluated on the train and the test set. Finally, model details and model scores will be presented in the modeling part of the application. The app should not take more than a few seconds to return the results to you.

If you've selected "Automatic optimization", the application will use Bayesian Optimization technique based on Gaussian Processes and implemented via [scikit-optimize](https://scikit-optimize.github.io/stable/) library to find the optimal model. More details on this will be provided in the last section. Note that automatic optimization relies on creation of many datasets and models. Because of this, the waiting time for the application to return results to you may be as long as 30 seconds.

When the app returns the results of the model, it will also provide some additional info which you can see in the last part of the app.

### Additional info on dataset and modeling

If the model was built automatically, the part on "Further info" will contain a figure of the dataset and a figure of the optimization process. If the model was built manually, only the figure of the dataset will be provided.

In most cases the dataset will have more than two predictive features, so in order to provide some intuition about the data, the app uses Principal Component Analysis (aka PCA) to plot the dataset in two dimensions. 

## Application design

## Stats (methodology)