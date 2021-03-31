"""
Flask app routes.
"""
from flask import current_app as app
from flask import render_template, request, make_response
from . import bayesian_optimizer as bo
import numpy as np
import os
import json
from time import perf_counter

data_params = {
    'switch': 'true',
    'ncols': 10,
    'nrows': 250,
    'train_ratio': 80,
    'tgt_ratio': 50,
    'ntrees': 5,
    'max_depth': 5,
    'min_samples_split': 50,
    'min_samples_leaf': 50,
    'max_features': 3
}
train_score = {'auc': 0,
            'prec': 0,
            'acc': 0,
            'rec': 0,
            'f1': 0}
test_score = {'auc': 0,
            'prec': 0,
            'acc': 0,
            'rec': 0,
            'f1': 0}

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global data_params, train_score, test_score
    # dataset preparation requested
    if request.method=="POST":
        time = perf_counter()
        for key in data_params.keys():
            try:
                data_params[key]=int(request.json[key])
            except:
                data_params[key]=request.json[key]
        train_x, test_x,\
        train_y, test_y = bo.get_data(  
                                ncols=data_params['ncols'], 
                                nrows=data_params['nrows'], 
                                train_ratio=data_params['train_ratio'], 
                                tgt_ratio=data_params['tgt_ratio']
                                )
        if data_params['switch']=='true':
            print('Searching for optimal parameters.')
            optimum, progress_data = bo.optimize(train_x, train_y, data_params)
            print('Optimal parameters determined: ', optimum.x)
            print('Objective function (f1-score) at minimum: ', -optimum.fun)
            data_params.update({ #can't put Int64 in JSON -> use 32bit int
                'ntrees': int(optimum.x[0]),
                'max_depth': int(optimum.x[1]),
                'min_samples_split': int(optimum.x[2]),
                'min_samples_leaf': int(optimum.x[3]),
                'max_features': int(optimum.x[4])
            })
        else:
            progress_data = "none"
        print('Proceeding with model building.')
        train_score, test_score = bo.build_model(data_params, 
                                                train_x, train_y, 
                                                test_x, test_y
                                                )
        pca_data = bo.plot_pca(train_x, train_y)
        print('Model built. Returning model scores.')
        print('train score: ', train_score)
        print('test score: ', test_score)
        response = make_response(( {'data_params': data_params,
                                    'train_score': train_score,
                                    'test_score': test_score,
                                    'pca_data': pca_data,
                                    'progress_data': progress_data}, 
                                    200
                                ))
        print(f"It took {perf_counter()-time:.2f} sec to process the request.")
        return response


    return  render_template(
                    'home.html',
                    data_params=data_params
                    )