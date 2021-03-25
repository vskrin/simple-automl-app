"""
Flask app routes for data preparation.
"""

from flask import current_app as app
from flask import render_template, request, url_for, flash, redirect, make_response, jsonify
from . import bayesian_optimizer as bo
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import time
import copy

data_params = {
    'switch': 'true',
    'ncols': 100,
    'nrows': 5000,
    'train_ratio': 80,
    'tgt_ratio': 50,
    'ntrees': 100,
    'max_depth': 5,
    'min_samples_split': 100,
    'min_samples_leaf': 100,
    'max_features': 10
}
model_scores = {'prec': 0,
                'acc': 0,
                'rec': 0,
                'f1': 0
                }

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global data_params, model_scores
    # dataset preparation requested
    if request.method=="POST":
        try:
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
                print('Starting automatic model building.')
                #optimum = bo.optimize(train)
                #print("Optimal parameters (minimum): ", optimum.x)
                #print("Objective function at minimum: ", optimum.fun)
                # TODO: update data_params to include optimum RF parameters
            else:
                print('Proceeding with manual model building.')
            # TODO: build RF model with params in data_params and evaluate on test set
            
            response = make_response(( {'params': data_params,
                                        'scores': model_scores}, 
                                        200
                                    ))
            return response
        except:
            flash("Model building failed :(", "warning")

    return  render_template(
                    'home.html',
                    data_params=data_params
                    )