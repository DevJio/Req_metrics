from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file
from flask import json
from flask_cors import CORS


import joblib
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os
import re
import metric_utils as utl

app = Flask(__name__)
CORS(app)

conf_matrix = 'any matrix'
cl_report = 'any report'

@app.route('/confusion_matrix')
def confusion_matrix():
        return conf_matrix

@app.route('/classification_report')
def classification_report():
        return cl_report


@app.route('/')
def hello_world():
    print("go! go! go!")
    return "<h1>test post service by json on /metrics {id:[ids], machineCategory': ['0', '1'], finalCategory: ['0', '2, 3']</h1>" 

@app.route('/badrequest400')
def bad_request():
    return abort(400)


@app.route('/metrics', methods=['POST'])
def metrics_calc():
    #try:
    content = request.get_json()
    data = pd.DataFrame(content)
    data.columns = ['id', 'predict', 'type_id']
    predict = data.predict
    expert = utl.prepateExpertCol(data.type_id)

    conf_matrix = utl.make_conf_matrix(expert, predict)
    cl_report = utl.make_cl_report(expert, predict)

    response = app.response_class(response='{"confusion_matrix":'+conf_matrix.to_json()+ ',"metrics":' + cl_report.to_json()+'}', 
        status=200, 
        mimetype='application/json')
    return response
    #except:
    #    return redirect(url_for('bad_request'))
    


app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))
