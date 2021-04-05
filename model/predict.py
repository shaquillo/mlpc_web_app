from os import path
from joblib import load
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb


model1_dir = path.join(path.dirname(__file__), 'model1_1.pkl')
model2_dir = path.join(path.dirname(__file__), 'model2_2')
model3_dir = path.join(path.dirname(__file__), 'model3_3.txt')
model1 = load(model1_dir)
model2 = CatBoostClassifier()
model2.load_model(fname=model2_dir)
model3 = lgb.Booster(model_file=model3_dir)

def preprocess_data(data):
    cat_cols = ['INF212', 'MATH217', 'INF222', 'PHY223', 'PHY224', 'PHY225', 'MATH226', 'MATH227', 'PHY213', 'PHY215', 'PHY214', 'MATH216', 'MATH218', 'PHY228', 'MATH_INF211', 'PHY211', 'MATH212', 'PHY221', 'PHY222', 'MATH_INF221', 'PHY111', 'MATH_INF121', 'MATH_INF111', 'MATH122', 'DES124', 'PHY121', 'PHY112']
    data1 = {}
    for col in cat_cols:
        data1[col] = float(data[col])
        if 0 <= data1[col] <= 5:
            data1[col] = 1
        elif 5 < data1[col] <= 8:
            data1[col] = 2
        elif 8 < data1[col] <= 10:
            data1[col] = 3
        elif 10 < data1[col] <= 12:
            data1[col] = 4
        elif 12 < data1[col] <= 14:
            data1[col] = 5
        elif 14 < data1[col] <= 16:
            data1[col] = 6
        elif 16 < data1[col] <= 18:
            data1[col] = 7
        elif 18 < data1[col] <= 20:
            data1[col] = 8
        data1[col] = int(data1[col])

    data1['av_INF'] = (float(data['INF212']) + float(data['INF222']))/2
    data1['av_MATH'] = (float(data['MATH217']) +  float(data['MATH226']) + float(data['MATH227']) + float(data['MATH216']) + float(data['MATH218']) + float(data['MATH_INF211']) + float(data['MATH212']) + float(data['MATH_INF221']) + float(data['MATH_INF121']) + float(data['MATH_INF111']) + float(data['MATH122'])) / 11
    data1['av_PHY'] = (float(data['PHY223']) +  float(data['PHY224']) +  float(data['PHY225']) +  float(data['PHY213']) +  float(data['PHY215']) +  float(data['PHY214']) +  float(data['PHY228']) +  float(data['PHY211']) +  float(data['PHY221']) +  float(data['PHY222']) +  float(data['PHY111']) +  float(data['DES124']) +  float(data['PHY121']) +  float(data['PHY112'])) / 14
    data1['av'] = (data1['av_INF'] + data1['av_MATH'] + data1['av_PHY']) /3

    
    return [[data1['INF212'], data1['MATH217'], data1['INF222'], data1['PHY223'], data1['PHY224'], data1['PHY225'], data1['MATH226'], data1['MATH227'], data1['PHY213'], data1['PHY215'], data1['PHY214'], data1['MATH216'], data1['MATH218'], data1['PHY228'], data1['MATH_INF211'], data1['PHY211'], data1['PHY221'], data1['PHY222'], data1['MATH_INF221'], data1['PHY111'], data1['MATH_INF121'], data1['MATH_INF111'], data1['DES124'], data1['PHY121'], data1['PHY112'], data1['MATH212'], data1['MATH122'], data1['av_INF'], data1['av_MATH'], data1['av_PHY'], data1['av']]]

def predict(data):
    to_pred = preprocess_data(data)
    result = []
    result.append(model1.predict(to_pred)[0])
    prediction2 = model2.predict(to_pred)[0]
    print('===============', prediction2)

    if prediction2 not in result:
        result.append(prediction2)

    prediction3 = model3.predict(to_pred)[0]

    i = np.argmax(prediction3)
    if i == 0:
        pred3 = 'GM'
    elif i == 1:
        pred3 = 'GELE'
    elif i == 2:
        pred3 = 'GI'
    elif i == 3:
        pred3 = 'GIND'
    elif i == 4:
        pred3 = 'GC'
    else :
        pred3 = 'GTEL'

    if pred3 not in result:
        result.append(pred3)
    print('---------',result)

    return result
