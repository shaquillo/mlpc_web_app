from os import path
import tensorflow as tf
import numpy as np

model_dir = path.join(path.dirname(__file__), 'model_092_089.h5')
model = tf.keras.models.load_model(model_dir)

def predict(data):
    to_pred = np.char.strip(np.array([data['INF212'], data['MATH217'], data['INF222'], data['PHY223'], data['PHY224'], data['PHY225'], data['MATH226'], data['MATH227'], data['PHY213'], data['PHY215'], data['PHY214'], data['MATH216'], data['MATH218'], data['PHY228'], data['MATH_INF211'], data['PHY211'], data['PHY221'], data['PHY222'], data['MATH_INF221'], data['PHY111'], data['MATH_INF121'], data['MATH_INF111'], data['DES124'], data['PHY121'], data['PHY112'], data['MATH212'], data['MATH122']])).reshape(1, 27).astype(np.float)
    prediction = model.predict(to_pred)

    best3Index = np.argsort(prediction, axis=1)[0][::-1]
    result = best3Index.tolist()
    return result[:3]
