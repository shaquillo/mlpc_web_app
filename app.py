from flask import Flask, url_for, render_template, request
from model import predict
import logging

app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    pred_result = predict.predict(request.form)
    return render_template('result.html', pred_result=pred_result)

if __name__ == '__main__':
    app.run()