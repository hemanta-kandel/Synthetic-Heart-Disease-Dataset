from flask import Flask, render_template
from model_training import disease_chance

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def About():
    return render_template('About.html')

@app.route('/')
def gohome():
    return render_template('index.html')

@app.route('/')
def model():
    return render_template('Model.html')
