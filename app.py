from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')
