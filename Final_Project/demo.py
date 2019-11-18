import flask
import pickle
import pandas as pd
from flask import Flask, jsonify, render_template

app = flask.Flask(__name__, template_folder='templates')

model1 = pickle.load(open("model1.pkl","rb"))
model1._make_predict_function()
model2 = pickle.load(open("model2.pkl","rb"))
model3 = pickle.load(open("model3.pkl","rb"))
model4 = pickle.load(open("model4.pkl","rb"))
model5 = pickle.load(open("model5.pkl","rb"))
@app.route("/")
def index():
    """Return the homepage."""
    
    return render_template("index.html")

@app.route('/ETL')
def ETL():
   return render_template("ETL.html")


@app.route('/MPL')
def MPL():
   return render_template("MPL.html")

@app.route('/Models')
def Models():
   return render_template("Models.html")



@app.route('/Demo', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('/Demo.html'))
    if flask.request.method == 'POST':
        age = flask.request.form['age']
        C = flask.request.form['C']
        Cr = flask.request.form['Cr']
        Mn = flask.request.form['Mn']
        P = flask.request.form['P']
        input_variables = pd.DataFrame([[age, C, Cr, Mn, P]], columns=['age', 'C', 'Cr', 'Mn', 'P'], dtype=float)
        pred = (model1.predict(input_variables)[0] + model2.predict(input_variables)[0] + model3.predict(input_variables)[0] + model4.predict(input_variables)[0] + model5.predict(input_variables)[0])/5

        return flask.render_template('/index1.html',
                                     original_input={'age': age,
                                                     'C': C,
                                                     'Cr': Cr,
                                                     'Mn': Mn,
                                                     'P': P},
                                     result=pred,)

if __name__ == '__main__':
    app.run()