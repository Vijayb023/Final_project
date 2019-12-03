import pickle
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template

app = Flask (__name__)

# app = flask.Flask(__name__, template_folder='templates')

model_SVC = pickle.load(open("pickle_SVC.pkl","rb"))
model_KNN = pickle.load(open("pickle-KNN_unscaled.pkl","rb"))

# model_logreg = pickle.load(open("pickle_LogReg.pkl","rb"))

# model_SVC._make_predict_function()

@app.route('/')  # methods=['GET', 'POST']
def main():
    # if flask.request.method == 'GET':
    #     return(flask.render_template('/index1.html'))
    # if flask.request.method == 'POST':
        new_X_test_data = np.array([[50.0, 66.0, 136.0, 110, 80, 21.948577, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]])
        # Predicting using SVC model and saving it to dictionary
        SVC_pred = model_SVC.predict(new_X_test_data)
        SVC_pred_list = SVC_pred.tolist()


        KNN_pred = model_KNN.predict(new_X_test_data)
        KNN_pred_list = KNN_pred.tolist()

        # result_df = pd.DataFrame(pred)
        result = {"SVC_Prediction": SVC_pred_list, "SVC_AUROC_Score": "79.5%",\
                  "KNN_Prediction": KNN_pred_list, "KNN_AUROC_Score": "80.7%"}


        return jsonify(result)

        # return jsonify(result_df.to_dict(orient="list"))
if __name__ == '__main__':
    app.run()