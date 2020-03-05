from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
 
mul_reg = open("new_predict_final.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('CTM_Average'))
        try:
            CTM_Average = float(request.form['CTM_Average'])
            CTM_Bad = float(request.form['CTM_Bad'])
            CTM_Good = float(request.form['CTM_Good'])
            CA_Average = float(request.form['CA_Average'])
            CA_Bad = float(request.form['CA_Bad'])
            CA_Good= float(request.form['CA_Good'])
            AS_No = float(request.form['AS_No'])
            AS_Yes = float(request.form['AS_Yes'])
            LP_Average = float(request.form['LP_Average'])
            LP_Bad = float(request.form['LP_Bad'])
            LP_Good = float(request.form['LP_Good'])
            FES_No = float(request.form['FES_No'])
            FES_Yes = float(request.form['FES_Yes'])
            R_In_Dhaka_City = float(request.form['R_In_Dhaka_City'])
            R_Outside_Dhaka_City = float(request.form['R_Outside_Dhaka_City'])
            SH = float(request.form['SH'])
            SM = float(request.form['SM'])
            ECA_No = float(request.form['ECA_No'])
            ECA_Yes = float(request.form['ECA_Yes'])
            DA_No = float(request.form['DA_No'])
            DA_Yes = float(request.form['DA_Yes'])
            RS_NO = float(request.form['RS_NO'])
            RS_Yes = float(request.form['RS_Yes'])
            FS_No = float(request.form['FS_No'])
            FS_Yes = float(request.form['FS_Yes'])
            FcS_No = float(request.form['FcS_No'])
            FcS_Yes = float(request.form['FcS_Yes'])
            PI_No = float(request.form['PI_No'])
            PI_Yes = float(request.form['PI_Yes'])
            RWF_Average = float(request.form['RWF_Average'])
            RWF_Bad = float(request.form['RWF_Bad'])
            RWF_Good = float(request.form['RWF_Good'])
            JTF_No = float(request.form['JTF_No'])
            JTF_Yes = float(request.form['JTF_Yes'])
            LS = float(request.form['LS'])

            pred_args = [CTM_Average,CTM_Bad,CTM_Good,CA_Average,CA_Bad,CA_Good,AS_No,AS_Yes,LP_Average,LP_Bad,LP_Good,
            FES_No,FES_Yes,R_In_Dhaka_City,R_Outside_Dhaka_City,SH,SM,ECA_No,ECA_Yes,DA_No,DA_Yes,RS_NO,RS_Yes,
            FS_No,FS_Yes,FcS_No,FcS_Yes,PI_No,PI_Yes,RWF_Average,RWF_Bad,RWF_Good,JTF_No,JTF_Yes,LS]  #17

#CTM_Average,CTM_Bad,CTM_Good,CA_Average,CA_Bad,CA_Good,AS_No,AS_Yes,LP_Average,LP_Bad,LP_Good,FES_No,FES_Yes,
#R_In_Dhaka_City,R_Outside_Dhaka_City,SH,SM,ECA_No,ECA_Yes,DA_No,DA_Yes,RS_NO,RS_Yes,FS_No,FS_Yes,FcS_No,FcS_Yes,
#PI_No,PI_Yes,RWF_Average,RWF_Bad,RWF_Good,JTF_No,JTF_Yes,LS


            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            # mul_reg = open("multiple_regression_model.pkl", "rb")
            # ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_pred = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
