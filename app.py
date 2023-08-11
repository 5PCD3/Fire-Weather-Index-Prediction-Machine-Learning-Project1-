import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
app = Flask(__name__)
# Storing Data in Mongodb Database






# Importing ridge regressor model and standard scaler pickle

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# Connect to MongoDB
client = MongoClient("mongodb+srv://pcd:mypassword@cluster0.1oyubnl.mongodb.net/?retryWrites=true&w=majority")  # Replace with your MongoDB URI
db = client['Fire_Weather_Index_Prediction_Database']  # Replace with your database name
Weather_Parameters = db['Weather_Parameters']  # Collection for user-entered data
Predictions = db['Predictions']  # Collection for ML predictions

#@app.route('/')
#def index():
#    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))
        #data_list = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
        #keys = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]
        
       
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        result = ridge_model.predict(new_data_scaled)
        
        # Store user-entered data
        user_data = {
            "Temperature": Temperature,
            "RH": RH,
            "Ws": Ws,
            "Rain": Rain,
            "FFMC": FFMC,
            "DMC": DMC,
            "ISI": ISI,
            "Classes": Classes,
            "Region": Region
        }
        Weather_Parameters.insert_one(user_data)


         # Store prediction data
        prediction_data = {
            "Temperature": Temperature,
            "RH": RH,
            "Ws": Ws,
            "Rain": Rain,
            "FFMC": FFMC,
            "DMC": DMC,
            "ISI": ISI,
            "Classes": Classes,
            "Region": Region,
            "FWI_Prediction": result[0]
        }
        Predictions.insert_one(prediction_data)


        return render_template('home.html', result=result[0])
        
    else:
        return render_template('home.html')







if __name__=="__main__":
    app.run(host="0.0.0.0")