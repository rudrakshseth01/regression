from flask import Flask,jsonify,request,render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app=application

# import ridge and scaler
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
std_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
          if request.method=='POST':
                    Tempratue=float(request.form.get('Temp'))
                    RH=float(request.form.get('RH'))
                    Ws=float(request.form.get('Ws'))
                    RAIN=float(request.form.get('RAIN'))
                    FFMC=float(request.form.get('FFMC'))
                    DMC=float(request.form.get('DMC'))
                    ISI=float(request.form.get('ISI'))
                    Classes=float(request.form.get('Classes'))
                    Region=float(request.form.get('Region'))
                    
                    new_data_scaled=std_scaler.transform([[Tempratue,RH,Ws,RAIN,FFMC,DMC,ISI,Classes,Region]])
                    result=ridge_model.predict(new_data_scaled)
                    
                    return render_template('home.html',results=result[0])
                    
          else:
                    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
