from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Azoto']
    P = request.form['Fosforo']
    K = request.form['Potassio']
    temp = request.form['Temperatura']
    humidity = request.form['Umidita']
    ph = request.form['pH']
    rainfall = request.form['Precipitazioni']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Riso", 
                 2: "Mais", 
                 3: "Juta", 
                 4: "Cotone", 
                 5: "Cocco", 
                 6: "Papaia", 
                 7: "Arancio",
                 8: "Mele", 
                 9: "Melone", 
                 10: "Cocomero", 
                 11: "Uva", 
                 12: "Mango", 
                 13: "Banane",
                 14: "Melograno", 
                 15: "Lenticchia (Legume)", 
                 16: "Fagiolo Mungo Nero (Legume)", 
                 17: "Fagiolo Mungo Verde (Legume)", 
                 18: "Fagiolo Moth (Legume)",
                 19: "Fagiolo dall'occhio (Legume)", 
                 20: "Fagiolo Rosso (Legume)", 
                 21: "Cece (Legume)", 
                 22: "Caffè"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{}".format(crop)
    else:
        result = "Siamo spiacenti, ma non siamo riusciti a determinare la migliore coltura da coltivare con i dati forniti."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)