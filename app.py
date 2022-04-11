from flask import Flask, render_template, request
from datetime import datetime
import pandas as pd, numpy as np
import requests, io, base64

import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from markupsafe import Markup

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("home.html")

@app.get("/get_statewise_input_data")
def get_statewise_input_data():
    return render_template("get_statewise_input.html")

@app.get("/get_country_input_data")
def get_country_input_data():
    return render_template("get_country_input.html")

@app.post("/display_state_data")
def display_state_data():
    
    state=request.form.get("state")
    days=int(request.form.get("days"))
    
    df = pd.read_csv(r"datasets/COVID-19 Cases statewise.csv",parse_dates=True)
    df1 = df.copy()
    df1.drop("S. No.",axis=1,inplace=True)
    df1 = df1.drop(df1[df1.Region == "India"].index)
    df1 = df1.drop(df1[df1.Region == "World"].index)
    df1 = df1.drop(df1[df1.Region == "State assignment pending"].index)
    df1.reset_index(inplace = True,drop=True)
    df_new = df1[df1["Region"]==state]
    df_new.set_index('Date',drop=True,inplace=True)
    df_new.drop(["Region"],axis=1,inplace=True)
    df_Res = df_new.tail(days)
    
    html_template = df_Res.to_html(classes='table table-stripped') 
    
    return render_template("display_state_data.html",table=Markup(html_template),days=days,state=state)

@app.post("/display_country_data")
def display_country_data():
    
    date = str(request.form.get("date"))
    
    df = pd.read_csv(r"datasets/COVID-19 Cases statewise.csv",parse_dates=True)
    df1 = df.copy()
   
    df1.drop("S. No.",axis=1,inplace=True)
    df1 = df1.drop(df1[df1.Region == "India"].index)
    df1 = df1.drop(df1[df1.Region == "World"].index)
    df1 = df1.drop(df1[df1.Region == "State assignment pending"].index)
    df1.reset_index(inplace = True,drop=True)
    df1['Date']= pd.to_datetime(df1['Date']).dt.date

    date = datetime.strptime(date,"%Y-%m-%d").date()

    df_new = df1.loc[df1["Date"]==date]
    df_new.reset_index(inplace=True,drop=True)
    df_new.set_index("Region",inplace=True,drop=True)
    df_new.drop("Date",axis=1,inplace=True)
    
    html_template = df_new.to_html(classes="table table-stripped")
    return render_template("display_country_data.html",table=Markup(html_template),date=date)

@app.get("/get_prediction_input_data")
def get_prediction_input_data():
    return render_template("get_prediction_input.html")

# @app.post("/display_predicted_data")
# def display_predicted_data():
#     days = request.form.get("days")
    
#     image_url_1 = "http://127.0.0.1:5001/"
#     # image_url_1 = "https://mlp-web-service.herokuapp.com/"
#     response_1 = requests.get(url=image_url_1,params={"days":days})
#     img_1 = response_1.content
#     buffered_1 = io.BytesIO(img_1)
#     img_url_1 = base64.b64encode(buffered_1.getvalue()).decode()
    
#     image_url_2 = "http://127.0.0.1:5002/"
#     # image_url_2 = "https://lr-web-service.herokuapp.com/"
#     response_2 = requests.get(url=image_url_2,params={"days":days})
#     img_2 = response_2.content
#     buffered_2 = io.BytesIO(img_2)
#     img_url_2 = base64.b64encode(buffered_2.getvalue()).decode()
    
#     image_url_3 = "http://127.0.0.1:5003/"
#     # image_url_3 = "https://poly-web-service.herokuapp.com/"
#     response_3 = requests.get(url=image_url_3,params={"days":days})
#     img_3 = response_3.content
#     buffered_3 = io.BytesIO(img_3)
#     img_url_3 = base64.b64encode(buffered_3.getvalue()).decode()
#     return render_template("display_predicted_data.html",images={ 'image_1': img_url_1,'image_2': img_url_2,'image_3': img_url_3  })

@app.get("/mlp_predict")
def mlp_predict(): 
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    mlp_model = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000)
    mlp_model.fit(X_train,Y_train)
    forecast = mlp_model.predict(X_test)
    actual = Y_test
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "MLP regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('display_predicted_data.html',images={ 'image': plot_url })

@app.get("/lin_predict")
def lin_predict(): 
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    linear_model = LinearRegression()
    linear_model.fit(X_train,Y_train)
    forecast = linear_model.predict(X_test)
    actual = Y_test
    
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "Linear regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('display_predicted_data.html',images={ 'image': plot_url })

@app.get("/poly_predict")
def poly_predict(): 
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    min_mape = 100
    opt_degree = 0

    for degree in range(1,11):    
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features_train = poly.fit_transform(X_train)
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features_train, Y_train)

        poly_features_test = poly.fit_transform(X_test)
        forecast = poly_reg_model.predict(poly_features_test)
        actual = Y_test
        calc_mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  

        if calc_mape < min_mape:
            min_mape = calc_mape
            opt_degree = degree

    poly = PolynomialFeatures(degree=opt_degree, include_bias=False)
    poly_features_train = poly.fit_transform(X_train)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features_train, Y_train)
    actual = Y_test
    poly_features_predict = poly.fit_transform(X_test)
    forecast = poly_reg_model.predict(poly_features_predict)
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "Polynomial regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('display_predicted_data.html',images={ 'image': plot_url })

@app.get("/svr_predict")
def svr_predict(): 
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    kernels = ["linear","poly","rbf","sigmoid"]
    min_mape = 100
    opt_kernel = ""

    for kernel in kernels:    
        svr_model = SVR(kernel=kernel)
        svr_model.fit(X_train,Y_train)
        forecast = svr_model.predict(X_test)
        actual = Y_test
        calc_mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) 

        if calc_mape < min_mape:
            min_mape = calc_mape
            opt_kernel = kernel

    svr_model = SVR(kernel=opt_kernel)
    svr_model.fit(X_train,Y_train)
    actual = Y_test
    actual = Y_test
    
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "Support Vector regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('display_predicted_data.html',images={ 'image': plot_url })

@app.get("/rfr_predict")
def rfr_predict(): 
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    rfr_model = RandomForestRegressor()
    rfr_model.fit(X_train,Y_train)
    forecast = rfr_model.predict(X_test)
    actual = Y_test
    
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "Random Forest regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('display_predicted_data.html',images={ 'image': plot_url })

@app.get("/options")
def options():
    return render_template("options.html")

@app.get("/safety_measures")
def safety_measures():
    return render_template("safety-measures.html")
    
@app.get("/vaccination")
def vaccination():
    return render_template("vaccination.html")

if __name__ == "__main__":
    app.run(debug=True)