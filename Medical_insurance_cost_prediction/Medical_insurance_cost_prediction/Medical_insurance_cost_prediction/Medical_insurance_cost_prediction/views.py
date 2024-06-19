from django.shortcuts import render
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    insurance_dataset = pd.read_csv(r"C:\Users\Arnab Pan\Desktop\CSV files\insurance.csv")
    insurance_dataset.replace({'sex': {'male': 1, 'female': 0}}, inplace=True)
    insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}},
                             inplace=True)
    x = insurance_dataset.drop(columns='charges', axis=1)
    y = insurance_dataset['charges']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    val1 = float(request.GET["n1"])
    val2 = float(request.GET["n2"])
    val3 = float(request.GET["n3"])
    val4 = float(request.GET["n4"])
    val5 = float(request.GET["n5"])
    val6 = float(request.GET["n6"])

    input_data = ([[val1,val2,val3,val4,val5,val6]])
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = regressor.predict(input_data_reshaped)

    result2 = prediction

    return render(request, 'predict.html', {"result2": result2})