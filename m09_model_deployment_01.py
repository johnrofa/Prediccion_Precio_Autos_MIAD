#!/usr/bin/python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os



def Modelo(db_02):

    print('db_02 :', db_02)

    # Importar objetos -modelos

    print('ruta con os.getcwd() /+ :', os.getcwd() + '/phishing_clf_01.pkl')

    leMake = joblib.load(os.path.dirname(__file__)  + '/leMake_01.pkl')
    leModel = joblib.load(os.path.dirname(__file__) + '/leModel_01.pkl')
    leState = joblib.load(os.path.dirname(__file__)  + '/leState_01.pkl')
    regRF11 = joblib.load(os.path.dirname(__file__) + '/phishing_clf_01.pkl')

    db_02["State"] = leState.transform(db_02.State)
    db_02["Make"] = leMake.transform(db_02.Make)
    db_02["Model"] = leModel.transform(db_02.Model)

    # Make prediction
    ypredRF11 = regRF11.predict(db_02)
    ypredRF11 = str(ypredRF11[0])

    return ypredRF11



def predict_proba(url):
    print('ruta : ', os.path.dirname(__file__) + '/phishing_clf.pkl')
    clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl')
    url_ = pd.DataFrame([url], columns=['url'])

    # Create features
    keywords = ['https', 'login', '.php', '.html', '@', 'sign']
    for keyword in keywords:
        url_['keyword_' + keyword] = url_.url.str.contains(keyword).astype(int)

    url_['lenght'] = url_.url.str.len() - 2
    domain = url_.url.str.split('/', expand=True).iloc[:, 2]
    url_['lenght_domain'] = domain.str.len()
    url_['isIP'] = (url_.url.str.replace('.', '') * 1).str.isnumeric().astype(int)
    url_['count_com'] = url_.url.str.count('com')

    # Make prediction
    p1 = clf.predict_proba(url_.drop('url', axis=1))[0, 1]

    return p1


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Please add an URL')

    else:

        url = sys.argv[1]

        p1 = predict_proba(url)

        print(url)
        print('Probability of Phishing: ', p1)
