import requests
from flask import Flask,render_template,url_for
from flask import request as req
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from termcolor import colored

app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize",methods=["GET","POST"])
def Summarize():
    if req.method == "POST":
        API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
        headers = {"Authorization": "Bearer api_cDqsshiYYdsPmHybqxvnlZYIctoHFwMovw"}


        dataa = pd.read_csv('BBC News Train.csv')
        dataa['CategoryId'] = dataa['Category'].factorize()[0]

        df = dataa[['Text', 'Category']]

        x = np.array(dataa.iloc[:, 0].values)
        y = np.array(dataa.CategoryId.values)
        cv = CountVectorizer(max_features=5000)
        x = cv.fit_transform(dataa.Text).toarray()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0).fit(x_train, y_train)

        data = req.form["data"]

        y_pred1 = cv.transform([data])

        yy = classifier.predict(y_pred1)
        if yy == [0]:
            Result = "Business News"
        elif yy == [1]:
            Result = "Tech News"
        elif yy == [2]:
            Result = "Politics News"
        elif yy == [3]:
            Result = "Sports News"
        elif yy == [1]:
            Result = "Entertainment News"

        maxL = int(req.form["maxL"])
        minL = maxL // 4

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": data,
            "parameters": {"min_length": minL, "max_length": maxL},
        })[0]
        Res = "Category:-" + " " + Result
        output["summary_text"] = "InShorts:-"+os.linesep+ output["summary_text"]

        final_output = output["summary_text"] + os.linesep+os.linesep + Res

        return render_template("index.html", result=final_output)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)