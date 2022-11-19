# import relevant libraries for flask,html rendering and loading the ML model
from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scale = joblib.load("Scale.pkl")
label_Region = joblib.load("label_Region.pkl")
label_Department = joblib.load("label_Department.pkl")


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():

    Department = request.form['Department']
    Department = pd.Series(Department)
    Department = label_Department.transform(Department)[0]

    Region = request.form['Region']
    Region = pd.Series(Region)
    Region = label_Region.transform(Region)[0]

    Education = request.form['Education']
    if Education == "Master's & above":
        Education = 1
    elif Education == "Bachelor's":
        Education = 2
    elif Education == "Below Secondary":
        Education = 3

    Gender = request.form['Gender']
    if Gender == "Female":
        Gender = 0
    else:
        Gender = 1

    RecruitmentChannel = request.form['Recruitment Channel']
    if RecruitmentChannel == "sourcing":
        RecruitmentChannel = 1
    elif RecruitmentChannel == "referred":
        RecruitmentChannel = 2
    elif RecruitmentChannel == "other":
        RecruitmentChannel = 3

    NumberOfTrainings = request.form['Number Of Trainings']
    Age = request.form['Age']
    PreviousYearRating = request.form['Previous Year Rating']
    LengthOfService = request.form['Length Of Service']

    KPIsMet = request.form['KPIs Met > 80%']
    if KPIsMet == "Yes":
        KPIsMet = 1
    else:
        KPIsMet = 0

    AwardsWon = request.form['Awards Won']
    if AwardsWon == "Yes":
        AwardsWon = 1
    else:
        AwardsWon = 0

    AverageTrainingScore = request.form['Average Training Score']

    rowDF = pd.DataFrame([pd.Series([Department,Region,Education,Gender,RecruitmentChannel,NumberOfTrainings,Age,PreviousYearRating,LengthOfService,KPIsMet,AwardsWon,AverageTrainingScore])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    # Model Prediction
    prediction = model.predict_proba(rowDF_new)
    print(f"The predicted value is : {prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round Value : {valPred*100}%")
        return render_template('result.html', pred=f'Good News...! \n\nThis Candidate will be PROMOTED...\n\nProbability of this Candidate being Promoted is {valPred*100:.2f}%.')
    else:
        valPred = round(prediction[0][0],3)
        print(f"The Round Value : {valPred*100}%")
        return render_template('result.html', pred=f'Bad News...! \n\nThis Candidate will NOT be PROMOTED...\n\nProbability of this Candidate not being Promoted is {valPred*100:.2f}%.')


if __name__ == '__main__':
    app.run(debug=True)
