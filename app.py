from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
cv_file = open("models/cv.pkl", "rb")
cv = pickle.load(
    cv_file
)
clf_file = open("models/clf.pkl", "rb")
clf = pickle.load(
    clf_file
)


def getPrediction(emailText):
    if emailText == "":
        return 0
    tokenized_email = cv.transform([emailText])
    prediction = clf.predict(tokenized_email)

    return prediction

@app.route('/')
def home():
    return 'POST to /predict for a prediction'

@app.route('/api/predict', methods=["POST"])
def predict():
    
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    prediction = getPrediction(email)
    prediction = 'Spam' if prediction == 1 else 'Not Spam'
    return jsonify({'prediction': prediction, 'email': email}) 




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080 debug=True)
