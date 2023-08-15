from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model using joblib
model = joblib.load("C:/Users/shriv/OneDrive/Desktop/Credit_card_model")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None  # Initialize result to None

    if request.method == "POST":
        input_df = request.form.get("input_df")
        if input_df:
            # Split the comma-separated input into a list of floats
            input_df_lst = [float(value.strip()) for value in input_df.split(',')]

            # Preprocess the input data (scaling and PCA)
            # ...

            # Make a prediction using the loaded model
            prediction = model.predict([input_df_lst])

            if prediction == 0:
                result = "Normal Transaction"
            else:
                result = "Fraud Transaction"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
