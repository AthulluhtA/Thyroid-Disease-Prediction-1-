import os, json, joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = joblib.load(os.path.join(BASE, "model", "model.pkl"))
FEATURES = json.load(open(os.path.join(BASE, "model", "features.json")))
METRICS = json.load(open(os.path.join(BASE, "model", "metrics.json")))

CONTINUOUS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
BINARY = [f for f in FEATURES if f not in CONTINUOUS]

RANGES = {"age": (0, 1), "TSH": (0, 0.6), "T3": (0, 0.1),
          "TT4": (0, 0.5), "T4U": (0, 0.4), "FTI": (0, 0.7)}

CLASS_LABELS = {
    1: "Class 1 - Abnormal (low thyroid function pattern)",
    2: "Class 2 - Abnormal (mild low-function pattern)",
    3: "Class 3 - Normal",
}


@app.route("/")
def index():
    rows = [{"label": CLASS_LABELS[int(c)], "class": c, **METRICS["report"][c]}
            for c in ["1", "2", "3"]]
    return render_template(
        "index.html",
        accuracy=round(METRICS["accuracy"] * 100, 2),
        rows=rows,
        macro=METRICS["report"]["macro avg"],
        weighted=METRICS["report"]["weighted avg"],
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, error, values, warnings = None, None, {}, []
    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f in CONTINUOUS}
            values["age"] = values["age"] / 100      # years -> dataset scale

            warnings = [f"{k} = {values[k]} is outside the typical training range {lo}-{hi}"
                        for k, (lo, hi) in RANGES.items() if not lo <= values[k] <= hi]

            for f in BINARY:
                values[f] = 1 if request.form.get(f) else 0
            X = pd.DataFrame([values])[FEATURES]
            pred = int(model.predict(X)[0])
            conf = float(model.predict_proba(X)[0].max()) * 100
            result = {"cls": pred, "label": CLASS_LABELS[pred], "conf": round(conf, 1)}
        except (ValueError, KeyError):
            error = "Please fill all numeric fields with valid numbers."
    return render_template("predict.html", result=result, error=error, warnings=warnings,
                           continuous=CONTINUOUS, binary=BINARY, values=values)


if __name__ == "__main__":
    app.run(debug=True)