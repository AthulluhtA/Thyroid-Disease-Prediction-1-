# Thyroid Disease Classification

This student project classifies thyroid conditions with the UCI `ann-thyroid` dataset. It includes the original Flask interface and a Django equivalent. Both applications reuse the trained Decision Tree artifact; the web apps do not retrain the model.

## Setup

From the repository root, create and activate the Python 3.13 virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The saved model files are already included under `flask_app/model/` and are copied into `django_app/predictor/ml_model/` for the Django app.

## Train the model

Training is optional because the artifacts are already present. If the model must be regenerated, run this from the `ml/` directory so its relative data paths resolve:

```powershell
cd ml
python train.py
```

The training script uses SMOTENC on the training set only, then fits a Decision Tree with `max_depth=8` and `min_samples_leaf=3`. It writes model metrics and EDA images to `flask_app/`.

## Run Flask

```powershell
cd flask_app
python app.py
```

Open `http://127.0.0.1:5000/`.

## Run Django

```powershell
cd django_app
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`. The results page is `/`, and the prediction form is `/predict/`.

## Interpretation and limitations

- Class 3 is labeled Normal. Classes 1 and 2 are labeled abnormal low-thyroid-function patterns inferred from their feature distributions; the source file does not provide a numeric-to-name mapping.
- The dataset is heavily imbalanced: class 3 is about 92% of the data. Accuracy alone is therefore misleading; inspect per-class recall, macro F1, and the confusion matrix.
- The lab values use the dataset's normalized 0-1 scale. The web form accepts age in years and converts it to that scale.
- Out-of-range values produce warnings but are still sent to the model. A Decision Tree can return near-100% confidence and does not extrapolate safely, so its confidence is not a calibrated medical probability.
- This is an educational project and is not a diagnostic tool.
