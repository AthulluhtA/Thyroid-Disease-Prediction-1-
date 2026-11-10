import json
from pathlib import Path

import joblib
import pandas as pd
from django.shortcuts import render


MODEL_DIR = Path(__file__).resolve().parent / 'ml_model'
model = joblib.load(MODEL_DIR / 'model.pkl')
FEATURES = json.loads((MODEL_DIR / 'features.json').read_text())
METRICS = json.loads((MODEL_DIR / 'metrics.json').read_text())

CONTINUOUS = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
BINARY = [feature for feature in FEATURES if feature not in CONTINUOUS]
RANGES = {
    'age': (0, 1),
    'TSH': (0, 0.6),
    'T3': (0, 0.1),
    'TT4': (0, 0.5),
    'T4U': (0, 0.4),
    'FTI': (0, 0.7),
}
CLASS_LABELS = {
    1: 'Class 1 - Abnormal (low thyroid function pattern)',
    2: 'Class 2 - Abnormal (mild low-function pattern)',
    3: 'Class 3 - Normal',
}


def index(request):
    report = METRICS['report']
    rows = []
    for class_id in ('1', '2', '3'):
        metrics = report[class_id]
        rows.append({
            'label': CLASS_LABELS[int(class_id)],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1-score'],
            'support': int(metrics['support']),
        })

    def summary_row(name):
        metrics = report[name]
        return {
            'label': name.title(),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1-score'],
            'support': int(metrics['support']),
        }

    rows.extend([summary_row('macro avg'), summary_row('weighted avg')])
    return render(request, 'predictor/index.html', {
        'accuracy': round(METRICS['accuracy'] * 100, 2),
        'rows': rows,
    })


def predict(request):
    error = None
    result = None
    warnings = []
    values = {field: '' for field in CONTINUOUS}
    checked = {field: False for field in BINARY}

    if request.method == 'POST':
        for field in CONTINUOUS:
            values[field] = request.POST.get(field, '')
        for field in BINARY:
            checked[field] = field in request.POST

        try:
            numeric_values = {field: float(values[field]) for field in CONTINUOUS}
            numeric_values['age'] /= 100

            for field, (lower, upper) in RANGES.items():
                if not lower <= numeric_values[field] <= upper:
                    warnings.append(
                        f'{field} = {numeric_values[field]} is outside the typical '
                        f'training range {lower}-{upper}'
                    )

            numeric_values.update({field: int(checked[field]) for field in BINARY})
            features = pd.DataFrame([numeric_values])[FEATURES]
            predicted_class = int(model.predict(features)[0])
            confidence = float(model.predict_proba(features)[0].max()) * 100
            result = {
                'label': CLASS_LABELS[predicted_class],
                'confidence': round(confidence, 1),
            }
        except (ValueError, KeyError):
            error = 'Please fill all numeric fields with valid numbers.'

    return render(request, 'predictor/predict.html', {
        'result': result,
        'error': error,
        'warnings': warnings,
        'cont_fields': [{'name': field, 'value': values[field]} for field in CONTINUOUS],
        'bin_fields': [{'name': field, 'checked': checked[field]} for field in BINARY],
    })