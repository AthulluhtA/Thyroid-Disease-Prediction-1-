import os, json, joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

cols = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
        "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
        "TSH", "T3", "TT4", "T4U", "FTI", "target"]

train = pd.read_csv("../data/ann-train.data", sep=r"\s+", header=None, names=cols)
test  = pd.read_csv("../data/ann-test.data",  sep=r"\s+", header=None, names=cols)

X_train, y_train = train.drop(columns="target"), train["target"]
X_test,  y_test  = test.drop(columns="target"),  test["target"]

# Decision Tree 
from imblearn.over_sampling import SMOTENC

CONT = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
binary_idx = [i for i, c in enumerate(X_train.columns) if c not in CONT]

# SMOTE on the TRAINING set only; the test set stays untouched
smote = SMOTENC(categorical_features=binary_idx, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_res.value_counts().to_dict())

model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42)
model.fit(X_res, y_res)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, output_dict=True)
print("Accuracy:", round(acc, 4))
print(classification_report(y_test, pred))

# output folders
out = "../flask_app"
os.makedirs(f"{out}/static/img", exist_ok=True)
os.makedirs(f"{out}/model", exist_ok=True)

# save model, feature order, metrics
joblib.dump(model, f"{out}/model/model.pkl")
json.dump(list(X_train.columns), open(f"{out}/model/features.json", "w"))
json.dump({"accuracy": acc, "report": report},
          open(f"{out}/model/metrics.json", "w"), indent=2)

# confusion matrix image
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=[1, 2, 3]).plot(cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.savefig(f"{out}/static/img/confusion_matrix.png", bbox_inches="tight"); plt.close()

# EDA images
sns.countplot(x="target", data=train); plt.title("Class distribution")
plt.savefig(f"{out}/static/img/class_dist.png", bbox_inches="tight"); plt.close()

for col in ["TSH", "FTI"]:
    sns.boxplot(x="target", y=col, data=train); plt.title(f"{col} by class")
    plt.savefig(f"{out}/static/img/box_{col}.png", bbox_inches="tight"); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(train.corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap")
plt.savefig(f"{out}/static/img/corr.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
sns.countplot(x=y_train, ax=ax[0]); ax[0].set_title("Before SMOTE")
sns.countplot(x=y_res, ax=ax[1]); ax[1].set_title("After SMOTE")
plt.savefig(f"{out}/static/img/smote.png", bbox_inches="tight"); plt.close()

import numpy as np

classes = ["1", "2", "3"]
prec = [report[c]["precision"] for c in classes]
rec  = [report[c]["recall"] for c in classes]
x = np.arange(len(classes))

plt.bar(x - 0.2, prec, 0.4, label="Precision")
plt.bar(x + 0.2, rec, 0.4, label="Recall")
plt.xticks(x, [f"Class {c}" for c in classes])
plt.ylim(0, 1.05)
plt.legend(); plt.title("Precision & Recall per class")
plt.savefig(f"{out}/static/img/prec_recall.png", bbox_inches="tight"); plt.close()