import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = ["x1", "x2", "x3"]
model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["churn"])

proba = model.predict_proba(test[features])[:, 1]
pd.DataFrame({"id": test["id"], "proba": proba}).to_csv(
    "predictions.csv", index=False
)
