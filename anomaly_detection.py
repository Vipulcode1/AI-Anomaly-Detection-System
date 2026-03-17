from sklearn.ensemble import IsolationForest
import numpy as np

# generate sample data
data = np.random.rand(100, 2)

# train model
model = IsolationForest(contamination=0.1)
model.fit(data)

# predict anomalies
predictions = model.predict(data)

print("Predictions:", predictions)