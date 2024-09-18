import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score

file_path = 'C:\\Users\\Admin\\Downloads\\hypothyroid.csv'
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")

label_encoders = {}
for column in ['Hypothyroid', 'Hyperthyroid', 'Thyroid Function']:  
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('Thyroid Function', axis=1)  
y = data['Thyroid Function']

model = BayesianNetwork([
    ('T3', 'Thyroid Function'),
    ('T4', 'Thyroid Function'),
    ('TSH', 'Thyroid Function'),
    ('Goiter', 'Thyroid Function')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)

predictions = []
for _, row in data.iterrows():
    evidence = {
        'T3': row['T3'],
        'T4': row['T4'],
        'TSH': row['TSH'],
        'Goiter': row['Goiter']
    }
    prediction = inference.map_query(variables=['Thyroid Function'], evidence=evidence)
    predictions.append(prediction['Thyroid Function'])

accuracy = accuracy_score(y, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
