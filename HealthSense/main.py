import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load datasets
symptoms_df = pd.read_csv('data/DiseaseAndSymptoms.csv')
precautions_df = pd.read_csv('data/Disease precaution.csv')

print("✅ Symptoms Data Loaded:", symptoms_df.shape)
print("✅ Precautions Data Loaded:", precautions_df.shape)

# Step 2: Explore columns
print("\nColumns in Symptoms Data:")
print(symptoms_df.columns)

# Step 3: The dataset likely has columns like: Disease, Symptom_1, Symptom_2, ...
# Convert this to one-hot encoded symptom features
all_symptoms = set()

# Collect all unique symptoms
for col in symptoms_df.columns:
    if col != 'Disease':
        all_symptoms.update(symptoms_df[col].dropna().unique())

# Create empty dataframe with one column per unique symptom
encoded_df = pd.DataFrame(0, index=np.arange(len(symptoms_df)), columns=sorted(all_symptoms))

# Mark 1 for each symptom present in that row
for index, row in symptoms_df.iterrows():
    for col in symptoms_df.columns:
        if col != 'Disease' and pd.notna(row[col]):
            encoded_df.at[index, row[col]] = 1

# Add Disease column
encoded_df['Disease'] = symptoms_df['Disease']

# Step 4: Encode Disease labels
le = LabelEncoder()
encoded_df['Disease'] = le.fit_transform(encoded_df['Disease'])

# Step 5: Split features and target
X = encoded_df.drop('Disease', axis=1)
y = encoded_df['Disease']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Prediction function
def predict_disease(symptom_list):
    """
    symptom_list: list of symptom names (strings)
    """
    input_vector = np.zeros(len(X.columns))
    for s in symptom_list:
        if s in X.columns:
            input_vector[X.columns.get_loc(s)] = 1

    pred = model.predict([input_vector])
    disease_name = le.inverse_transform(pred)[0]

    # Find precautions
    row = precautions_df[precautions_df['Disease'] == disease_name]
    if not row.empty:
        precautions = row.iloc[0, 1:].dropna().tolist()
    else:
        precautions = ["No precautions found"]

    return disease_name, precautions

# Step 10: Example prediction
user_symptoms = ['itching', 'skin_rash', 'chills']
disease, precautions = predict_disease(user_symptoms)

print("\n🩺 Predicted Disease:", disease)
print("💊 Recommended Precautions:")
for i, p in enumerate(precautions, start=1):
    print(f"   {i}. {p}")
