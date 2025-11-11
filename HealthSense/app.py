import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    symptoms_df = pd.read_csv('data/DiseaseAndSymptoms.csv')
    precautions_df = pd.read_csv('data/Disease precaution.csv')

    all_symptoms = set()
    for col in symptoms_df.columns:
        if col != 'Disease':
            all_symptoms.update(symptoms_df[col].dropna().unique())

    encoded_df = pd.DataFrame(0, index=np.arange(len(symptoms_df)), columns=sorted(all_symptoms))
    for index, row in symptoms_df.iterrows():
        for col in symptoms_df.columns:
            if col != 'Disease' and pd.notna(row[col]):
                encoded_df.at[index, row[col]] = 1
    encoded_df['Disease'] = symptoms_df['Disease']

    le = LabelEncoder()
    encoded_df['Disease'] = le.fit_transform(encoded_df['Disease'])

    X = encoded_df.drop('Disease', axis=1)
    y = encoded_df['Disease']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le, X, precautions_df


model, le, X, precautions_df = load_data()


st.set_page_config(page_title="HealthSense: AI-Powered Disease Prediction", page_icon="🩺", layout="wide")

st.title("🩺 HealthSense: AI-Powered Disease Prediction")
st.write("Chat with our AI Health Assistant or select symptoms manually to predict diseases.")

with st.expander("🧩 Manual Symptom Selection", expanded=False):
    selected_symptoms = st.multiselect("Select your symptoms:", options=list(X.columns))
    if st.button("🔍 Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = np.zeros(len(X.columns))
            for s in selected_symptoms:
                if s in X.columns:
                    input_vector[X.columns.get_loc(s)] = 1
            pred = model.predict([input_vector])
            disease_name = le.inverse_transform(pred)[0]

            # Get precautions
            row = precautions_df[precautions_df['Disease'] == disease_name]
            if not row.empty:
                precautions = row.iloc[0, 1:].dropna().tolist()
            else:
                precautions = ["No precautions found"]

            # Display
            st.success(f"### 🧠 Predicted Disease: **{disease_name}**")
            st.markdown("#### 💊 Recommended Precautions:")
            for i, p in enumerate(precautions, start=1):
                st.write(f"✅ {p}")

# -------------------------------
# Chatbot Section
# -------------------------------
st.markdown("---")
st.header("💬 HealthSense Chatbot")
st.caption("Talk with our AI assistant about your symptoms!")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! 👋 I'm your HealthSense assistant. Tell me your symptoms, and I’ll try to predict your possible disease."}
    ]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    mentioned_symptoms = [s for s in X.columns if s.lower() in prompt.lower()]
    if not mentioned_symptoms:
        response = "I couldn’t detect any known symptoms. Could you rephrase or mention specific symptoms (like fever, cough, headache)?"
    else:
        input_vector = np.zeros(len(X.columns))
        for s in mentioned_symptoms:
            input_vector[X.columns.get_loc(s)] = 1

        pred = model.predict([input_vector])
        disease_name = le.inverse_transform(pred)[0]

        row = precautions_df[precautions_df['Disease'] == disease_name]
        if not row.empty:
            precautions = row.iloc[0, 1:].dropna().tolist()
        else:
            precautions = ["No precautions found"]

        response = f"Based on your symptoms ({', '.join(mentioned_symptoms)}), you may have **{disease_name}**.\n\n**Recommended Precautions:**\n" + "\n".join([f"✅ {p}" for p in precautions])

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning | HealthSense © 2025")
