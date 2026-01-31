import streamlit as st
import pickle
import pandas as pd
import csv
import datetime
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Doctor", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

columns = pd.read_csv("Training.csv").drop("prognosis", axis=1).columns.tolist()


# ---------------- IMPORT DICTIONARIES ----------------
from main import translations, disease_info, doctor_advice, default_advice

# ---------------- LABELS (HEADINGS TRANSLATION) ----------------
labels = {
    "tablets": {"en": "Tablets", "te": "మందులు", "hi": "दवाइयाँ"},
    "precautions": {"en": "Precautions", "te": "జాగ్రత్తలు", "hi": "सावधानियाँ"},
    "history": {"en": "Your History", "te": "మీ చరిత్ర", "hi": "आपका इतिहास"},
    "predicted": {
        "en": "Predicted Disease",
        "te": "అంచనా వ్యాధి",
        "hi": "अनुमानित बीमारी"
    }
}

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- HISTORY CSV ----------------
if not os.path.exists("history.csv"):
    with open("history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["email", "name", "age", "time", "symptoms", "disease"])

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.title("🔐 AI Doctor Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email == "shalini1234@gmail.com" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.user = email
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Email or Password")

# ---------------- MAIN APP ----------------
def main_app():
    st.title("🩺 AI Doctor – Disease Prediction")

    # -------- Sidebar --------
    st.sidebar.header("Patient Details")
    name = st.sidebar.text_input("Patient Name")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120)

    language = st.sidebar.selectbox(
        "Language",
        ["en", "te", "hi"],
        format_func=lambda x: {"en": "English", "te": "తెలుగు", "hi": "हिंदी"}[x]
    )

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    # -------- Symptoms --------
    st.subheader("Select Symptoms")

    symptom_labels = {
        col: translations.get(col, {}).get(
            language, col.replace("_", " ").title()
        )
        for col in columns
    }

    selected = st.multiselect(
        "Symptoms",
        options=columns,
        format_func=lambda x: symptom_labels[x]
    )

    if st.button("Predict Disease"):
        if not selected:
            st.warning("Please select at least one symptom")
            return

        # Build input vector
        input_data = [1 if col in selected else 0 for col in columns]
        disease = le.inverse_transform(model.predict([input_data]))[0]

        # Save history
        with open("history.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                st.session_state.user,
                name,
                age,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(selected),
                disease
            ])

        # -------- OUTPUT --------
        info = disease_info.get(disease, {})
        display_disease = info.get("name", {}).get(language, disease)

        st.success(
            f"🦠 {labels['predicted'][language]}: **{display_disease}**"
        )

        info = disease_info.get(disease, {})
        display_disease = info.get("name", {}).get(language, disease)

        tablets = info.get("tablets", {}).get(language, [])
        precautions = info.get("precautions", {}).get(language, [])
        advice = doctor_advice.get(disease, {}).get(language, default_advice.get(language, ""))

        # -------- TABLETS --------
        st.subheader(f"💊 {labels['tablets'][language]}")
        if tablets:
            for t in tablets:
                st.write("•", t)
        else:
            st.write("—")

        # -------- PRECAUTIONS --------
        st.subheader(f"⚠️ {labels['precautions'][language]}")
        if precautions:
            for p in precautions:
                st.write("•", p)
        else:
            st.write("—")

        if advice:
            st.warning(advice)

 # -------- HISTORY --------
    st.divider()
    # యూజర్ ఎంచుకున్న భాషలో హెడ్డింగ్ కనిపిస్తుంది
    st.subheader(f"📜 {labels['history'][language]}")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        # లాగిన్ అయిన యూజర్ డేటాను మాత్రమే ఫిల్టర్ చేస్తుంది
        user_df = df[df["email"] == st.session_state.user].copy()

        if not user_df.empty:
            # 1. వ్యాధి పేరును ఎంచుకున్న భాషలోకి మార్చడం
            user_df["disease"] = user_df["disease"].apply(
                lambda x: disease_info.get(x, {}).get("name", {}).get(language, x)
            )

            # 2. లక్షణాలను (Symptoms) ఎంచుకున్న భాషలోకి మార్చడం
            def translate_symptoms(symptoms_str):
                if pd.isna(symptoms_str) or symptoms_str == "":
                    return ""
                # కామాలతో ఉన్న లక్షణాలను విడదీస్తుంది
                s_list = [s.strip() for s in str(symptoms_str).split(",")]
                # translations డిక్షనరీ నుండి ఆ భాషా పదాన్ని తీసుకుంటుంది
                translated_list = [
                    translations.get(s, {}).get(language, s.replace("_", " ").title()) 
                    for s in s_list
                ]
                return ", ".join(translated_list)

            user_df["symptoms"] = user_df["symptoms"].apply(translate_symptoms)

            # ట్రాన్స్‌లేట్ అయిన టేబుల్‌ను చూపిస్తుంది
            st.dataframe(user_df, use_container_width=True)
        else:
            # చరిత్ర లేకపోతే మెసేజ్
            no_hist_msg = {"en": "No history found.", "te": "చరిత్ర కనుగొనబడలేదు.", "hi": "कोई इतिहास नहीं मिला।"}
            st.write(no_hist_msg.get(language, "No history found."))
    else:
        st.error("history.csv file not found!")

# ---------------- ROUTER ----------------
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
