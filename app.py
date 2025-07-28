import streamlit as st
import pandas as pd
import pickle

# Load model dan fitur
with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

st.set_page_config(page_title="Prediksi Depresi Mahasiswa")
st.title("🎓 Prediksi Depresi Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan mengalami depresi.")

# Input pengguna
gender = st.radio("Jenis Kelamin", ('Laki-laki', 'Perempuan'))
age = st.number_input("Usia", min_value=15, max_value=40, value=20)
year = st.selectbox("Tahun Studi", [1, 2, 3, 4])
cgpa = st.slider("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
married = st.radio("Status Pernikahan", ("Ya", "Tidak"))
anxiety = st.radio("Apakah mengalami kecemasan?", ("Ya", "Tidak"))
panic = st.radio("Apakah pernah mengalami serangan panik?", ("Ya", "Tidak"))
treatment = st.radio("Apakah pernah konsultasi ke spesialis?", ("Ya", "Tidak"))

# Proses input
data = {
    'Choose your gender': 1 if gender == 'Laki-laki' else 0,
    'Age': age,
    'Your current year of Study': year,
    'CGPA_numeric': cgpa,
    'Marital status': 1 if married == 'Ya' else 0,
    'Do you have Anxiety?': 1 if anxiety == 'Ya' else 0,
    'Do you have Panic attack?': 1 if panic == 'Ya' else 0,
    'Did you seek any specialist for a treatment?': 1 if treatment == 'Ya' else 0
}

df = pd.DataFrame([data])
df_selected = df[selected_features]

# Prediksi
if st.button("Prediksi"):
    result = model.predict(df_selected)[0]
    if result == 1:
        st.error("⚠️ Mahasiswa kemungkinan mengalami **depresi**.")
    else:
        st.success("✅ Mahasiswa kemungkinan **tidak mengalami depresi**.")
