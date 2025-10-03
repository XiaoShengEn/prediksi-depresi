"""\
apps_final.py
-----------------
A single Streamlit app that includes:
- Language switcher (Indonesia / English)
- Navigation: Beranda/Home, Tentang/About, Prediksi Depresi/Depression Prediction
- Prediction form using pre-trained model_depresi.pkl, scaler.pkl, label_encoders.pkl
- Consistent styling & graceful error handling

Place this file in the same directory as:
- model_depresi.pkl
- scaler.pkl
- label_encoders.pkl
- Depression Student Dataset.csv (optional; only loaded for reference)
- Depresi.jpeg (image shown on Beranda/Home)

Run:
    streamlit run apps_final.py
"""

import streamlit as st
import pandas as pd
import joblib
import pickle
from pathlib import Path

# ------------------------------------------------------------------
# Page Config (call as early as possible)
# ------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Risiko Depresi", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# Global CSS (App background + typography + sidebar styling)
# ------------------------------------------------------------------
APP_BASE_CSS = """
    <style>
    .custom-text {
        font-size: 18px !important;
        color: #e6e6e6 !important;
        text-align: justify !important;
        line-height: 1.75 !important;
        max-width: 900px;
        margin: auto;
        padding: 1rem 1.5rem;
    }

    .custom-text ul, .custom-text ol {
        padding-left: 1.3rem;
        margin-bottom: 1rem;
    }

    .custom-text li {
        margin-bottom: 0.5rem;
    }

    .custom-text h3, .custom-text h4 {
        margin-top: 1.5rem;
        color: #ffffff;
    }

    .stApp {
        background-color: #1e1f26;
    }

    /* Sidebar Style */
    [data-testid="stSidebar"] {
        background-color: #2C2F33;
        padding: 20px 10px;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: white;
        text-align: center;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #dcdcdc;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #40444b;
        color: white;
        border-radius: 6px;
        border: 1px solid #5865F2;
    }
    [data-testid="stSidebar"] hr {
        margin: 15px 0;
        border: 1px solid #3a3f47;
    }
    </style>
"""

st.markdown(APP_BASE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load assets (model, scaler, encoders, dataset optional)
# ------------------------------------------------------------------
## Lokasi dua file CSV
PRIMER_PATH = Path("Depression Student Dataset Primer.csv")
SEKUNDER_PATH = Path("Depression Student Dataset Sekunder.csv")

# Load dan gabung tanpa ubah kolom
try:
    df_primer = pd.read_csv(PRIMER_PATH)
    df_sekunder = pd.read_csv(SEKUNDER_PATH)
    
    # Gabung langsung
    df = pd.concat([df_primer, df_sekunder], ignore_index=True)

except Exception as e:
    df = None
    print(f"❌ Gagal memuat dataset: {e}")

MODEL_PATH = Path("model_depression.joblib")

# Load bundle
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
scaler = model_bundle["scaler"]
label_encoders = model_bundle["encoders"]

# ------------------------------------------------------------------
# Long-form copy (Home & About) - Indonesia
# ------------------------------------------------------------------
HOME_DESC_ID = '''
### 💬 Kenali Risiko Depresi dengan Cara yang Mudah dan Empatik

Selamat datang di aplikasi prediksi risiko depresi untuk mahasiswa 🎓🧠

Depresi adalah salah satu gangguan kesehatan mental yang paling umum dan sering disertai dengan kecemasan. Menurut WHO (2023), depresi ditandai oleh suasana hati yang tertekan, hilangnya minat terhadap aktivitas sehari-hari dalam jangka waktu yang lama, serta dapat mengganggu fungsi di lingkungan kerja atau pendidikan. 
Tingkat keparahan depresi bervariasi, dari yang ringan dan sementara hingga yang berat dan berlangsung lama. Beberapa orang mungkin hanya mengalaminya sekali, sementara yang lain dapat mengalaminya berulang kali. Meskipun depresi dapat meningkatkan risiko bunuh diri, hal ini dapat dicegah jika individu mendapatkan dukungan yang tepat, terutama bagi remaja yang mengalami pikiran untuk mengakhiri hidup.

🧩 **Bagaimana aplikasi ini bekerja?**
- Isi pertanyaan - pertanyaan yang ada pada halaman "Prediksi Depresi".
- Klik tombol "Prediksi" untuk memuatkan hasil.
- Hasil beserta dengan saran akan muncul dibawah.
'''

ABOUT_DESC_ID = '''
### 🎯 Tujuan Aplikasi

Untuk mengidentifikasi seseorang yang mengalami depresi, diperlukan pendekatan berbasis analisis data terhadap gejala dan faktor risiko yang terkait dengan kondisi psikologis, seperti tekanan aktivitas, kualitas tidur, stres ekonomi, dan kepuasan hidup. Oleh karena itu, penggunaan teknologi sangat dibutuhkan untuk membantu mendeteksi risiko depresi secara sistematis dan otomatis. 
- 🧠 Meningkatkan kesadaran mahasiswa terhadap pentingnya kesehatan mental.
- 📊 Memberikan estimasi awal terkait risiko depresi secara otomatis dan personal.
- 🤝 Menjadi alat bantu tambahan bagi dosen, konselor kampus, dan tenaga pendidik dalam memahami kondisi mahasiswa.

---

### ❓ Mengapa Aplikasi Ini Dibuat?

Berdasarkan berbagai penelitian, mahasiswa termasuk dalam kelompok yang rentan mengalami gangguan kesehatan mental, terutama **depresi**, karena berbagai tekanan seperti:

- **Beban akademik yang tinggi**
- **Stres finansial**
- **Kurangnya waktu tidur**
- **Pola makan**
- **Minimnya akses terhadap layanan psikologis**

Namun, banyak dari mereka yang tidak menyadari atau enggan mencari bantuan karena stigma atau kurangnya informasi. Aplikasi ini hadir sebagai **langkah awal yang mudah, cepat, dan empatik** untuk mengenali kondisi tersebut.

---

### 🧪 Teknologi yang Digunakan

Aplikasi ini menggunakan pendekatan **Machine Learning** untuk memprediksi risiko depresi. Berikut teknologi dan proses yang digunakan:

- 🔍 **Model Machine Learning**: `Binary Logistic Regression` Model ini dipilih karena kesederhanaannya, kecepatan dalam inferensi, serta interpretabilitas tinggi untuk klasifikasi biner (*Depresi* vs *Tidak Depresi*).
- 📈 **Preprocessing Data**: 
  - Encoding label pada data kategorikal menggunakan `LabelEncoder`
  - Standarisasi fitur numerik menggunakan `StandardScaler`
- 🧪 **Pelatihan Model**:
  - Data dibagi menjadi data latih dan uji 80:20 `train_test_split`
  - Model dilatih untuk memprediksi variabel target: apakah seseorang mengalami depresi atau tidak berdasarkan data survei.
- 🧠 **Fitur yang dipertimbangkan**:
  - Jenis Kelamin
  - Usia
  - Tekanan Akademik
  - Kepuasan Belajar
  - Durasi Tidur
  - Kebiasaan Makan
  - Pikiran untuk Bunuh Diri
  - Jam Belajar per Hari
  - Stres Finansial
  - Riwayat Gangguan Mental dalam Keluarga

---

### ⚠️ Catatan Penting

Hasil prediksi dari aplikasi ini **bukan merupakan diagnosis klinis**, melainkan estimasi berbasis data. Untuk diagnosis atau penanganan lebih lanjut, silakan konsultasi dengan tenaga kesehatan mental profesional.

---

### 💡 Harapan Pengembang

Semoga aplikasi ini dapat menjadi:
- 🌱 Awal dari peningkatan kesadaran akan pentingnya kesehatan mental.
- 🔑 Alat bantu sederhana dalam deteksi dini risiko depresi.
- 🧩 Bagian dari solusi digital dalam mendukung kesejahteraan emosional mahasiswa.
'''

# ------------------------------------------------------------------
# Long-form copy (Home & About) - English
# ------------------------------------------------------------------
HOME_DESC_EN = '''
### 💬 Understand Depression Risk Easily and Empathetically

Welcome to the student depression risk prediction app 🎓🧠

Depression is one of the most common mental health disorders, often accompanied by anxiety. According to WHO (2023), depression is characterized by a persistently low mood, loss of interest in daily activities over a long period, and can interfere with functioning at work or in education.  
The severity of depression varies, from mild and temporary to severe and long-lasting. Some people may experience it only once, while others may experience it repeatedly. Although depression can increase the risk of suicide, it can be prevented if individuals receive proper support, especially for adolescents who have suicidal thoughts.

🧩 **How does this app work?**
- Fill in the questions on the "Depression Prediction" page.
- Click the "Predict" button to load the results.
- The result and suggestions will appear below.
'''

ABOUT_DESC_EN = '''
### 🎯 Purpose of the Application

To identify someone experiencing depression, a data-driven approach is needed to analyze symptoms and risk factors related to psychological conditions, such as activity pressure, sleep quality, financial stress, and life satisfaction.  
Therefore, technology is essential to help systematically and automatically detect depression risk.

- 🧠 Increase students' awareness of the importance of mental health.
- 📊 Provide an initial automated and personalized estimate of depression risk.
- 🤝 Serve as an additional tool for lecturers, campus counselors, and educators to understand students' conditions.

---

### ❓ Why Was This App Created?

Research shows that students are among the groups most vulnerable to mental health problems, especially **depression**, due to various pressures such as:

- **High academic workload**
- **Financial stress**
- **Lack of sleep**
- **Dietary habits**
- **Limited access to psychological services**

However, many students are unaware or reluctant to seek help due to stigma or lack of information. This app is a **first step that is simple, quick, and empathetic** in recognizing this condition.

---

### 🧪 Technology Used

This application uses a **Machine Learning** approach to predict depression risk. Technologies and processes used include:

- 🔍 **Machine Learning Model**: `Binary Logistic Regression`  
  This model is chosen for its simplicity, fast inference, and high interpretability for binary classification (*Depression* vs *No Depression*).
- 📈 **Data Preprocessing**:
  - Label encoding for categorical data using `LabelEncoder`
  - Standardizing numeric features using `StandardScaler`
- 🧪 **Model Training**:
  - Data is split into training and test sets (80:20) using `train_test_split`
  - The model is trained to predict the target variable: whether someone is experiencing depression based on survey data.
- 🧠 **Features considered**:
  - Gender
  - Age
  - Academic Pressure
  - Study Satisfaction
  - Sleep Duration
  - Dietary Habits
  - Suicidal Thoughts
  - Study Hours per Day
  - Financial Stress
  - Family History of Mental Illness

---

### ⚠️ Important Note

The predictions from this app **are not clinical diagnoses**, but data-based estimates.  
For further diagnosis or treatment, please consult a mental health professional.

---

### 💡 Developer’s Hope

We hope this app can be:
- 🌱 A starting point to raise awareness of the importance of mental health.
- 🔑 A simple tool for early detection of depression risk.
- 🧩 A part of digital solutions to support students' emotional well-being.
'''

# ------------------------------------------------------------------
# Label dictionaries (short UI strings)
# ------------------------------------------------------------------
LABELS_ID = {
    "home_title": "Prediksi Risiko Depresi Untuk Mahasiswa",
    "home_desc": HOME_DESC_ID,
    "about_title": "ℹ️ Tentang Website",
    "about_desc": ABOUT_DESC_ID,
    "predict_title": "🧠 Prediksi Risiko Depresi Mahasiswa",
    "predict_desc": "Isi formulir di bawah untuk memprediksi apakah anda berpotensi mengalami depresi.",
    "gender": "Jenis Kelamin",
    "age": "Usia",
    "academic_pressure": "Seberapa Besar Tekanan Akademik Yang Anda Rasakan",
    "study_satisfaction": "Seberapa Puas Kepuasan Belajar Anda Dalam Belajar",
    "sleep_duration": "Berapa Durasi Tidur Anda Dalam Sehari",
    "dietary_habits": "Bagaimana Pola Makan Anda Sehari Hari",
    "suicidal_thoughts": "Pernahkah Anda Berpikir Untuk Bunuh Diri?",
    "study_hours": "Seberapa Lama Anda Belajar Dalam Satu Hari",
    "financial_stress": "Seberapa Besar Stres Finansial Anda",
    "family_history": "Apakah Ada Riwayat Gangguan Mental Dalam Keluarga Anda",
    "predict_button": "Prediksi",
    "result_yes": "🚨 Mahasiswa ini kemungkinan mengalami depresi.",
    "result_no": "✅ Mahasiswa ini tidak terindikasi mengalami depresi.",
    "warning": "⚠️ Lengkapi semua pilihan!",
}

LABELS_EN = {
    "home_title": "Student Depression Risk Prediction",
    "home_desc": HOME_DESC_EN,
    "about_title": "ℹ️ About This Website",
    "about_desc": ABOUT_DESC_EN,
    "predict_title": "🧠 Student Depression Risk Prediction",
    "predict_desc": "Fill out the form below to predict whether you are at risk of depression.",
    "gender": "Gender",
    "age": "Age",
    "academic_pressure": "How Much Academic Pressure Do You Feel?",
    "study_satisfaction": "How Satisfied Are You With Your Studies?",
    "sleep_duration": "How Long Do You Sleep Daily?",
    "dietary_habits": "What Is Your Daily Dietary Habit?",
    "suicidal_thoughts": "Have You Ever Had Suicidal Thoughts?",
    "study_hours": "How Many Hours Do You Study Daily?",
    "financial_stress": "How Much Financial Stress Do You Experience?",
    "family_history": "Is There a Family History of Mental Illness?",
    "predict_button": "Predict",
    "result_yes": "🚨 This student is likely experiencing depression.",
    "result_no": "✅ This student does not show signs of depression.",
    "warning": "⚠️ Please complete all selections.",
}

# ------------------------------------------------------------------
# Sidebar: language + navigation
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1>🧠</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Depression App</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    lang = st.selectbox("🌐 Pilih Bahasa / Choose Language", ["Indonesia", "English"], index=0)
    st.markdown("<hr>", unsafe_allow_html=True)

    if lang == "Indonesia":
        nav_title = "Navigasi"
        page_options = ["Beranda", "Tentang", "Prediksi Depresi"]
        pilih_halaman_label = "Pilih Halaman"
    else:
        nav_title = "Navigation"
        page_options = ["Home", "About", "Depression Prediction"]
        pilih_halaman_label = "Choose Page"

    st.markdown(f"<h4>{nav_title}</h4>", unsafe_allow_html=True)
    page_label = st.selectbox(pilih_halaman_label, page_options, index=0)
    st.markdown("<hr>", unsafe_allow_html=True)

# map page_label -> internal id
PAGE_MAP_ID = {"Beranda": "home", "Tentang": "about", "Prediksi Depresi": "predict"}
PAGE_MAP_EN = {"Home": "home", "About": "about", "Depression Prediction": "predict"}
page_id = PAGE_MAP_ID.get(page_label) if lang == "Indonesia" else PAGE_MAP_EN.get(page_label)

# choose language labels
LABELS = LABELS_ID if lang == "Indonesia" else LABELS_EN

# option mappings (display->canonical English used by encoders)
if lang == "Indonesia":
    gender_options = {"Laki-laki": "Male", "Perempuan": "Female"}
    sleep_options = {
        "Kurang dari 5 jam": "Less than 5 hours",
        "5-6 jam": "5-6 hours",
        "7-8 jam": "7-8 hours",
        "Lebih dari 8 jam": "More than 8 hours",
    }
    diet_options = {"Sehat": "Healthy", "Sedang": "Moderate", "Tidak sehat": "Unhealthy"}
    suicidal_options = {"Ya": "Yes", "Tidak": "No"}
    family_history_options = {"Ya": "Yes", "Tidak": "No"}

    # tambahan mapping skala (1–5 / 0–5)
    academic_pressure_map = {
        "Sangat ringan": 1,
        "Ringan": 2,
        "Sedang": 3,
        "Berat": 4,
        "Sangat berat": 5
    }
    study_satisfaction_map = {
        "Sangat tidak puas": 1,
        "Tidak puas": 2,
        "Netral": 3,
        "Puas": 4,
        "Sangat puas": 5
    }
    financial_stress_map = {
        "Tidak ada": 0,
        "Sangat rendah": 1,
        "Rendah": 2,
        "Sedang": 3,
        "Tinggi": 4,
        "Sangat tinggi": 5
    }

else:
    gender_options = {"Male": "Male", "Female": "Female"}
    sleep_options = {
        "Less than 5 hours": "Less than 5 hours",
        "5-6 hours": "5-6 hours",
        "7-8 hours": "7-8 hours",
        "More than 8 hours": "More than 8 hours",
    }
    diet_options = {"Healthy": "Healthy", "Moderate": "Moderate", "Unhealthy": "Unhealthy"}
    suicidal_options = {"Yes": "Yes", "No": "No"}
    family_history_options = {"Yes": "Yes", "No": "No"}

    # tambahan mapping skala (1–5 / 0–5)
    academic_pressure_map = {
        "Very light": 1,
        "Light": 2,
        "Moderate": 3,
        "Heavy": 4,
        "Very heavy": 5
    }
    study_satisfaction_map = {
        "Very dissatisfied": 1,
        "Dissatisfied": 2,
        "Neutral": 3,
        "Satisfied": 4,
        "Very satisfied": 5
    }
    financial_stress_map = {
        "None": 0,
        "Very low": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very high": 5
    }

# ------------------------------------------------------------------
# Page render helpers
# ------------------------------------------------------------------

def show_home():
    # Judul (centered)
    st.markdown(
        f'''<h1 style="text-align:center; font-size: 36px; font-weight: bold; margin-bottom: 2rem;">
            {LABELS["home_title"]}
        </h1>''',
        unsafe_allow_html=True,
    )

    # Gambar di bawah judul (centered pakai column)
    col_center = st.columns([3, 3, 3])[1]
    with col_center:
        st.image("Depresi.jpeg", width=600)

    # Spacer
    st.markdown("<br>", unsafe_allow_html=True)

    # Deskripsi (justify + max width)
    st.markdown(
        f'''
        <div class="custom-text" style="max-width: 900px; margin: 0 auto;">
            {HOME_DESC_ID if lang == "Indonesia" else HOME_DESC_EN}
        ''',
        unsafe_allow_html=True,
    )

    # Disclaimer card
    if lang == "Indonesia":
        st.markdown(
            '''
            <div style="background: #2c2f33; border-radius: 12px; padding: 1.5rem; margin-top: 2rem; 
                     border-left: 6px solid #5865F2; box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
                     max-width: 900px; margin-left: auto; margin-right: auto;">

              <div style="margin-bottom: 1.2rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">✨ <strong>Harap diingat:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  Prediksi ini bukan pengganti diagnosis profesional. Hasil yang ditampilkan hanya bersifat estimasi berdasarkan data input yang diberikan.
                </p>
              </div>

              <div style="margin-bottom: 1.2rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">💬 <strong>Jika kamu merasa kesulitan:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  Jangan ragu untuk mencari bantuan profesional seperti psikolog kampus, konselor, atau layanan kesehatan mental lainnya.
                </p>
              </div>

              <div>
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">💡 <strong>Tips:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  Gunakan aplikasi ini sebagai langkah awal untuk mengenali, memahami, dan menjaga kesehatan mentalmu!
                </p>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '''
            <div style="background: #2c2f33; border-radius: 12px; padding: 1.5rem; margin-top: 2rem; 
                        border-left: 6px solid #5865F2; box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
                        max-width: 900px; margin-left: auto; margin-right: auto;">

              <div style="margin-bottom: 1.2rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">✨ <strong>Note:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  This prediction is not a substitute for professional diagnosis. The results shown are only estimates based on your input.
                </p>
              </div>

              <div style="margin-bottom: 1.2rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">💬 <strong>If you feel distressed:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  Do not hesitate to seek professional help such as campus psychologists, counselors, or mental health services.
                </p>
              </div>

              <div>
                <h4 style="margin: 0; color: #ffffff; font-size: 20px;">💡 <strong>Tips:</strong></h4>
                <p style="margin: 0.3rem 0 0; color: #dcdcdc; font-size: 16px;">
                  Use this app as a first step to recognize, understand, and maintain your mental health!
                </p>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

def show_about():
    st.markdown(
        f'''<div class="custom-text">
            <h1 style="text-align:center; font-size: 36px; font-weight: bold; margin-bottom: 2rem;">
                {LABELS["about_title"]}
            </h1>
            {LABELS["about_desc"]}
        ''',
        unsafe_allow_html=True,
    )

def show_predict():
    st.title(LABELS["predict_title"])
    st.write("")
    st.caption(LABELS["predict_desc"])

    placeholder_txt = "- Pilih -" if lang == "Indonesia" else "- Select -"

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(LABELS["gender"], [placeholder_txt] + list(gender_options.keys()))
        
        # 👉 Age manual input
        age = st.number_input(LABELS["age"], min_value=18, max_value=34, step=1)

        academic_pressure = st.selectbox(
            LABELS["academic_pressure"],
            [placeholder_txt] + list(academic_pressure_map.keys())
        )

        study_satisfaction = st.selectbox(
            LABELS["study_satisfaction"],
            [placeholder_txt] + list(study_satisfaction_map.keys())
        )

        sleep_duration = st.selectbox(
            LABELS["sleep_duration"],
            [placeholder_txt] + list(sleep_options.keys())
        )

    with col2:
        dietary_habits = st.selectbox(LABELS["dietary_habits"], [placeholder_txt] + list(diet_options.keys()))
        suicidal_thoughts = st.selectbox(LABELS["suicidal_thoughts"], [placeholder_txt] + list(suicidal_options.keys()))
        study_hours = st.selectbox(LABELS["study_hours"], [placeholder_txt] + [str(i) for i in range(0, 13)])
        financial_stress = st.selectbox(LABELS["financial_stress"], [placeholder_txt] + list(financial_stress_map.keys()))
        family_history = st.selectbox(LABELS["family_history"], [placeholder_txt] + list(family_history_options.keys()))

    # Predict button ---------------------------------------------------------------
    if st.button(LABELS["predict_button"]):
        # Validasi input kosong
        if (
            gender == placeholder_txt
            or academic_pressure == placeholder_txt
            or study_satisfaction == placeholder_txt
            or sleep_duration == placeholder_txt
            or dietary_habits == placeholder_txt
            or suicidal_thoughts == placeholder_txt
            or study_hours == placeholder_txt
            or financial_stress == placeholder_txt
            or family_history == placeholder_txt
        ):
            st.warning(LABELS["warning"])
            return

        if not all([model is not None, scaler is not None, label_encoders is not None]):
            if lang == "Indonesia":
                st.error("Komponen model belum lengkap.")
            else:
                st.error("Model components are not fully loaded.")
            return

        try:
            input_data = pd.DataFrame([
                {
                    'Gender': label_encoders['Gender'].transform([gender_options[gender]])[0],
                    'Age': int(age),
                    'Academic Pressure': academic_pressure_map[academic_pressure],
                    'Study Satisfaction': study_satisfaction_map[study_satisfaction],
                    'Sleep Duration': label_encoders['Sleep Duration'].transform([sleep_options[sleep_duration]])[0],
                    'Dietary Habits': label_encoders['Dietary Habits'].transform([diet_options[dietary_habits]])[0],
                    'Have you ever had suicidal thoughts ?': label_encoders['Have you ever had suicidal thoughts ?'].transform([suicidal_options[suicidal_thoughts]])[0],
                    'Study Hours': int(study_hours),
                    'Financial Stress': financial_stress_map[financial_stress],
                    'Family History of Mental Illness': label_encoders['Family History of Mental Illness'].transform([family_history_options[family_history]])[0],
                }
            ])
        except Exception as e:
            if lang == "Indonesia":
                st.error(f"Terjadi kesalahan saat encoding input: {e}")
            else:
                st.error(f"An error occurred during input encoding: {e}")
            return

        # Scale semua fitur lalu kembalikan Age ke nilai asli ------------------------
        try:
            age_raw = input_data["Age"].values

            input_scaled = pd.DataFrame(
                scaler.transform(input_data),
                columns=input_data.columns
            )

            # overwrite Age dengan nilai asli
            input_scaled["Age"] = age_raw

            # lakukan prediksi
            prediction = model.predict(input_scaled)[0]

        except Exception as e:
            if lang == "Indonesia":
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
            else:
                st.error(f"An error occurred during prediction: {e}")
            return

        # Display result -----------------------------------------------------------
        if prediction == 1:
            if lang == "Indonesia":
                st.toast("🚨 Prediksi: Mahasiswa ini berisiko mengalami depresi.", icon="⚠️")
                st.error("🚨 **Hasil Prediksi: Mahasiswa ini berisiko mengalami depresi.**")
                st.markdown(
                    """
                    ### 💡 Rekomendasi Langkah Selanjutnya:
                    - 🧠 **Cari bantuan profesional:** Konsultasi dengan psikolog/psikiater sangat dianjurkan.
                    - 🤝 **Buka diri:** Bicarakan perasaan Anda dengan teman, keluarga, atau mentor yang dapat Anda percaya.  
                    - 🧘 **Coba teknik relaksasi:** Meditasi 5 menit/hari, pernapasan dalam, atau olahraga ringan.  
                    - 🎶 **Musik penyemangat:** Dengarkan musik positif untuk mengubah suasana hati.
                    """
                )
                st.info("💬 *Motivasi:* _“Setiap badai pasti berlalu. Bantuan selalu ada jika kita mau mencarinya.”_")
                st.markdown(
                    """
                    #### 📞 Layanan Konsultasi & Bantuan:
                    1. **Yayasan Pulih** – Kontak admin kami melalui WA: +62 811 843 6633 (Chat only)  
                    2. **SEJIWA Kemensos** – Layanan dukungan psikososial gratis: 119 ext. 8  
                    3. **Halo Kemenkes** – Call Center 24 jam: 1500-567  
                    ---
                    #### ✨Ingat:
                    Kamu tidak sendirian. Ada banyak orang yang peduli dan siap membantu.
                    """
                )
            else:
                st.toast("🚨 Prediction: This student is likely experiencing depression.", icon="⚠️")
                st.error("🚨 **Prediction Result: This student is likely experiencing depression.**")
                st.markdown(
                    """
                    ### 💡 Recommended Next Steps:
                    - 🧠 **Seek professional help:** Consult a psychologist or psychiatrist as soon as possible.
                    - 🤝 **Open up:** Share your feelings with trusted friends, family, or a mentor.  
                    - 🧘 **Try relaxation techniques:** Meditate for 5 minutes a day, deep breathing, or light exercise.  
                    - 🎶 **Listen to uplifting music:** Music can help shift your mood positively.
                    """
                )
                st.info("💬 *Motivation:* _“Every storm passes. Help is always there when you're willing to seek it.”_")
                st.markdown(
                    """
                    #### 📞 Support & Helpline (Indonesia):
                    1. **Pulih Foundation** – WhatsApp: +62 811 843 6633 (Chat only)  
                    2. **SEJIWA by Ministry of Social Affairs** – Free psychosocial support: 119 ext. 8  
                    3. **Halo Kemenkes** – 24/7 Call Center: 1500-567  
                    ---
                    #### ✨Remember:
                    You are not alone. There are people who care and want to help you.
                    """
                )
        else:
            if lang == "Indonesia":
                st.toast("✅ Prediksi: Mahasiswa ini tidak menunjukkan indikasi depresi.", icon="✅")
                st.success("✅ **Hasil Prediksi: Mahasiswa ini tidak menunjukkan indikasi depresi.**")
                st.markdown(
                    """
                    ### 🎯 Rekomendasi Gaya Hidup Sehat:
                    - 😴 **Tidur cukup:** Usahakan 7-8 jam/hari untuk kesehatan mental.  
                    - 🏃 **Olahraga ringan:** 15-30 menit per hari untuk mengurangi stres.  
                    - 📚 **Manajemen waktu belajar:** Jangan terlalu memaksakan diri, gunakan teknik *Pomodoro*.  
                    - 👥 **Sosialisasi:** Berkumpul dengan teman/keluarga untuk menjaga mood positif.  
                    - 🎨 **Aktivitas kreatif:** Melukis, menulis, atau mendengarkan musik bisa menjadi terapi.
                    """
                )
                st.info("🌟 *Motivasi:* _“Sehat mental adalah kunci produktivitas. Jaga dirimu, karena kamu berharga.”_")
                st.markdown(
                    """
                    #### 📌 Aktivitas yang Disarankan:
                    - 🧘 **Meditasi atau yoga** 10 menit sehari.  
                    - 📖 **Membaca buku inspirasi atau mendengarkan podcast positif.**  
                    - ☀️ **Berjemur di pagi hari** untuk vitamin D alami & mood booster.  
                    """
                )
            else:
                st.toast("✅ Prediction: This student does not show signs of depression.", icon="✅")
                st.success("✅ **Prediction Result: This student does not show signs of depression.**")
                st.markdown(
                    """
                    ### 🎯 Healthy Lifestyle Recommendations:
                    - 😴 **Get enough sleep:** Aim for 7–8 hours per day for better mental health.  
                    - 🏃 **Exercise lightly:** Do 15–30 minutes of light activity to reduce stress.  
                    - 📚 **Manage study time:** Don’t overwork; use techniques like *Pomodoro*.  
                    - 👥 **Socialize:** Spend time with friends or family to maintain a positive mood.  
                    - 🎨 **Do creative things:** Painting, journaling, or music can help you feel better.
                    """
                )
                st.info("🌟 *Motivation:* _“Mental wellness is key to productivity. Take care of yourself—because you're worth it.”_")
                st.markdown(
                    """
                    #### 📌 Suggested Activities:
                    - 🧘 **Meditation or yoga** for 10 minutes daily  
                    - 📖 **Read inspiring books or listen to positive podcasts**  
                    - ☀️ **Get morning sun exposure** for natural vitamin D & mood boost  
                    """
                )

# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------
if page_id == "home":
    show_home()
elif page_id == "about":
    show_about()
elif page_id == "predict":
    show_predict()
else:  # fallback safety
    st.error("Halaman tidak ditemukan / Page not found.")
