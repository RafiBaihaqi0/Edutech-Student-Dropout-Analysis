import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/model_prediksi_dropout.joblib"
ENCODER_PATH = "model/label_encoder_status.joblib"

FEATURE_ORDER = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification",
    "Previous_qualification_grade", "Nacionality", "Mothers_qualification",
    "Fathers_qualification", "Mothers_occupation", "Fathers_occupation",
    "Admission_grade", "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
    "Age_at_enrollment", "International", "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]

MARITAL_STATUS_OPTIONS = {
    1: "Single", 2: "Menikah", 3: "Duda/Janda",
    4: "Bercerai", 5: "Persatuan Faktual", 6: "Berpisah Secara Hukum",
}

YA_TIDAK = {1: "Ya", 0: "Tidak"}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


def yes_no_input(label, key, default_index=0):
    pilihan = st.selectbox(label, options=list(YA_TIDAK.keys()),
                            format_func=lambda x: YA_TIDAK[x],
                            index=default_index, key=key)
    return pilihan


def main():
    st.set_page_config(page_title="Prediksi Status Mahasiswa", page_icon="🎓", layout="centered")
    st.title("🎓 Prototype Prediksi Status Mahasiswa")
    st.caption(
        "Prototype sistem machine learning untuk memprediksi status mahasiswa "
        "(Dropout / Enrolled / Graduate) pada Jaya Jaya Institut berdasarkan model "
        "Random Forest yang telah dilatih pada tahap analisis data sebelumnya."
    )

    model, label_encoder = load_artifacts()

    with st.form("form_prediksi"):
        st.subheader("Data Pendaftaran & Demografi")
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox(
                "Status Pernikahan", options=list(MARITAL_STATUS_OPTIONS.keys()),
                format_func=lambda x: MARITAL_STATUS_OPTIONS[x]
            )
            application_mode = st.number_input("Kode Moda Pendaftaran (Application Mode)", 1, 57, 17)
            application_order = st.number_input("Urutan Pilihan Pendaftaran", 0, 9, 1)
            course = st.number_input("Kode Program Studi (Course)", 33, 9991, 9238)
            daytime_evening = st.selectbox(
                "Waktu Perkuliahan", options=[1, 0],
                format_func=lambda x: "Siang (Daytime)" if x == 1 else "Malam (Evening)"
            )
            previous_qualification = st.number_input("Kode Kualifikasi Sebelumnya", 1, 43, 1)
            previous_qualification_grade = st.number_input(
                "Nilai Kualifikasi Sebelumnya", 95.0, 190.0, 133.1, step=0.1
            )
            nacionality = st.number_input("Kode Kewarganegaraan", 1, 109, 1)
        with col2:
            mothers_qualification = st.number_input("Kode Kualifikasi Ibu", 1, 44, 19)
            fathers_qualification = st.number_input("Kode Kualifikasi Ayah", 1, 44, 19)
            mothers_occupation = st.number_input("Kode Pekerjaan Ibu", 0, 194, 5)
            fathers_occupation = st.number_input("Kode Pekerjaan Ayah", 0, 195, 7)
            admission_grade = st.number_input("Nilai Penerimaan (Admission Grade)", 95.0, 190.0, 126.1, step=0.1)
            gender = st.selectbox("Jenis Kelamin", options=[1, 0],
                                   format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
            age_at_enrollment = st.number_input("Usia Saat Pendaftaran", 17, 70, 20)
            international = yes_no_input("Mahasiswa Internasional?", "international")

        st.subheader("Kondisi Sosial-Ekonomi")
        col3, col4 = st.columns(2)
        with col3:
            displaced = yes_no_input("Berstatus Displaced?", "displaced")
            educational_special_needs = yes_no_input("Berkebutuhan Khusus?", "special_needs")
        with col4:
            debtor = yes_no_input("Berstatus Debitur?", "debtor")
            tuition_fees_up_to_date = yes_no_input("SPP Dibayar Tepat Waktu?", "tuition", default_index=0)
            scholarship_holder = yes_no_input("Penerima Beasiswa?", "scholarship")

        st.subheader("Performa Akademik Semester 1")
        col5, col6 = st.columns(2)
        with col5:
            cu1_credited = st.number_input("SKS Diakui (Credited)", 0, 20, 0, key="cu1_credited")
            cu1_enrolled = st.number_input("SKS Diambil (Enrolled)", 0, 26, 6, key="cu1_enrolled")
            cu1_evaluations = st.number_input("Jumlah Evaluasi", 0, 45, 8, key="cu1_eval")
        with col6:
            cu1_approved = st.number_input("SKS Disetujui (Approved)", 0, 26, 5, key="cu1_approved")
            cu1_grade = st.number_input("Rata-rata Nilai", 0.0, 20.0, 12.3, step=0.1, key="cu1_grade")
            cu1_without_eval = st.number_input("Tanpa Evaluasi", 0, 12, 0, key="cu1_without")

        st.subheader("Performa Akademik Semester 2")
        col7, col8 = st.columns(2)
        with col7:
            cu2_credited = st.number_input("SKS Diakui (Credited)", 0, 19, 0, key="cu2_credited")
            cu2_enrolled = st.number_input("SKS Diambil (Enrolled)", 0, 23, 6, key="cu2_enrolled")
            cu2_evaluations = st.number_input("Jumlah Evaluasi", 0, 33, 8, key="cu2_eval")
        with col8:
            cu2_approved = st.number_input("SKS Disetujui (Approved)", 0, 20, 5, key="cu2_approved")
            cu2_grade = st.number_input("Rata-rata Nilai", 0.0, 20.0, 12.2, step=0.1, key="cu2_grade")
            cu2_without_eval = st.number_input("Tanpa Evaluasi", 0, 12, 0, key="cu2_without")

        st.subheader("Indikator Ekonomi Makro")
        col9, col10, col11 = st.columns(3)
        with col9:
            unemployment_rate = st.number_input("Tingkat Pengangguran (%)", 7.6, 16.2, 11.1, step=0.1)
        with col10:
            inflation_rate = st.number_input("Tingkat Inflasi (%)", -0.8, 3.7, 1.4, step=0.1)
        with col11:
            gdp = st.number_input("GDP", -4.06, 3.51, 0.32, step=0.01)

        submitted = st.form_submit_button("Prediksi Status Mahasiswa")

    if submitted:
        input_data = {
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Course": course,
            "Daytime_evening_attendance": daytime_evening,
            "Previous_qualification": previous_qualification,
            "Previous_qualification_grade": previous_qualification_grade,
            "Nacionality": nacionality,
            "Mothers_qualification": mothers_qualification,
            "Fathers_qualification": fathers_qualification,
            "Mothers_occupation": mothers_occupation,
            "Fathers_occupation": fathers_occupation,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": educational_special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition_fees_up_to_date,
            "Gender": gender,
            "Scholarship_holder": scholarship_holder,
            "Age_at_enrollment": age_at_enrollment,
            "International": international,
            "Curricular_units_1st_sem_credited": cu1_credited,
            "Curricular_units_1st_sem_enrolled": cu1_enrolled,
            "Curricular_units_1st_sem_evaluations": cu1_evaluations,
            "Curricular_units_1st_sem_approved": cu1_approved,
            "Curricular_units_1st_sem_grade": cu1_grade,
            "Curricular_units_1st_sem_without_evaluations": cu1_without_eval,
            "Curricular_units_2nd_sem_credited": cu2_credited,
            "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
            "Curricular_units_2nd_sem_evaluations": cu2_evaluations,
            "Curricular_units_2nd_sem_approved": cu2_approved,
            "Curricular_units_2nd_sem_grade": cu2_grade,
            "Curricular_units_2nd_sem_without_evaluations": cu2_without_eval,
            "Unemployment_rate": unemployment_rate,
            "Inflation_rate": inflation_rate,
            "GDP": gdp,
        }

        input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

        prediksi_encoded = model.predict(input_df)[0]
        prediksi_label = label_encoder.inverse_transform([prediksi_encoded])[0]
        probabilitas = model.predict_proba(input_df)[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if prediksi_label == "Dropout":
            st.error(f"Status Prediksi: **{prediksi_label}**")
        elif prediksi_label == "Enrolled":
            st.warning(f"Status Prediksi: **{prediksi_label}**")
        else:
            st.success(f"Status Prediksi: **{prediksi_label}**")

        proba_df = pd.DataFrame({
            "Status": label_encoder.classes_,
            "Probabilitas": probabilitas,
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

        st.dataframe(
            proba_df.style.format({"Probabilitas": "{:.2%}"}),
            hide_index=True, use_container_width=True
        )
        st.bar_chart(proba_df.set_index("Status"))


if __name__ == "__main__":
    main()
