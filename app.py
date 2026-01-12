import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚗", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #E0E0E0;
}
h1, h2, h3 {
    text-align: center;
    color: #00FF88;
}
.prediction-box {
    animation: fadeIn 1.5s ease-in-out;
    text-align: center;
    color: #00FF00;
    font-size: 30px;
    font-weight: bold;
    margin-top: -10px;
}
.emoji {
    display: inline-block;
    animation: bounce 2s infinite;
    font-size: 40px;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
    40% {transform: translateY(-10px);}
    60% {transform: translateY(-5px);}
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
try:
    model = joblib.load("accident_model.pkl")
except:
    st.error("❌ Model file not found! Please make sure 'accident_model.pkl' is in the same folder.")
    st.stop()

# -------------------- TITLE --------------------
st.title("🚗 Accident Severity Prediction System")
st.markdown("### Predict how severe an accident could be based on various real-world factors.")

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🧠 Enter Accident Details")

day = st.sidebar.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
light = st.sidebar.selectbox("Light Conditions", ['Daylight', 'Darkness'])
sex = st.sidebar.selectbox("Sex of Driver (0=Male, 1=Female)", [0, 1])
vehicle = st.sidebar.number_input("Vehicle Type (0-10)", 0, 10)
speed = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 40)
pedestrian = st.sidebar.number_input("Pedestrian Crossing (0-5)", 0, 5)
road = st.sidebar.number_input("Road Type (1-6)", 1, 6)
special = st.sidebar.number_input("Special Conditions (0-3)", 0, 3)
passengers = st.sidebar.number_input("Number of Passengers", 1, 10)

# -------------------- ENCODING --------------------
day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
light_map = {'Daylight': 0, 'Darkness': 1}

day_num = day_map[day]
light_num = light_map[light]

# -------------------- PREDICTION --------------------
if st.sidebar.button("🚀 Predict Severity"):
    features = pd.DataFrame([[day_num, light_num, sex, vehicle, speed, pedestrian, road, special, passengers]],
                            columns=['Day_of_Week','Light_Conditions','Sex_Of_Driver','Vehicle_Type',
                                     'Speed_limit','Pedestrian_Crossing','Road_Type','Special_Conditions_at_Site',
                                     'Number_of_Pasengers'])
    try:
        prediction = model.predict(features)
        result_value = prediction[0]

        # map numerical prediction to string labels
        label_map = {0: "Slight", 1: "Serious", 2: "Fatal"}
        result = label_map.get(result_value, "Unknown")

        # Emoji or Image based on prediction
        if result == "Slight":
            emoji = "😊"
            img = "https://cdn-icons-png.flaticon.com/512/1048/1048310.png"
        elif result == "Serious":
            emoji = "😟"
            img = "https://cdn-icons-png.flaticon.com/512/942/942799.png"
        else:
            emoji = "🚨"
            img = "https://cdn-icons-png.flaticon.com/512/564/564619.png"

        st.markdown(f"<div class='prediction-box'>Predicted Accident Severity: {result.upper()} <span class='emoji'>{emoji}</span></div>", unsafe_allow_html=True)
        st.image(img, width=180)
        st.success("✅ Prediction Complete!")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------- DATA INSIGHTS --------------------
st.markdown("---")
st.header("📊 Data Insights (Visual Analysis)")

try:
    df = pd.read_csv("accidents_india.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accidents by Day of Week")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Day_of_Week', hue='Accident_Severity', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Effect of Speed Limit on Severity")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Accident_Severity', y='Speed_limit', ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Road Type vs Severity")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Road_Type', hue='Accident_Severity', ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Light Conditions Effect")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Light_Conditions', hue='Accident_Severity', ax=ax)
        st.pyplot(fig)

except Exception as e:
    st.warning("⚠️ Dataset not found or invalid for plotting. Please ensure 'accidents_india.csv' exists.")
