import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import time
import random
import requests
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import xgboost



# Set page configuration
st.set_page_config(page_title="Health & Wellness App", page_icon="💙", layout="wide")


# Load models
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: Model file {model_path} not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

models = {
    "diabetes": load_model("Models/xgboost_diabetes_model.joblib"),
    "heart_disease": load_model("Models/xgboost_heartdisease_model.joblib"),
    "lung_cancer": load_model("Models/xgboost_Lung_Cancer_model.joblib"),
    "parkinsons": load_model("Models/xgboost_parkinsons_model.joblib"),
    "hypo_thyroid": load_model("Models/xgboost_hypothyroid_model.joblib")
}

# Sidebar navigation
with st.sidebar:
    #st.image("images/logo.png", width=180)  # Your app's logo
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Health Dashboard</h1>", unsafe_allow_html=True)

    # 🌟 Stylish Navigation Menu with Icons
    selected = option_menu(
        "Navigation",
        options=[
            "🏠 Home",
            "🩺 Disease Prediction",
            "💪 Fitness Tracking",
            "🧠 Mental Health",
            "🥗 Nutrition Guidance",
            "🤖 AI Assistant"
        ],
        icons=["house", "stethoscope", "activity", "brain", "emoji-food-beverage", "robot"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "nav-link": {
                "font-size": "18px",
                "color": "black",
                "padding": "10px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#d3f9d8",
            },
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
        }
    )

    steps_today = random.randint(5000, 10000)
    calories_burned = random.randint(1500, 2500)
    sleep_hours = round(random.uniform(4.5, 9), 1)

    # 📊 Quick Stats Section (Real-Time Simulated)
    st.markdown("### 📊 Quick Health Stats")
    st.metric(label="🔋 Steps Today", value=f"{steps_today}", delta=random.randint(-500, 500))
    st.metric(label="🔥 Calories Burned", value=f"{calories_burned} kcal", delta=random.randint(-200, 200))
    st.metric(label="💤 Sleep Hours", value=f"{sleep_hours} hrs", delta=round(random.uniform(-1, 1), 1))

    # 📌 Expandable Sections for More Insights
    with st.expander("🔎 About This App"):
        st.write("""
            This AI-powered **Health Dashboard** helps you:
            - Predict & analyze diseases 🩺
            - Track workouts & fitness 💪
            - Monitor mental well-being 🧠
            - Get personalized nutrition plans 🥗
            - Chat with an AI assistant 🤖
            """)

    with st.expander("🆕 What's New?"):
        st.write("🔥 **New features added:** AI Health Chatbot, Fitness API integration, and more!")

    # 📞 Support Contact
    st.markdown("---")
    st.markdown("📩 **Need Help?** Contact [Support](mailto:support@yourapp.com)")

# Load environment variables
env_path = r"HealthApp\apikey.env"
load_dotenv(env_path)

# Get API key
api_key = st.secrets["api"]["GOOGLE_GEMINI_API_KEY"]



# Home Page
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if selected == "🏠 Home":

    # Display an animated banner
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏥 Your Personal Health & Wellness Assistant</h1>",
                unsafe_allow_html=True)

    st.markdown(
        "<h3 style='text-align: center; color: #AAAAAA; text-shadow: 2px 2px 10px rgba(0, 255, 0, 0.3);'>Empowering you with AI-driven health insights!</h3>",
        unsafe_allow_html=True
    )
    lottie_animation = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_puciaact.json")
    st_lottie(lottie_animation, height=300, key="health_animation")


    # Brief description
    st.write(
        """
        Welcome to your ultimate health companion! This app helps you:
        - 🏋️ Track fitness progress
        - 🍏 Get personalized diet recommendations
        - 🏥 Predict and assess disease risks
        - 💬 Interact with an AI-powered chatbot & voice assistant
        - 📊 Receive AI-driven health insights and reports
        """
    )

    # Footer message
    st.markdown(
        "<p style='text-align: center; color: #888;'>Your health, your future – take charge today! 💪</p>",
        unsafe_allow_html=True
    )

# Disease Prediction Page
elif selected == "🩺 Disease Prediction":
    st.title("Disease Prediction")
    st.write("Select a disease and enter details for prediction.")

    disease = st.selectbox("Select a Disease",
                           ["Diabetes", "Heart Disease", "Lung Cancer", "Parkinson's", "Hypo-Thyroid"])

    if disease:
        st.subheader(f"Enter details for {disease} prediction")

        input_data = []

        if disease == "Diabetes":
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0)
                glucose = st.number_input("Glucose Level", min_value=0)
                bp = st.number_input("Blood Pressure", min_value=0)
                bmi = st.number_input("BMI", min_value=0.0)
            with col2:
                skin_thickness = st.number_input("Skin Thickness", min_value=0)
                insulin = st.number_input("Insulin Level", min_value=0)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
                age = st.number_input("Age", min_value=0)

            input_data = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]

        elif disease == "Heart Disease":
            # Split into two columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=0)
                sex = st.radio("Sex", ["Male", "Female"])
                sex = 1 if sex == "Male" else 0  # Convert to numerical
                cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
                fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
                fbs = 1 if fbs == "Yes" else 0
                restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2)

            with col2:
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
                exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
                exang = 1 if exang == "Yes" else 0
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1)
                slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2)
                ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3)
                thal = st.number_input("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", min_value=0,
                                       max_value=2)

            # Collect input data into a list
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        elif disease == "Lung Cancer":
            # Two columns for better UI
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=0)
                gender = st.radio("Gender", ["Male", "Female"])
                gender = 1 if gender == "Male" else 0

                smoking = st.radio("Smoking?", ["No", "Yes"])
                smoking = 1 if smoking == "Yes" else 0

                yellow_fingers = st.radio("Yellow Fingers?", ["No", "Yes"])
                yellow_fingers = 1 if yellow_fingers == "Yes" else 0

                anxiety = st.radio("Anxiety?", ["No", "Yes"])
                anxiety = 1 if anxiety == "Yes" else 0

                peer_pressure = st.radio("Peer Pressure?", ["No", "Yes"])
                peer_pressure = 1 if peer_pressure == "Yes" else 0

                chronic_disease = st.radio("Chronic Disease?", ["No", "Yes"])
                chronic_disease = 1 if chronic_disease == "Yes" else 0

                fatigue = st.radio("Fatigue?", ["No", "Yes"])
                fatigue = 1 if fatigue == "Yes" else 0

            with col2:
                allergy = st.radio("Allergy?", ["No", "Yes"])
                allergy = 1 if allergy == "Yes" else 0

                wheezing = st.radio("Wheezing?", ["No", "Yes"])
                wheezing = 1 if wheezing == "Yes" else 0

                alcohol_consuming = st.radio("Alcohol Consumption?", ["No", "Yes"])
                alcohol_consuming = 1 if alcohol_consuming == "Yes" else 0

                coughing = st.radio("Coughing?", ["No", "Yes"])
                coughing = 1 if coughing == "Yes" else 0

                shortness_of_breath = st.radio("Shortness of Breath?", ["No", "Yes"])
                shortness_of_breath = 1 if shortness_of_breath == "Yes" else 0

                swallowing_difficulty = st.radio("Swallowing Difficulty?", ["No", "Yes"])
                swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0

                chest_pain = st.radio("Chest Pain?", ["No", "Yes"])
                chest_pain = 1 if chest_pain == "Yes" else 0

            # Collect input data
            input_data = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                          allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty,
                          chest_pain]


        elif disease == "Parkinson's":
            col1, col2, col3 = st.columns(3)

            with col1:
                fo = st.number_input("MDVP:Fo (Hz)", min_value=0.0)
                fhi = st.number_input("MDVP:Fhi (Hz)", min_value=0.0)
                flo = st.number_input("MDVP:Flo (Hz)", min_value=0.0)
                jitter_percent = st.number_input("MDVP:Jitter (%)", min_value=0.0)
                jitter_abs = st.number_input("MDVP:Jitter (Abs)", min_value=0.0)
                rap = st.number_input("MDVP:RAP", min_value=0.0)
                ppq = st.number_input("MDVP:PPQ", min_value=0.0)
                ddp = st.number_input("Jitter:DDP", min_value=0.0)

            with col2:
                shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
                shimmer_db = st.number_input("MDVP:Shimmer (dB)", min_value=0.0)
                apq3 = st.number_input("Shimmer:APQ3", min_value=0.0)
                apq5 = st.number_input("Shimmer:APQ5", min_value=0.0)
                apq = st.number_input("MDVP:APQ", min_value=0.0)
                dda = st.number_input("Shimmer:DDA", min_value=0.0)
                nhr = st.number_input("NHR", min_value=0.0)
                hnr = st.number_input("HNR", min_value=0.0)

            with col3:
                rpde = st.number_input("RPDE", min_value=0.0)
                dfa = st.number_input("DFA", min_value=0.0)
                spread1 = st.number_input("Spread1", min_value=0.0)
                spread2 = st.number_input("Spread2", min_value=0.0)
                d2 = st.number_input("D2", min_value=0.0)
                ppe = st.number_input("PPE", min_value=0.0)

            # Collect input data
            input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                          apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]


        elif disease == "Hypo-Thyroid":
            col1, col2 = st.columns(2)


            with col1:
                age = st.number_input("Age", min_value=0, max_value=120)
                sex = st.selectbox("Sex", ["Female", "Male"])
                sex = 1 if sex == "Male" else 0
                on_thyroxine = st.radio("On Thyroxine?", ["No", "Yes"])
                on_thyroxine = 1 if on_thyroxine == "Yes" else 0
                tsh = st.number_input("TSH Level", min_value=0.0, format="%.2f")

            with col2:
                t3_measured = st.radio("T3 Measured?", ["No", "Yes"])
                t3_measured = 1 if t3_measured == "Yes" else 0
                t3 = st.number_input("T3 Level", min_value=0.0, format="%.2f")
                tt4 = st.number_input("TT4 Level", min_value=0.0, format="%.2f")

            # Store input values in a list
            input_data = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]

        # === PREDICTION BUTTON ===
        if st.button(f"Predict {disease}"):
            # Normalize disease key
            disease_key = disease.replace("'", "").replace("-", "_").replace(" ", "_").lower()

            # Check if the model is loaded
            if disease_key not in models or models[disease_key] is None:
                st.warning("⚠️ Model is not loaded! Please load the models.")

            # Check if all input fields are filled
            elif any(v is None for v in input_data):
                st.error("⚠️ Please fill in all required fields before predicting!")

            else:
                # Perform prediction
                prediction = models[disease_key].predict([input_data])
                if prediction[0] == 1:
                    st.success(f"✅ **{disease} Detected**")
                    st.warning("⚠️ Please consult a doctor for further evaluation and guidance.")
                else:
                    st.success(f"🟢 **No Signs of {disease} Detected**")
                    st.info("✅ Maintain a healthy lifestyle to prevent future risks.")




# Other sections placeholders
elif selected == "💪 Fitness Tracking":
    st.title("💪 Fitness Tracking")
    st.write("Log your daily workouts and track progress.")

    # File to store workout data
    CSV_FILE = "fitness_logs.csv"

    # Load existing data or create a new dataframe
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Exercise", "Duration (min)", "Calories Burned"])

    # --- USER INPUT FORM ---
    st.subheader("🏋️ Log Your Workout")
    cols = st.columns(2)

    # Exercise selection: dropdown with common exercises
    exercise_options = ["Running", "Cycling", "Yoga", "Swimming", "Weight Lifting", "Jump Rope", "Dancing", "Custom"]
    exercise = cols[0].selectbox("Exercise Type", exercise_options)

    # If "Custom" is selected, allow text input
    if exercise == "Custom":
        exercise = cols[0].text_input("Enter Custom Exercise", placeholder="E.g., Kickboxing, Pilates")

    # Duration input
    duration = cols[1].number_input("Duration (minutes)", min_value=1, step=5)

    # Calories estimation (if not entered manually)
    calories_per_minute = {
        "Running": 10, "Cycling": 8, "Yoga": 4, "Swimming": 9, "Weight Lifting": 6,
        "Jump Rope": 12, "Dancing": 7
    }
    estimated_calories = duration * calories_per_minute.get(exercise, 5)
    calories = st.number_input("Calories Burned (Optional)", min_value=0, value=int(estimated_calories))

    # Date input
    date = st.date_input("Date", datetime.date.today())

    # Log Workout Button
    if st.button("📌 Log Workout"):
        if exercise:  # Ensure exercise name is not empty
            new_entry = pd.DataFrame([[date, exercise, duration, calories]], columns=df.columns)
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            st.success("✅ Workout logged successfully!")
            st.rerun()
        else:
            st.warning("⚠️ Please enter the exercise type.")

    # --- WORKOUT HISTORY ---
    st.subheader("📊 Your Workout History")
    if not df.empty:
        st.dataframe(df.sort_values("Date", ascending=False))

        # Delete specific workouts
        delete_cols = st.columns(2)
        selected_date = delete_cols[0].date_input("📅 Select Date to Delete")
        selected_exercise = delete_cols[1].text_input("🏋️ Enter Exercise to Delete")

        if st.button("🗑️ Delete Workout"):
            df = df[(df["Date"] != str(selected_date)) | (df["Exercise"] != selected_exercise)]
            df.to_csv(CSV_FILE, index=False)
            st.success("✅ Workout deleted successfully!")
            st.rerun()

        # Clear all history
        if st.button("⚠️ Clear All Workout History"):
            os.remove(CSV_FILE)
            df = pd.DataFrame(columns=["Date", "Exercise", "Duration (min)", "Calories Burned"])
            st.success("✅ All workout history cleared!")
            st.rerun()

        # --- WORKOUT VISUALIZATION ---
        st.subheader("📈 Workout Trends")

        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Calories burned over time
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df, x="Date", y="Calories Burned", marker="o", color="red", ax=ax)
        ax.set_title("🔥 Calories Burned Over Time", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Calories Burned")
        ax.grid(True)
        st.pyplot(fig)

        # Workout count per type
        workout_counts = df["Exercise"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=workout_counts.index, y=workout_counts.values, ax=ax, palette="viridis")
        ax.set_title("🏋️‍♂️ Most Frequent Workouts", fontsize=14)
        ax.set_xlabel("Exercise Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    else:
        st.info("ℹ️ No workouts logged yet.")


elif selected == "🧠 Mental Health":

    # --- PAGE TITLE ---
    st.title("🧠 Mental Health Support")
    st.write("Track your mood and get personalized relaxation tips.")

    # --- MOOD TRACKING ---
    MOOD_CSV = "mood_logs.csv"

    # Load mood logs
    if os.path.exists(MOOD_CSV):
        mood_df = pd.read_csv(MOOD_CSV)
        if "Date" in mood_df.columns:
            mood_df["Date"] = pd.to_datetime(mood_df["Date"], errors='coerce').dt.date
        else:
            mood_df = pd.DataFrame(columns=["Date", "Mood"])
    else:
        mood_df = pd.DataFrame(columns=["Date", "Mood"])

    st.write("### How do you feel today?")
    cols = st.columns(5)
    mood_options = ["😊 Happy", "😞 Sad", "😰 Stressed", "😟 Anxious", "😌 Relaxed"]

    # Mood styles for better UI
    mood_styles = {
        "😊 Happy": "background-color:#FFD700; color:black; font-weight:bold; padding:8px; border-radius:10px;",
        "😞 Sad": "background-color:#4682B4; color:white; font-weight:bold; padding:8px; border-radius:10px;",
        "😰 Stressed": "background-color:#FF4500; color:white; font-weight:bold; padding:8px; border-radius:10px;",
        "😟 Anxious": "background-color:#8B0000; color:white; font-weight:bold; padding:8px; border-radius:10px;",
        "😌 Relaxed": "background-color:#32CD32; color:black; font-weight:bold; padding:8px; border-radius:10px;"
    }

    # Ensure session state exists for mood selection
    if "selected_mood" not in st.session_state:
        st.session_state["selected_mood"] = None

    # Mood selection using buttons
    for i, mood_option in enumerate(mood_options):
        if cols[i].button(mood_option, key=f"mood_{i}"):
            st.session_state["selected_mood"] = mood_option

    # Get the selected mood from session state
    mood = st.session_state["selected_mood"]

    # Display selected mood with styling
    if mood:
        st.markdown(
            f'<div style="{mood_styles[mood]}; text-align:center;">{mood}</div>',
            unsafe_allow_html=True
        )

    # Date input
    date = st.date_input("Select Date", datetime.date.today())

    # Log mood
    if mood and st.button("Log Mood", key="log_mood"):
        #date_str = str(date)
        # Check if the mood for this date is already logged
        if not ((mood_df["Date"] == date) & (mood_df["Mood"] == mood)).any():
            new_mood_entry = pd.DataFrame([[date, mood]], columns=mood_df.columns)
            mood_df = pd.concat([mood_df, new_mood_entry], ignore_index=True)
            mood_df.to_csv(MOOD_CSV, index=False)
            st.success(f"✅ Mood logged successfully: {mood}")
        else:
            st.warning("⚠️ You've already logged this mood for today.")

    # --- RELAXATION TIPS ---
    st.subheader("💡 Relaxation Tips for You")

    mood_tips = {
        "😊 Happy": "✨ Keep doing what makes you happy! Share your positivity with others! 😊",
        "😞 Sad": "🎵 Try listening to music, journaling your thoughts, or talking to a friend. 💙",
        "😰 Stressed": "🧘 Take deep breaths, practice meditation, or do some light stretching.",
        "😟 Anxious": "🌿 Try guided breathing exercises, mindfulness meditation, or a short walk.",
        "😌 Relaxed": "🎶 Maintain your calm with gratitude journaling or soft music."
    }

    if mood:
        st.markdown(f"""
            <div style="border-radius: 12px; background: linear-gradient(135deg, #1e3c72, #2a5298); 
                        padding: 15px; box-shadow: 3px 3px 12px rgba(0,0,0,0.2);">
                <h3 style="color: #FFD700; text-align:center;">💡 Relaxation Tip</h3>
                <p style="font-size: 18px; color: white; font-weight: bold; text-align:center;">{mood_tips[mood]}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- MOOD TRACKER ---
    st.subheader("📊 Your Mood Tracker")

    if not mood_df.empty:
        mood_df["Date"] = pd.to_datetime(mood_df["Date"])

        # Show Data Table
        with st.expander("📜 View Mood History"):
            st.dataframe(mood_df.sort_values("Date", ascending=False))

        # Button to Clear History
        if st.button("🗑️ Clear Mood History", key="clear_history"):
            os.remove(MOOD_CSV)
            mood_df = pd.DataFrame(columns=["Date", "Mood"])
            st.success("✅ Mood history cleared successfully!")
            st.rerun()

        # Plot Mood Trends (Bar Chart)
        mood_counts = mood_df["Mood"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_facecolor("white")

        sns.barplot(x=mood_counts.index, y=mood_counts.values, ax=ax,
                    hue=mood_counts.index, palette=["#FFD700", "#4682B4", "#FF4500", "#8B0000", "#32CD32"],
                    legend=False)

        ax.set_title("Mood Frequency Over Time", fontsize=14, color="black")
        ax.set_ylabel("Count", fontsize=12, color="black")
        ax.set_xlabel("Mood", fontsize=12, color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")

        st.pyplot(fig)

        # --- CALENDAR HEATMAP ---
        if not mood_df.empty:
            mood_df["Date"] = pd.to_datetime(mood_df["Date"])
            mood_df["Mood_Score"] = mood_df["Mood"].map({
                "😊 Happy": 5, "😞 Sad": 2, "😰 Stressed": 1, "😟 Anxious": 2, "😌 Relaxed": 4
            })

            mood_data = mood_df.groupby("Date")["Mood_Score"].sum()

            mood_data = pd.Series(mood_data, index=mood_data.index)

            current_year = datetime.datetime.today().year
            mood_data = mood_data[mood_data.index.year == current_year]

            date_range = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31")
            mood_data = mood_data.reindex(date_range, fill_value=0)

            if not mood_data.empty:
                df_heatmap = mood_data.reset_index()
                df_heatmap['Day'] = df_heatmap['index'].dt.day
                df_heatmap['Month'] = df_heatmap['index'].dt.month
                df_heatmap.rename(columns={0: 'Mood_Score'}, inplace=True)

                pivot_data = df_heatmap.pivot_table(index='Month', columns='Day', values='Mood_Score', fill_value=0)

                fig, ax = plt.subplots(figsize=(12, 4))
                sns.heatmap(pivot_data, cmap="YlGnBu", linewidths=0.5, linecolor='gray')
                plt.title('Mood Calendar Heatmap')
                st.pyplot(fig)
            else:
                st.warning("⚠️ No mood data available for this year.")
    else:
        st.info("🚀 No mood logs recorded yet. Start tracking your mood today!")

elif selected == "🥗 Nutrition Guidance":
    st.markdown("""
        <style>
            /* Global background with gradient */
            .stApp {
                background: linear-gradient(to bottom, #cce5ff, #99ccff, #66b2ff); /* Soft blue gradient */
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }

            /* Curved header */
            .curved-section {
                background: linear-gradient(90deg, #0073e6, #3399ff);
                border-radius: 50% 50% 0 0;
                padding: 30px;
                text-align: center;
                color: white;
                font-size: 26px;
                font-weight: bold;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            }

            /* Glassmorphism Effect for Cards */
            .glass-card {
                background: rgba(255, 255, 255, 0.2); /* Transparent white */
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 2px 2px 15px rgba(0,0,0,0.2);
                margin: 10px 0;
                text-align: center;
                color: white;
            }

            .glass-card h4 {
                color: #ffcc00;
                font-size: 20px;
                margin-bottom: 10px;
            }

            /* Nutrition section with better contrast */
            .nutrition-section {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 15px;
                text-align: center;
                margin-top: 15px;
                font-size: 16px;
                color: white;
            }

            /* Styled Buttons */
            .glow-btn {
                display: block;
                width: 100%;
                padding: 12px;
                font-size: 18px;
                color: white;
                background: linear-gradient(90deg, #ff7e5f, #feb47b);
                border: none;
                border-radius: 25px;
                cursor: pointer;
                text-align: center;
                box-shadow: 0 0 10px rgba(255, 126, 95, 0.5);
                transition: 0.3s;
            }
            .glow-btn:hover {
                box-shadow: 0 0 20px rgba(255, 126, 95, 0.8);
            }

        </style>
    """, unsafe_allow_html=True)

    # Curved Section for Intro
    st.markdown('<div class="curved-section">🥗 Personalized Indian Nutrition Guidance</div>', unsafe_allow_html=True)

    # Nutrition Guidance Info
    st.markdown(
        "<p style='text-align: center; font-size: 18px;'>Get meal suggestions tailored to your health goals.</p>",
        unsafe_allow_html=True)

    
    col1, col2 = st.columns(2)
    with col1:
        dietary_preference = st.selectbox("Select Your Dietary Preference:",
                                          ["Vegetarian", "Vegan", "Keto", "High-Protein", "Low-Carb", "Balanced Diet"])
    with col2:
        health_goal = st.selectbox("Select Your Health Goal:",
                                   ["Weight Loss", "Muscle Gain", "Improve Digestion", "Boost Energy",
                                    "General Well-being"])

    # Meal Plan Data
    meal_plans = {
        "Vegetarian": {
            "Weight Loss": {
                "Breakfast": "Poha with peanuts 🥜",
                "Lunch": "Dal, roti, and sabzi 🥬",
                "Dinner": "Khichdi with curd 🍚"
            },
            "Muscle Gain": {
                "Breakfast": "Paneer paratha with curd 🧀",
                "Lunch": "Rajma chawal 🍛",
                "Dinner": "Soya chunk curry with rice 🍚"
            },
            "Improve Digestion": {
                "Breakfast": "Sprouts chaat 🌱",
                "Lunch": "Lauki dal and rice 🥘",
                "Dinner": "Curd rice with jeera tadka 🍚"
            },
            "Boost Energy": {
                "Breakfast": "Banana smoothie with almonds 🍌",
                "Lunch": "Bhindi sabzi with dal and rice 🥒",
                "Dinner": "Mixed veg curry with chapati 🥗"
            },
            "General Well-being": {
                "Breakfast": "Moong dal chilla 🌮",
                "Lunch": "Baingan bharta with jowar roti 🍆",
                "Dinner": "Palak paneer with paratha 🥘"
            }
        },
        "Vegan": {
            "Weight Loss": {
                "Breakfast": "Ragi porridge with nuts 🌰",
                "Lunch": "Chana masala with brown rice 🍛",
                "Dinner": "Mixed vegetable dalia 🥣"
            },
            "Muscle Gain": {
                "Breakfast": "Peanut butter toast 🥜",
                "Lunch": "Masoor dal and quinoa 🍚",
                "Dinner": "Soya chunk pulao 🍲"
            },
            "Improve Digestion": {
                "Breakfast": "Papaya with flaxseeds 🍈",
                "Lunch": "Ridge gourd sabzi with rice 🥗",
                "Dinner": "Methi thepla with curd 🌿"
            },
            "Boost Energy": {
                "Breakfast": "Coconut water with chia seeds 🥥",
                "Lunch": "Rajgira roti with sabzi 🥬",
                "Dinner": "Sweet potato tikki with salad 🥔"
            },
            "General Well-being": {
                "Breakfast": "Bajra roti with jaggery 🌾",
                "Lunch": "Dal baati 🥘",
                "Dinner": "Tofu bhurji with paratha 🍽"
            }
        },
        "Keto": {
            "Weight Loss": {
                "Breakfast": "Paneer bhurji with butter 🧈",
                "Lunch": "Egg curry with spinach 🍳",
                "Dinner": "Grilled fish with ghee sautéed veggies 🐟"
            },
            "Muscle Gain": {
                "Breakfast": "Cheese omelet with avocado 🥑",
                "Lunch": "Butter chicken with salad 🍗",
                "Dinner": "Fish tikka with mint chutney 🐠"
            },
            "Improve Digestion": {
                "Breakfast": "Coconut flour dosa 🥥",
                "Lunch": "Bhindi stir-fry with paneer 🍛",
                "Dinner": "Mushroom masala with raita 🍄"
            },
            "Boost Energy":{
                "Breakfast": "Bulletproof coffee ☕",
                "Lunch": "Palak chicken with ghee rice 🥘",
                "Dinner": "Lamb kebabs with cucumber salad 🍢"
            },
            "General Well-being": {
                "Breakfast": "Almond flour pancakes 🥞",
                "Lunch": "Cauliflower rice biryani 🍛",
                "Dinner": "Tandoori fish with sautéed greens 🐟"
            }
        },
        "High-Protein": {
            "Weight Loss": {
                "Breakfast": "Besan chilla with mint chutney 🥞",
                "Lunch": "Sprouted moong dal salad 🥗",
                "Dinner": "Grilled tofu and veggies 🍢"
            },
            "Muscle Gain": {
                "Breakfast": "Boiled eggs with almonds 🥚",
                "Lunch": "Mutton curry with rice 🍖",
                "Dinner": "Paneer tikka with whole wheat roti 🧀"
            },
            "Improve Digestion": {
                "Breakfast": "Buttermilk with jeera powder 🥤",
                "Lunch": "Oats khichdi with dal 🥘",
                "Dinner": "Spinach soup with grilled paneer 🍲"
            },
            "Boost Energy": {
                "Breakfast": "Chickpea pancakes with chutney 🌮",
                "Lunch": "Fish curry with brown rice 🐠",
                "Dinner": "Dal makhani with roti 🥘"
            },
            "General Well-being": {
                "Breakfast": "Masala oats with nuts 🌰",
                "Lunch": "Egg curry with chapati 🍛",
                "Dinner": "Soya chunks and vegetable stir-fry 🍲"
            }
        },
        "Low-Carb": {
            "Weight Loss": {
                "Breakfast": "Boiled eggs with green tea 🍵",
                "Lunch": "Paneer salad with olive oil 🧀",
                "Dinner": "Grilled fish with steamed veggies 🐟"
            },
            "Muscle Gain": {
                "Breakfast": "Scrambled eggs with cheese 🧀",
                "Lunch": "Chicken tikka with green chutney 🍗",
                "Dinner": "Grilled prawns with spinach 🦐"
            },
            "Improve Digestion": {
                "Breakfast": "Herbal tea with nuts 🍵",
                "Lunch": "Pumpkin soup with grilled paneer 🍲",
                "Dinner": "Steamed veggies with lemon dressing 🥗"
            },
            "Boost Energy": {
                "Breakfast": "Avocado smoothie with seeds 🥑",
                "Lunch": "Eggplant bharta with curd 🍆",
                "Dinner": "Mutton kebabs with cucumber salad 🍖"
            },
            "General Well-being": {
                "Breakfast": "Greek yogurt with berries 🍓",
                "Lunch": "Grilled chicken with mushrooms 🍄",
                "Dinner": "Fish curry with sautéed spinach 🥘"
            }
        },
        "Balanced Diet": {
            "Weight Loss": {
                "Breakfast": "Upma with vegetables 🍛",
                "Lunch": "Dal tadka with roti 🥘",
                "Dinner": "Vegetable pulao with raita 🍚"
            },
            "Muscle Gain": {
                "Breakfast": "Egg paratha with curd 🥚",
                "Lunch": "Chicken biryani with cucumber raita 🍛",
                "Dinner": "Rajma with brown rice 🍲"
            },
            "Improve Digestion": {
                "Breakfast": "Banana with soaked almonds 🍌",
                "Lunch": "Dhokla with mint chutney 🥮",
                "Dinner": "Curd rice with coriander tadka 🍚"
            },
            "Boost Energy": {
                "Breakfast": "Dry fruit laddoo with milk 🥛",
                "Lunch": "Masala dal with ghee chapati 🥘",
                "Dinner": "Fish fry with sautéed veggies 🐠"
            },
            "General Well-being": {
                "Breakfast": "Idli with sambar 🥥",
                "Lunch": "Mixed dal khichdi 🥘",
                "Dinner": "Methi roti with aloo sabzi 🥔"
            }
        }
    }

    # Button to Generate Meal Plan
    if st.button("✨ Get My Meal Plan", key="meal_plan"):
        if dietary_preference in meal_plans and health_goal in meal_plans[dietary_preference]:
            st.success(f"🍽 Your {dietary_preference} Meal Plan for {health_goal}")
            meals = meal_plans[dietary_preference][health_goal]

            # Display meals in styled cards
            if meals and isinstance(meals, dict):
                for meal_time, meal_name in meals.items():
                    with st.container():
                        st.markdown(f"""
                                        <div class="glass-card">
                                            <h4>{meal_time}</h4>
                                            <p>{meal_name}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                        st.write("")
        else:
            st.warning("Sorry, no meal plan found. Try another combination.")

    st.markdown("""
                                <div class="nutrition-section">
                                    <h4>📊 Nutrition Facts</h4>
                                    Your meal plan is balanced with macros and micronutrients, ensuring good health.
                                </div>
                            """, unsafe_allow_html=True)


    if dietary_preference in meal_plans and health_goal in meal_plans[dietary_preference]:
        meal_text = f"Meal Plan for {dietary_preference} - {health_goal}\n\n"
        for meal_time, meal_name in meal_plans[dietary_preference][health_goal].items():
            meal_text += f"{meal_time}: {meal_name}\n"

        st.download_button(label="📥 Download Your Meal Plan", data=meal_text, file_name="meal_plan.txt",
                           mime="text/plain", help="Click to download your personalized meal plan.")


# AI Assistant in Streamlit
elif selected == "🤖 AI Assistant":
    st.title("🤖 AI Health Chatbot")
    st.write("💡 Ask any health-related question and get AI-powered insights instantly!")

    # 🔹 Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! 😊 I'm your AI Health Assistant. How can I help you today?"}
        ]

    # 🔹 Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.write(message["content"])

    # 🔹 User Input with Chat UI
    if user_query := st.chat_input("Ask me anything about health..."):
        # Append user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Display user's message immediately
        with st.chat_message("user"):
            st.write(user_query)

        # Simulate AI Typing Effect
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.write("🤖 AI is thinking...")

            # ⏳ Simulate a delay for realism
            time.sleep(2)

            try:
                # 🔑 Configure AI Model
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(user_query)

                # Update AI response
                typing_placeholder.write(response.text)

                # Save AI response in history
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            except Exception as e:
                typing_placeholder.write("⚠️ API Error: Unable to generate a response.")
                st.error(f"Error: {e}")

st.sidebar.write("Developed with ❤️ for better health!")
