import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("attention_model.pkl", "rb"))

st.title("Attention Span Estimator")

st.write("Enter user interaction data to predict attention span:")

col1, col2 = st.columns(2)


with col1:
    session_duration = st.number_input("Session Duration (seconds)", min_value=0, value=1000)

    scroll_depth = st.slider("Scroll Depth (%)", 0.0, 100.0, 50.0)

    clicks_per_minute = st.number_input("Clicks Per Minute", min_value=0.0, value=5.0)

with col2:
    idle_time = st.number_input("Idle Time (seconds)", min_value=0.0, value=2.0)

    mouse_distance = st.number_input("Mouse Movement Distance", min_value=0.0, value=100.0)

    tab_switches = st.number_input("Tab Switches", min_value=0, value=1)



if st.button("Predict Attention Level"):

    input_data = np.array([[session_duration,
                            scroll_depth,
                            clicks_per_minute,
                            idle_time,
                            mouse_distance,
                            tab_switches]])

    prediction = model.predict(input_data)

    result = prediction[0].upper()

    if result == "HIGH":
        st.success(f"The predicted attention level is: *{result}*")
    else:
        st.warning(f"The predicted attention level is: *{result}*")

