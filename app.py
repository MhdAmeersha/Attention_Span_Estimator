import streamlit as st
import pickle
import numpy as np

# Load trained model

with open("attention_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Attention Span Estimator")
st.write("Enter metrics to predict if the user's attention level is *HIGH* or *LOW*.")


col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Session Duration (sec)", min_value=0, value=1000)
    scroll_depth = st.slider("Scroll Depth (%)", 0, 100, 50)
    clicks = st.number_input("Clicks per Minute", min_value=0, value=15)

with col2:
    idle_time = st.number_input("Idle Time (sec)", min_value=0, value=20)
    mouse_dist = st.number_input("Mouse Movement Distance", min_value=0, value=400)
    tab_switches = st.number_input("Tab Switches", min_value=0, value=2)

#prediction

if st.button("Predict Attention Level"):
    input_data = np.array([[duration, scroll_depth, clicks, idle_time, mouse_dist, tab_switches]])

    prediction = model.predict(input_data)
    

    result = prediction[0].upper()
    if result == "HIGH":
        st.success(f"The predicted attention level is: *{result}*")
    else:
        st.warning(f"The predicted attention level is: *{result}*")

