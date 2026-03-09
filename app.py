import streamlit as st

st.title("Test App - Basic Deployment Check")
st.write("If this loads, deployment works! No deps beyond Streamlit.")

if st.button("Test Button"):
    st.success("Button clicked - app is live!")
