import streamlit as st

# Simple rule-based functions (temporary - replace with HF models once deployment works)
def detect_stress(dream_text):
    if not dream_text.strip():
        return 'Unknown'
    # Basic keyword-based (replace with HF pipeline later)
    if any(word in dream_text.lower() for word in ['fear', 'fall', 'chase']):
        return 'High'
    elif any(word in dream_text.lower() for word in ['anger', 'argument']):
        return 'Medium'
    else:
        return 'Low'

def generate_interp_rec(dream_text, stress_level):
    if not dream_text.strip():
        return "Invalid input.", "Please provide a dream description."
    # Basic example (replace with HF generation later)
    interp = f"Zhou Gong interpretation for '{dream_text}': Symbolizes change or fortune."
    rec = f"For {stress_level} stress: Try relaxation techniques."
    return interp, rec

# Streamlit UI
st.title("Dream Analyzer Business App (Minimal Version)")
st.write("Enter your dream for basic analysis. (Full HF models coming soon after deployment fix.)")

dream_input = st.text_area("Describe your dream:", height=150)
if st.button("Analyze Dream"):
    if dream_input.strip():
        with st.spinner("Analyzing..."):
            stress_level = detect_stress(dream_input)
            interpretation, recommendation = generate_interp_rec(dream_input, stress_level)
            
            st.subheader("Estimated Stress Level")
            st.write(stress_level)
            
            st.subheader("Zhou Gong Interpretation")
            st.write(interpretation)
            
            st.subheader("Personalized Recommendation")
            st.write(recommendation)
    else:
        st.error("Please enter a dream description.")
