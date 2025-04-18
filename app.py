import streamlit as st
import tempfile
from predict import predict_lipreading
from moviepy import VideoFileClip



# Page setup
st.set_page_config(page_title="Chaplin", page_icon="🧠", layout="centered")

# Show logo
st.image("chaplinLogo.png", width=250)

st.title("Chaplin: A Lipreading AI")
st.write("Upload a short video of someone speaking. The model will predict what was said!")

with st.expander("What kind of sentences does the model predict?"):
    st.write("""
The model predicts sentences in the following format:

**command** + **color** + **preposition** + **letter** + **digit** + **adverb**

**Word categories:**
- **Command (4):** `bin`, `lay`, `place`, `set`  
- **Color (4):** `blue`, `green`, `red`, `white`  
- **Preposition (4):** `at`, `by`, `in`, `with`  
- **Letter (25):** `A–Z` (excluding `W`)  
- **Digit (10):** `zero` to `nine`  
- **Adverb (4):** `again`, `now`, `please`, `soon`
    """)



# Upload video
uploaded_file = st.file_uploader("📤 Upload a video file (.mpg or .mp4)", type=["mpg", "mp4"])

if uploaded_file:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mpg") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    # Convert to mp4 for preview
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
            clip = VideoFileClip(input_path)
            clip.write_videofile(tmp_output.name, codec="libx264", audio=False)
            preview_path = tmp_output.name
    except Exception as e:
        st.error(f"Video conversion failed: {e}")
        preview_path = None

    # Create two columns side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Video Preview")
        if preview_path:
            st.video(preview_path)

    with col2:
        st.subheader("Prediction")
        if st.button("Run Prediction"):
            with st.spinner("Analyzing the video..."):
                try:
                    prediction = predict_lipreading(input_path)
                    st.markdown(f"""
                        <div style='
                            font-size: 1.8em;
                            padding: 1em;
                            border-radius: 10px;
                            background-color: #1e1e1e;
                            color: #ffffff;
                            border: 1px solid #555;
                            text-align: center;
                        '>
                            {prediction}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        