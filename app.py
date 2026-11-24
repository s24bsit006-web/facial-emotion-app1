import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")
st.title("😊 Facial Emotion Recognition App")
st.write("Koi bhi chehre ki tasveer upload karein, model uska emotion batayega.")

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="s24bsit/facial-emotion-vit")

classifier = load_model()

uploaded_file = st.file_uploader("Tasveer chunein (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Upload ki gayi tasveer", use_container_width=True)
    
    with st.spinner("Emotion check ho raha hai..."):
        results = classifier(image)
        
    st.subheader("Result:")
    top_emotion = results[0]['label'].upper()
    top_score = results[0]['score'] * 100
    st.success(f"**Sab se zyada ghalib emotion:** {top_emotion} ({top_score:.1f}%)")
    
    st.write("### Tamam Emotions ka Score:")
    for res in results:
        st.write(f"- **{res['label'].capitalize()}**: {res['score']*100:.1f}%")
