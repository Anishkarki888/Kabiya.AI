import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

st.set_page_config(page_title="Kabiya's Baddie AI", page_icon="💅")

# UI styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(45deg, #FF6B8B, #FFE569, #FF8E8E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: bold;
}
.baddie-vibes {
    text-align: center;
    color: #FF6B8B;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💋 Kabiya\'s Baddie AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="baddie-vibes">💅 Too hot to handle, too cool to care • She slays, AI obeys</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎛️ Baddie Settings")
    attitude_level = st.slider("Attitude Level", 0.1, 1.0, 0.8, 0.1)
    response_length = st.slider("Response Length", 20, 200, 80, 10)
    st.markdown("---")
    st.info("Using small, CPU-friendly model for Streamlit deployment.")

# Main layout
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("💬 Ask Queen Kabiya")
    example_prompts = [
        "How to slay like Kabiya?",
        "Best comeback for haters?",
        "Quick confidence boost?",
        "Main character energy tips"
    ]
    selected = st.selectbox("Or pick a baddie topic:", [""] + example_prompts)
    if selected:
        user_input = st.text_area("Your question:", value=selected, height=80)
    else:
        user_input = st.text_area(
            "Your question:", 
            placeholder="Ask anything about confidence, glow-up, or being iconic…",
            height=80
        )

with col2:
    st.subheader("👑 Baddie Rules")
    st.markdown("""
    - Confidence = oxygen 🔥  
    - Your vibe = your power 🌟  
    - Haters watch, winners slay 💃  
    """)

st.markdown("---")

if st.button("💋 Get Baddie Advice", type="primary"):
    if not user_input.strip():
        st.warning("👀 Queen needs a question!")
    else:
        with st.spinner("💅 Slaying your question…"):
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

                prompt = f"Answer in 3-4 short, sassy lines:\n{user_input}"
                inputs = tokenizer(prompt, return_tensors="pt")

                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=response_length,
                    temperature=attitude_level
                )

                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.subheader("👑 Queen K Says:")
                st.success(response_text)

            except Exception as e:
                st.error(f"❌ Error generating response:\n{e}")

st.markdown("---")
st.caption("Made for Queen Kabiya • Stay iconic ✨")
