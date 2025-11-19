import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
    response_length = st.slider("Max Answer Length", 20, 200, 80, 10)
    attitude_level = st.slider("Attitude Level", 0.1, 1.0, 0.8, 0.1)
    st.markdown("---")
    st.info("Hybrid: factual QA + small creative model for fun answers.")

# Main layout
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("💬 Ask Queen Kabiya")
    example_prompts = [
        "What is the capital of Nepal?",
        "What is 1+1?",
        "Quick confidence boost?",
        "Best comeback for haters?"
    ]
    selected = st.selectbox("Or pick a baddie topic:", [""] + example_prompts)
    if selected:
        user_input = st.text_area("Your question:", value=selected, height=80)
    else:
        user_input = st.text_area(
            "Your question:", 
            placeholder="Ask anything factual or sassy…",
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
                # --- Factual QA pipeline ---
                qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2"
                )

                # Context can be tiny or generic
                context = (
                    "Queen Kabiya is sassy, confident, and always gives short, iconic advice. "
                    "She slays haters and boosts confidence."
                )

                result = qa_pipeline(question=user_input, context=context)
                factual_answer = result.get("answer", "").strip()

                # Check if answer seems reasonable (non-empty / short)
                if len(factual_answer) > 0 and factual_answer.lower() not in ["unknown", "no answer"]:
                    st.subheader("👑 Queen K Says (Factual):")
                    st.success(factual_answer)
                else:
                    # --- Fall back to small creative model ---
                    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

                    prompt = f"Answer in 3-4 short, sassy lines:\n{user_input}"
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=response_length,
                        temperature=attitude_level
                    )
                    creative_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.subheader("👑 Queen K Says (Sassy):")
                    st.success(creative_answer)

            except Exception as e:
                st.error(f"❌ Error generating response:\n{e}")

st.markdown("---")
st.caption("Made for Queen Kabiya • Stay iconic ✨")
