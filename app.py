from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
import streamlit as st
import os

# Load token
token_loaded = False
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    token_loaded = True
else:
    from dotenv import load_dotenv
    load_dotenv()
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        token_loaded = True

st.set_page_config(page_title="Kabiya's Baddie AI", page_icon="💅")

def main():
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

    # Sidebar for settings
    with st.sidebar:
        st.header("🎛️ Baddie Settings")
        attitude_level = st.slider("Attitude Level", 0.1, 1.0, 0.8, 0.1)
        response_length = st.slider("Response Length", 20, 200, 80, 10)

        st.markdown("---")
        # Model selection
        model_choice = st.selectbox(
            "Choose Model (smaller = safer on Streamlit)",
            [
                "google/flan-t5-small",
                "t5-small",
                # Add more if you want to test heavier ones
            ]
        )

        st.info("Using a smaller model helps avoid crashes on limited memory servers.")

    # Main area
    col1, col2 = st.columns([2, 1])
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
            return

        if not token_loaded:
            st.error("❌ HuggingFace token not found! Put it in Streamlit Secrets.")
            return

        with st.spinner("💅 Slaying your question…"):
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=model_choice,
                    max_new_tokens=response_length,
                    temperature=attitude_level,
                    top_p=0.9
                )
                model = ChatHuggingFace(llm=llm)

                prompt = f"""
                You are Kabiya's sassy AI. Give SHORT (3‑4 lines), ICONIC, confident but fun advice.

                QUESTION: {user_input}

                Respond with:
                - Sassy energy  
                - Fire emojis  
                - Very short advice
                """

                messages = [HumanMessage(content=prompt)]
                resp = model.invoke(messages)

                st.subheader("👑 Queen K Says:")
                st.success(resp.content)

            except Exception as e:
                st.error(f"❌ Error from model call:\n{e}")

    st.markdown("---")
    st.caption("Made for Queen Kabiya • Stay iconic ✨")

if __name__ == "__main__":
    main()
