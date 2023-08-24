import streamlit as st
from interactive import InteractiveDiffusion
import pandas as pd

st.title("Diffusion LLM demo")

if 'engine' not in st.session_state:
    st.session_state.engine = None

model_args = st.sidebar.text_input("model args", "/mnt/bn/research-hl/ckpts/flan_2022.llama13B.fix/args.json")
model_ckpt = st.sidebar.text_input("model ckpt", "/mnt/bn/research-hl/ckpts/flan_2022.llama13B.fix/checkpoint-10000")
def get_engine(model_args, model_ckpt):
    with st.spinner('Loading model'):
        # if hasattr(st.session_state, "engine"):
        #     del st.session_state.engine 
        st.session_state.engine = InteractiveDiffusion(model_args, model_ckpt)
    st.success("model loaded")
    
load_model_bttn = st.sidebar.button("load model", on_click=get_engine, args=(model_args, model_ckpt))
 
lengths = st.sidebar.text_input("lengths", "50 ")    # st.sidebar.number_input("length", value=100)
length_beam = st.sidebar.number_input("length beam", value=1)
mbr = st.sidebar.number_input("mbr", value=1)
max_iterations = st.sidebar.number_input("max_iterations", value=50)
argmax_decoding = st.sidebar.checkbox("argmax_decoding", True)
strategy = st.sidebar.selectbox(
    "strategy",
    options=[
        "reparam-uncond-deterministic-cosine", 
        "cmlm", 
        "ar",
        "reparam-cond-deterministic-cosine"
    ]
)

def show_results(prompt):
    outputs = []
    for i, output in enumerate(st.session_state.engine.sample(
        prompt, lengths, argmax_decoding=argmax_decoding, strategy=strategy,
        max_iterations=max_iterations, length_beam=length_beam, mbr=mbr
    )):
        st.text(f"STEP-{i}\t{output}")
        # outputs.append(output)
        # st.dataframe(pd.DataFrame(outputs))

prompt = st.text_area("prompt")
run = st.button("run", on_click=show_results, args=(prompt, ))
