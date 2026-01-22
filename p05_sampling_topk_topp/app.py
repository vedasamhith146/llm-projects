import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from train_gpt2 import GPT2,GPT2Config,device
from sample import generate
import torch

config=GPT2Config(vocab_size=50304)
model=GPT2(config)
checkpoint_path = "gpt2_step_11500.pt" 
state_dict = torch.load(checkpoint_path, map_location=device)
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.to(device)
model.eval() 

st.title("LLM Sample Dashboard")

st.sidebar.header("Parameters")
temp=st.sidebar.slider("Temparature",0.0,2.0,1.0,step=0.1)
top_k=st.sidebar.slider("Top-K",0,100,40)
top_p=st.sidebar.slider("Top-P",0.0,1.0,0.9)
max_tok=st.sidebar.slider("Max Tokens",10,200,50)

prompt=st.text_area("Enter prompt:",value="The meaning of life is")

if st.button("Generate Output"):
    output_text,entropies=generate(model,prompt,max_tok,top_k,top_p,temp)

    st.subheader("Generated Text:")
    st.write(output_text)

    st.subheader("Model Uncertanity (Entropy) per token")
    fig,ax=plt.subplots()
    ax.plot(entropies)
    ax.set_ylabel("Entropy(Higher=more chaos)")
    ax.set_xlabel("Token Step")
    st.pyplot(fig)

st.divider()
st.header("Expirement:The sweep")

if st.button("Run parameter sweep"):
    st.write("Running sweep......this might take a minute")

    temps=[0.1,0.5,0.8,1.0,1.2,1.5,2.0]
    avg_entropies=[]
    diversity_scores=[]

    progress_bar=st.progress(0)

    for i,t in enumerate(temps):
        samples=[]
        batch_entropy=[]

        for _ in range(5):
            text, ent_hist=generate(model,prompt,20,top_k,top_p,t)
            samples.append(text)
            batch_entropy.append(np.mean(ent_hist))

        avg_entropies.append(np.mean(batch_entropy))

        unique_texts=len(set(samples))
        diversity_scores.append(unique_texts/5.0)

        progress_bar.progress((i+1)/len(temps))

    fig2,ax2=plt.subplots()
    ax2.plot(temps,avg_entropies,label='Entropy (chaos)',marker='o')
    ax2.plot(temps,diversity_scores,label='Diversity(uniqueness)',marker='x',linestyle='--')
    ax2.set_xlabel("temparature")
    ax2.legend()
    st.pyplot(fig2)


