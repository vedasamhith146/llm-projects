import streamlit as st
from sample import generate_text,next_token_generator
import plotly.graph_objects as go

st.title('Play with my GPT-2 model')

with st.sidebar:
    temp=st.slider('Temparature',min_value=0.0,max_value=2.0,value=0.7,step=0.1)
    top_k=st.slider('Top-k',min_value=10,max_value=50,value=30,step=10)
    top_p=st.slider('Top-p',min_value=0.7,max_value=0.95,value=0.90,step=0.05)
    max_new_tokens=st.slider('max_output_tokens',min_value=20,max_value=100,value=30,step=10)

col1,col2=st.columns([5,1])
with col1:
    prompt=st.text_input("prompt",label_visibility="collapsed")
with col2:
    generate=st.button("Generate")

temp_params=[0.1,0.3,0.5,0.7,0.9,1.2,1.5]

if generate:
    output,entr,div=generate_text(prompt,max_new_tokens=max_new_tokens,temp=temp,top_k=top_k,top_p=top_p)
    st.write(output)
    entropies=[]
    diversities=[]
    for t in temp_params:
        _,entropy,diversity=generate_text(prompt,max_new_tokens=max_new_tokens,temp=t,top_k=top_k,top_p=top_p)
        entropies.append(entropy)
        diversities.append(diversity)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=entropies,y=diversities,mode='lines+markers'))
    fig.update_layout(title="Entropy vs diversity",xaxis_title="Entropy",yaxis_title="diversity")

    st.plotly_chart(fig)




