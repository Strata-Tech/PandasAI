from dotenv import load_dotenv
import os
import streamlit as st

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
#from pandasai.llm.google_palm import GooglePalm
from pandasai.llm.starcoder import Starcoder
import matplotlib
import plotly.express as px


#matplotlib.use('TkAgg')



load_dotenv()


API_KEY=os.environ['OPENAI_API_KEY']

llm=OpenAI(api_token=API_KEY)


st.title("Data Analysis with PandasAI using conversational text")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df.head(5))

  prompt=st.text_area("Enter your question:")
  if st.button("Generate"):
    if prompt:
      with st.spinner("Response generating.. please wait"):
        sdf = SmartDataframe(df, config={"llm": llm})
        st.write(sdf.chat(prompt))


    else:
      st.warning("No question inputted,please ask a question first")
