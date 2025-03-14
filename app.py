import streamlit as st
import numpy as np 
import requests

st.set_page_config(
    page_title="Hotel Search and Recommendation via Question Answering",
    layout="centered"
)
st.markdown(
    "<h1 style='font-size: 36px; color: black; text-align: center;'>Hotel Search and Recommendation via Question Answering</h1>",
    unsafe_allow_html=True
)

input_text = st.text_input("Enter Question")

col1, col2 = st.columns([5, 5])
with col1:
    cities = st.selectbox("Location", ("Huế", "Hà Nội", "Hội An", "Nha Trang", "Phú Quốc", 
                                       "Sa Pa", "TP. Hồ Chí Minh", "Vũng Tàu", "Đà Lạt", "Đà Nẵng"))
with col2:
    rate_arr = np.arange(1, 10.5, 0.5)
    rating = st.selectbox("Rating", rate_arr)

submit = st.button("Ask Question")

if submit:
    response = requests.post(
        url="http://127.0.0.1:8000/predict",
        json={"question": input_text + "ở " + cities + " với rating là " + str(rating)}
    )

    if response.status_code == 200:
        answer = response.json()['answer']
        st.write(answer)
    else: 
        st.error("Có lỗi khi gọi API")