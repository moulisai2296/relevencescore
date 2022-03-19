import streamlit as st
st.title('Search Relevance')
search = st.text_input("Search term")
title = st.text_input("Enter product title")

test_example = [[search, title]]
st.write(predict_test(test_example))