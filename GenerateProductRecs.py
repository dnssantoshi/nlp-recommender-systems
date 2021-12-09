#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Santoshi Diddi
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Define Constants
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.45rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

def load_data():
    # load the data from pickle files
    # These files are created as part of the model building process in BuildProductRecs.py
    df = pd.read_pickle('df.pkl')
    indices = pd.read_pickle('indices.pkl')
    cosine_sim = pd.read_pickle('cosine_sim.pkl')

    return df, cosine_sim, indices

df, cosine_sim, indices = load_data()
print("Data Load Complete!")

class GenerateProductRecs:

    def __init__(self):
        # Define embedder
        self.count = CountVectorizer(stop_words='english')

    def construct_sidebar(self):
        # Construct the input sidebar for user to choose the input
        st.sidebar.markdown(
            '<p class="font-style"><b>Beauty & Personal Care</b></p><p class="font-style"><b>Product Search</b></p>',
            unsafe_allow_html=True
        )

        num_recs = st.sidebar.selectbox(
            f"Please select the number of recommendations",
            sorted(range(1, 11)), index=2)

        title = st.sidebar.selectbox(
            f"Please select your product",
            df.productTitle)

        if not title:
            st.sidebar.warning("Please select all required fields")

        return num_recs + 1, title

    def construct_menu_tabs(self):

        st.markdown(
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
            unsafe_allow_html=True,
        )
        st.markdown("""<style> .sticky { background:white; position: fixed; top: 60px; width: 100%;} </style>""", unsafe_allow_html=True)

        query_params = st.experimental_get_query_params()
        tabs = ["Search", "Reports", "Overview"]
        if "tab" in query_params:
            active_tab = query_params["tab"][0]
        else:
            active_tab = "Search"

        if active_tab not in tabs:
            st.experimental_set_query_params(tab="Search")
            active_tab = "Search"

        li_items = "".join(
            f"""
            <li class="nav-item">
                <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
            </li>
            """
            for t in tabs
        )
        tabs_html = f"""
            <ul class="nav nav-tabs sticky">
            {li_items}
            </ul>
        """

        st.markdown(tabs_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if active_tab == "Search":
            self.get_recommendations()
        elif active_tab == "Reports":
            self.construct_reports()
        elif active_tab == "Overview":
            self.technical_overview()
        else:
            st.error("Something has gone terribly wrong.")

    def construct_reports(self):
        # Construct the radio menu items and embed all the reports
        st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.sidebar.markdown(
            '<p class="font-style"><b>Please choose from the following list of reports</b></p>',
            unsafe_allow_html=True
        )

        selection = st.sidebar.radio("", ("Products Metadata", "User Reviews", "Reviews Dashboard"))

        if selection == 'Products Metadata':
            components.iframe(
                "https://datastudio.google.com/embed/reporting/5ed885e1-e55c-44d6-b0e6-6c0a9d29196b/page/3oIhC",
                height=1000)

        if selection == 'User Reviews':
            components.iframe(
                "https://datastudio.google.com/embed/reporting/30d14b66-5fa5-4eb7-8bb6-260b133762a1/page/WQHhC",
                height=1200)

        if selection == 'Reviews Dashboard':
            components.iframe(
                "https://datastudio.google.com/embed/reporting/66773423-a76a-4657-bdd9-9f739d032c50/page/j1BhC",
                height=1000)

    def pretty_print(self, title, asin, description, image):
        st.write(HTML_WRAPPER.format(
            '<table><tr><td style="border-color:white;"><span style="font-weight:400;font-style:normal">'
            + "<b>Product Name:  </b>" + title
            + "<br/><br/><b>ASIN:  </b>" + asin
            + "<br/><b>Description:  </b>" + description
            + ' </span></td><td style="border-color:white;"><img src="' + image
            + '" width="150" height="300" border-radius="10"></td></tr></table>'), unsafe_allow_html=True)

    # Function that takes in product title as input and outputs most similar products
    def get_recommendations(self):
        num_recs, title = self.construct_sidebar()
        # Get the index of the product that matches the given product title
        idx = indices[title]

        # Get the pairwise similarity scores of all products with given product
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the products based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar products
        sim_scores = sim_scores[0:num_recs]

        # Get the product indices
        product_indices = [i[0] for i in sim_scores]

        # Get Product Titles
        product_titles = df['productTitle'].iloc[product_indices]

        # Get Product Images
        product_images = df['imageURLHighRes'].iloc[product_indices]

        # Get Product Description
        product_desc = df['description'].iloc[product_indices].transform(
            lambda x: str(x).split('<img', 1)[0].strip() if '<img' in x else x)

        # Get Product ASIN
        product_asin = df['asin'].iloc[product_indices]

        st.markdown('<p class="font-style"><b>You searched for: </b></p>', unsafe_allow_html=True)

        self.pretty_print(product_titles.values[0], product_asin.values[0], product_desc.values[0],
                          product_images.values[0][0])

        st.markdown(
            '<p class="font-style"><b>Here\'s the Top ' + str(num_recs - 1) + ' Product Recommendations</b></p>',
            unsafe_allow_html=True
        )
        for i in range(1, num_recs):
            col_size = len(product_images.values[i])
            with st.beta_expander("Please click here to view additional images for " + product_titles.values[i]):
                cols = st.beta_columns(col_size)
                for j in range(0, col_size):
                    cols[j].image(product_images.values[i][j], width=75)
                st.write("</br>", unsafe_allow_html=True)
            self.pretty_print(product_titles.values[i], product_asin.values[i], product_desc.values[i],
                              product_images.values[i][0])

        return df['productTitle'].iloc[product_indices]

    def technical_overview(self):
        st.sidebar.write("Technical overview")
        technical = st.sidebar.radio("", (
            "Abstract", "Data Wrangling", "Text Embeddings", "Cosine Similarities", "Recommendations", "Reports"))

        if technical == 'Abstract':
            st.subheader("Abstract")
            st.write(
                "This application recommends products that are similar to a particular product. To acheive this, computed pairwise cosine similarity scores for all products based on the descriptions and brand attributes. The recommendation is products is based on this similarity score threshold.")

        if technical == 'Data Wrangling':
            st.subheader("Data Wrangling")
            st.write(
                "The dataset has been wrangled before using this for creating word vectors. This include imputing any missing values, creating any aggregated attributes, removing stopwords and punctuations and retrieving the important keywords for creating word vectors.")

        if technical == "Text Embeddings":
            st.subheader("Text Embeddings")
            st.write(
                'This is a natural language processing problem and we cannot use the raw text to compute similarities. So inorder to extract the same kind of featured products the word vectors of each description. Word vectors are the vectorized description of words in a document. The vectors carry the semantic meaning with it.')

        if technical == 'Cosine Similarities':
            st.subheader("Cosine Similarities")
            st.write(
                "Used cosine similarity to calculate the numeric quantity that denotes the similarity between two products. Used cosine similarity score is independent of the magnitude and it is easy and fast to calculate when compared to TF-IDF. This would return an matric of n*n where n is the total number of products. So each product will have a 1 x n mappings.")

        if technical == "Recommendations":
            st.subheader("Product Recommendations")
            st.write('1. Get the index of the product based on the product title from the user')
            st.write('2. Retrieve the cosine similarity scores for that particular product with all products')
            st.write('3. Enumerate with the position and the similarity score')
            st.write(
                '4. Get the top n recommendations where n is the user input for how many recommendations the user wants to retrieve')
            st.write('5. Return the products corresponding to the indexes of the top recommendations')

        if technical == "Reports":
            st.subheader("Product Metadata")
            st.write(
                "This reflects all the product metadata along with additional attributes such as price, brand, shipping information. This also provides the capability to search for a particular product based on ASIN, Brand and Item number. It also allows the user to export the data displayed on this table.")
            st.subheader("User Reviews Dataset")
            st.write(
                "This is a data dump of all the user reviews in the amazon luxury beauty product category. This provides additional information such as the product review image if any uploaded, review description, summary of the review etc.,This enables user to filter the data based on the ASIN, Reviewer Id and user rating. It also allows the user to export the data displayed on this table.")
            st.subheader("User Reviews Dashboard")
            st.write(
                "This dashboard provides all the trends and the statistics of all the user reviews provided to luxury beauty category products. Some of the important questions and the kind of trends in getting user reviews are depicted over time. It enables user to filter on the data time frame they would like to see the trends for. Apart from this, user can filter with ASIN and Rating attributes.")

    def construct_app(self):
        # This is the main point for application start up which triggers the rest of the function calls
        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            .table { border-collapse:collapse }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
        <style>
        .table-style { border-collapse:collapse }
        </style>
        """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        self.construct_menu_tabs()
        # st.write(
        #     '<style>body { margin: 0; font-family: Arial, Helvetica, sans-serif;} .header{margin: 0; background:white;padding: 0px 720px 0px 520px; '
        #     'position:fixed;top:0;}</style><div class="header" id="myHeader">'
        #     '<img src="https://www.freepnglogos.com/uploads/amazon-png-logo-vector/woodland-gardening-amazon-png-logo-vector-8.png" alt="Not Available" style="width:300px;height:80px;" />'
        #     '</div>',
        #     unsafe_allow_html=True)

        return self


gpr = GenerateProductRecs()
gpr.construct_app()
