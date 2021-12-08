#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Santoshi Diddi
"""

import streamlit as st
import os
import json
import gzip
import re
import pandas as pd
import numpy as np
from urllib.request import urlopen
import warnings
import networkx as nx
import pickle as pkl
import plotly.graph_objects as go
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


warnings.filterwarnings('ignore')

# Define stop words
stopwords = list(STOP_WORDS)


class BuildProductRecs:

    def read_data(self):
        '''Load the Amazon luxury beauty products metadata'''
        data = []
        with gzip.open('meta_Luxury_Beauty.json.gz') as f:
            for l in f:
                data.append(json.loads(l.strip()))

        # Total length of list, this number equals total number of products
        st.write("Total number of products: ", len(data))

        # First row of the list
        st.write("Sample first row of data: ", data[1])

        return data

    def clean_data(self):
        ''' Data Wrangling: Amazon Product Category - Luxury Beauty'''
        # Get data
        data = self.read_data()

        # Convert list into pandas dataframe
        df = pd.DataFrame.from_dict(data)
        st.write("Total Products in Amazon Luxury Beauty category: ", len(df))

        # Normalize details json attribute and extract all features
        details = pd.json_normalize(df.details)

        # Rename columns more appropriately
        details.columns = ['productDimensions', 'shippingWeight', 'domesticShipping', 'internationalShipping', 'asin',
                           'itemModelNumber', 'discontinuedByManufacturer', 'batteries', 'itemWeight',
                           'shippingAdvisory', 'asinDuplicate', 'upc']

        # Remove duplicate and empty columns
        details.drop(columns=['asinDuplicate', 'itemWeight', 'shippingAdvisory', 'upc'], inplace=True)

        # Format Shipping weight column values
        details['shippingWeight'] = details['shippingWeight'].transform(lambda x: str(x).replace('(', '').strip())

        # Merge details to the original data
        df = pd.merge(df, details, on="asin")

        # Imputing the title to reflect the product clearly
        df['title'] = df['title'].replace(['Klorane'], 'Klorane Dry Shampoo')

        # Generate Product Title
        df['productTitle'] = df['title'].transform(lambda x: x.split(' ', 1)[1])
        df['productTitle'] = df['productTitle'].transform(lambda x: x.replace('&amp;', '').strip())

        # Retrieve rank
        df['ranking'] = df['rank'].transform(lambda x: str(x).split(' ', 1)[0].strip() if len(x) != 0 else 0)


        # Drop duplicate products
        df = df.drop_duplicates(subset=['productTitle'])

        # Retrieve brand
        df.brand = df.title.transform(lambda x: x.split(' ')[0])

        # Retrieve keywords
        df['description'] = df.description.transform(lambda x: ' '.join(x))

        # Retaining products that have descriptions
        df = df[df.description.transform(lambda x: len(x) != 0)]

        # Parse Keywords
        df['keywords'] = df['description'].apply(lambda x: re.sub('[^a-zA-z\s]', '', str.lower(x)))
        df['keywords'] = df['keywords'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

        # Format price
        df['price'] = df['price'].replace({'\$': ''}, regex=True)

        # Include Primary and Secondary categories
        df['primaryCategory'] = 'Beauty & Personal Care'
        df['secondaryCategory'] = 'Luxury Beauty'

        # Impute empty image urls
        df['imageURLHighRes'] = df['imageURLHighRes'].transform(lambda x: ['https://via.placeholder.com/250?text=Image+Not+Available'] if len(x) == 0 else x)

        # Removing unused columns
        df.drop(columns=['category', 'tech1', 'fit', 'title', 'tech2', 'feature', 'main_cat', 'similar_item', 'date',
                         'imageURL', 'details'], inplace=True)

        # Reset index
        df = df.reset_index(drop=True)

        # Re-arrange columns
        df = df[['asin', 'productTitle', 'primaryCategory', 'secondaryCategory', 'description', 'keywords', 'brand',
                 'price', 'itemModelNumber', 'imageURLHighRes', 'ranking', 'productDimensions', 'shippingWeight',
                 'domesticShipping', 'internationalShipping', 'discontinuedByManufacturer', 'batteries']]
        print("Clean complete!")

        # # Write to csv file (used later for reporting)
        # Useful tip: cat wrangled_data.json | jq -c '.[]' when loading data to BigQuery
        df.reset_index().to_json('wrangled_data.json',orient='records')   

        df[['asin', 'productTitle', 'primaryCategory', 'secondaryCategory', 'brand',
                 'price', 'itemModelNumber', 'imageURLHighRes', 'ranking', 'productDimensions', 'shippingWeight',
                 'domesticShipping', 'internationalShipping', 'discontinuedByManufacturer', 'batteries']].reset_index(drop=True).to_csv('wrangled_data.csv')

        return df

    def generate_embeddings(self):
        df = self.clean_data()

        # Generate embeddings as a count matrix
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['keywords'])

        # Compute the Cosine Similarity matrix based on the count_matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        # Print cleaned dataframe
        st.write(df)

        # Print the shape of count
        st.write("Count Matrix: ", count_matrix.shape)

        # Print shape of cosine similarity
        st.write("Cosine Similarity matrix: ", cosine_sim.shape)

        # Generate product title indices
        indices = pd.Series(df.index, index=df['productTitle'])
        st.write("Sample Product title indices", indices[:10])

        # Dump to pickle file to use later for prediction
        with open("df.pkl", "wb") as file1:
            pkl.dump(df, file1)

        with open("cosine_sim.pkl", "wb") as file1:
            pkl.dump(cosine_sim, file1)

        with open("indices.pkl", "wb") as file1:
            pkl.dump(indices, file1)

        return cosine_sim


    def view(self):
        self.generate_embeddings()
        print("Build Complete!")
        return self

bpr = BuildProductRecs()
bpr.view()