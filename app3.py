import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from contractions import fix
from emoji import demojize
import plotly.express as px

# Load the models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
combined_labels = joblib.load('combined_labels.pkl')  
svd = joblib.load('truncated_svd_model.pkl')
kmeans_combined = joblib.load('kmeans_combined_model.pkl')
spectral_clustering_model = joblib.load('spectral_clustering_model.pkl')
hierarchical_clustering_model = joblib.load('hierarchical_clustering_model.pkl')

# Initialize NLTK tools
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Define important words to keep
important_words = {
    'free', 'discount', 'offer', 'deal', 'sale', 'buy', 'order', 'limited time',
    'ad', 'promotion', 'sponsor', 'click', 'like', 'share', 'comment', 'subscribe',
    'follow', 'join', 'win', 'giveaway', 'contest', 'prize', 'http', 'www', 'link',
    'url', 'twitter', 'instagram', 'facebook', 'handle', 'scam', 'fraud', 'hack',
    'cheat', 'money', 'cash', 'earn', 'pay', 'download', 'software', 'app', 'apk',
    'video', 'content', 'upload', 'watch', 'channel', 'subscriber', 'viewer',
    'stream', 'amazing', 'awesome', 'great', 'best ever', 'must see', 'check out'
}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Demojize
    text = demojize(text)
    # Handle contractions
    text = fix(text)
    # Remove punctuations and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove white spaces
    text = text.strip()
    # Lowercasing
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords but keep important words
    tokens = [word for word in tokens if word not in stop_words or word in important_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Rejoin tokens into a single string
    text = ' '.join(tokens)
    return text

st.title('YouTube Comments Clustering')

# Customizable parameters for TF-IDF and SVD
max_features = st.sidebar.slider('Max Features for TF-IDF', min_value=100, max_value=10000, value=5000)
ngram_range = st.sidebar.slider('N-Gram Range', min_value=1, max_value=3, value=1)
svd_components = st.sidebar.slider('Number of SVD Components', min_value=2, max_value=100, value=2)

# Select Clustering Method
clustering_method = st.sidebar.selectbox("Choose Clustering Method", 
                                ('K-Means Ensembled', 'Spectral Clustering', 'Hierarchical Clustering'))

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file with YouTube comments", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Comments' not in df.columns:
        st.error("CSV file must contain a column named 'Comments'")
    else:
        # Preprocess text data
        df['processed_comments'] = df['Comments'].apply(preprocess_text)

        # Vectorize text data using customizable TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_range))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_comments'])
        
        # Apply SVD
        svd_arpack = TruncatedSVD(n_components=svd_components, algorithm='arpack', random_state=42)
        tfidf_svd_arpack = svd_arpack.fit_transform(tfidf_matrix)
        
        # Cluster based on user selection
        if clustering_method == 'K-Means Ensembled':
            df['cluster'] = kmeans_combined.fit_predict(tfidf_svd_arpack)
        elif clustering_method == 'Spectral Clustering':
            df['cluster'] = spectral_clustering_model.fit_predict(tfidf_svd_arpack)
        elif clustering_method == 'Hierarchical Clustering':
            df['cluster'] = hierarchical_clustering_model.fit_predict(tfidf_svd_arpack)
        
        # Define cluster labels (assuming binary clusters for spam detection)
        cluster_labels = {
            0: 'Not Spam',
            1: 'Spam'
        }
        df['label'] = df['cluster'].map(cluster_labels)
        
        # Display data
        st.write(df.head())
        
        # Display clusters
        st.subheader('Clustered Comments')
        for label in cluster_labels.values():
            st.subheader(label)
            cluster_comments = df[df['label'] == label]['Comments']
            st.write(cluster_comments.tolist())
        
        # Search functionality
        search_term = st.text_input("Search comments")
        if search_term:
            # Split the search term into individual words
            search_terms = search_term.lower().split()
            
            # Filter comments containing any of the search words
            filtered_df = df[df['Comments'].apply(lambda x: any(term in x.lower() for term in search_terms))]
            
            # Display filtered comments
            st.write(filtered_df)

        
        # Visualization
        st.subheader('Cluster Visualization')
        fig = px.scatter(
            x=tfidf_svd_arpack[:, 0], 
            y=tfidf_svd_arpack[:, 1], 
            color=df['label'],
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title='Cluster Visualization'
        )
        st.plotly_chart(fig)
        
        # Display model performance metrics
        silhouette_avg = silhouette_score(tfidf_svd_arpack, df['cluster'])
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
