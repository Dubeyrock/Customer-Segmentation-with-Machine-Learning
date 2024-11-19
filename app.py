import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
@st.cache
def load_data():
   # Using raw string literal to avoid the unicodeescape error
    df = pd.read_csv(r'C:\Users\dubey\Downloads\customer-segmentation\Source Code\customer-segmentation-dataset\Mall_Customers.csv')

    return df

df = load_data()

# Title of the Streamlit app
st.title("Customer Segmentation Dashboard")

# Show data sample
st.write("### Customer Data", df.head())

# Preprocess the data
st.write("### Data Preprocessing")
df = df.dropna()  # Removing missing values

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Sidebar for user input
st.sidebar.header("Cluster Configuration")
n_clusters = st.sidebar.slider("Number of Clusters:", min_value=2, max_value=10, value=5)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clustering results (Seaborn Plot)
st.write(f"### Customer Segmentation (K-Means with {n_clusters} clusters)")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', ax=ax)
st.pyplot(fig)

# Plotly Interactive Visualization
fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Cluster',
                 title="Customer Segments",
                 labels={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spending Score'})
st.plotly_chart(fig)

# Display Cluster Summary
st.write("### Cluster Summary")
cluster_summary = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
})
st.write(cluster_summary)

# Add interactivity for filtering data by cluster
selected_cluster = st.sidebar.selectbox("Select a Cluster to View", df['Cluster'].unique())
filtered_data = df[df['Cluster'] == selected_cluster]
st.write(f"### Details for Cluster {selected_cluster}")
st.write(filtered_data)

