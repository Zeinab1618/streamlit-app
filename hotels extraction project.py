import pandas as pd  
import numpy as np   
import seaborn as sns 
import matplotlib.pyplot as plt  
import re
import streamlit as st
from pymongo import MongoClient
import certifi
st.title("Hotel Data Analysis")


df = pd.read_csv('hotels_data.csv')

st.write(df.head())

st.write("\nMissing values in the dataset:")
st.write(df.isnull().sum())

df.replace('N/A', pd.NA, inplace=True)

df['rating'] = df['rating'].fillna(df['rating'].median())
df['category'] = df['category'].fillna('Unknown')  

df.dropna(subset=['name', 'price', 'type'], inplace=True)

duplicates = df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicates}")

df['price'] = df['price'].apply(lambda x: float(re.sub(r'[^\d.]', '', x)))
st.write("Sample Cleaned Prices:\n", df['price'].head())

def clean_reviews(review_str):
    if pd.isna(review_str):
        return 0
    match = re.search(r'(\d{1,3}(?:,\d{3})*)', review_str)
    return int(match.group(1).replace(',', '')) if match else 0

df['number of reviews'] = df['number of reviews'].apply(clean_reviews)
st.write("Sample Cleaned Number of Reviews:\n", df['number of reviews'].head())

def clean_distance(distance_str):
    if pd.isna(distance_str):
        return None
    match = re.search(r'(\d+\.?\d*)', distance_str)
    return float(match.group(1)) if match else None

df['Distance from Downtown'] = df['Distance from Downtown'].apply(clean_distance)
st.write("Sample Cleaned Distances:\n", df['Distance from Downtown'].head())

def clean_name(name):
   
    cleaned = re.sub(r'\s+', ' ', re.sub(r'[^\w\s-]', '', name)).strip()
    return cleaned.title() 

df['name'] = df['name'].apply(clean_name)
st.write("Sample Cleaned Names:\n", df['name'].head())

st.write("Unique values in 'category' before standardization:")
st.write(df['category'].unique())

category_counts = df['category'].value_counts()
st.write("Category Counts:")
st.write(category_counts)


df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df['price'] = df['price'].astype(float)
df['number of reviews'] = df['number of reviews'].astype(int)
df['Distance from Downtown'] = df['Distance from Downtown'].astype(float)
df['rating'] = df['rating'].astype(float)

st.write("Missing Values After Cleaning:\n", df.isnull().sum())
st.write("Data Types:\n", df.dtypes)
st.write("Sample Cleaned Data:\n", df.head())

st.write("Price Statistics Before Cleaning:")
st.write(f"Mean: {df['price'].mean()}")
st.write(f"Median: {df['price'].median()}")
st.write(f"Min: {df['price'].min()}")
st.write(f"Max: {df['price'].max()}")
st.write(f"Standard Deviation: {df['price'].std()}")
st.write(f"Number of rows: {len(df)}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(x=df['price'])
ax.set_title('Box Plot of Hotel Prices (Before Removing Outliers)')
ax.set_xlabel('Price (USD)')
st.pyplot(fig)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dff = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

st.write(f"Before: {len(df)} rows")
st.write(f"After: {len(dff)} rows")

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(x=dff['price'])
ax.set_title('Box Plot of Hotel Prices (After Removing Outliers)')
ax.set_xlabel('Price (USD)')
st.pyplot(fig)

stats = dff[['price', 'rating', 'number of reviews', 'Distance from Downtown']].describe()
st.write(stats)

# keep the outliers near 2000 USD in Box Plot
# because they likely represent real prices for luxury hotels in cities like Paris or Tokyo, which can reasonably reach such values.
# and removing them might exclude valuable insights about premium hotel pricing.

"""##**1. Scatter Plot:** Price Vs Rating
Do better-rated hotels cost more?
"""

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='rating', y='price')
ax.set_title('Hotel Price vs Rating')
ax.set_xlabel('Rating')
ax.set_ylabel('Price (USD)')
ax.grid(True)
st.pyplot(fig)

"""##**2. Scatter Plot:** Distance from Downtown vs. Price

This shows if hotels closer to downtown are more expensive.

"""

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Distance from Downtown', y='price')
ax.set_title('Hotel Price vs Distance from Downtown')
ax.set_xlabel('Distance from Downtown (km)')
ax.set_ylabel('Price (USD)')
ax.grid(True)
st.pyplot(fig)

"""##**3. Bar Plot:** Hotel Category Counts (Excellent, Very Good, Good)
Shows how hotels are rated overall.
"""

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='category', hue='category', order=df['category'].value_counts().index, palette='viridis', legend=False)
ax.set_title('Hotel Category Distribution')
ax.set_xlabel('Category')
ax.set_ylabel('Number of Hotels')
ax.grid(axis='y')
st.pyplot(fig)

"""##**4. Correlation Heatmap**
Shows relationships between numbers: price, rating, distance, reviews
"""

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[['price', 'rating', 'number of reviews', 'Distance from Downtown']].corr(), annot=True, cmap='coolwarm')
ax.set_title('Correlation Between Hotel Features')
st.pyplot(fig)

"""##**5. Line Plot:** Average Hotel Price vs. Rating
See if better-rated hotels cost more on average.
"""


price_rating = df.groupby('rating')['price'].mean().reset_index()


fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=price_rating, x='rating', y='price', marker='o')
ax.set_title('Average Hotel Price vs Rating')
ax.set_xlabel('Rating')
ax.set_ylabel('Average Price (USD)')
ax.grid(True)
st.pyplot(fig)

"""##**6. Top 10 Most Expensive Hotels**
Show the most luxurious hotels.
"""


df = df[df['name'].apply(lambda x: bool(re.match('^[a-zA-Z0-9\s.,&\'-]+$', x)))]

top10_expensive = df.sort_values('price', ascending=False).head(10)


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top10_expensive, x='price', y='name', hue='name', palette='rocket', dodge=False, legend=False)
ax.set_title('Top 10 Most Expensive Hotels')
ax.set_xlabel('Price (USD)')
ax.set_ylabel('Hotel Name')
ax.grid(axis='x')
st.pyplot(fig)

"""##**7. Top 10 Closest Hotels to Downtown**
See which hotels are the nearest.

"""

top10_closest = df.sort_values('Distance from Downtown', ascending=True).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top10_closest, x='Distance from Downtown', y='name', hue='name', palette='crest', legend=False)
ax.set_title('Top 10 Closest Hotels to Downtown')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Hotel Name')
ax.grid(axis='x')
st.pyplot(fig)

"""##**8.Scatter Plot:** Reviews vs. Rating
Hotels with more reviews — do they have better ratings?
"""

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='number of reviews', y='rating', hue='category', palette='tab10')
ax.set_title('Hotel Rating vs Number of Reviews')
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('Rating')
ax.set_xscale('log')
ax.grid(True)
st.pyplot(fig)

# Upload to MongoDB
uri = st.secrets["mongo"]["uri"]
client = MongoClient(uri)
db = client['my_database']
collection = db['processed_data']

import streamlit as st

try:
    st.write("Mongo URI:", st.secrets["mongo"]["uri"][:10] + "...")  # Show first 10 chars only
except Exception as e:
    st.error(f"Secrets error: {e}")

# data = dff.to_dict("records")

# if st.button("Upload Cleaned Data to MongoDB"):
#     if collection.count_documents({}) == 0:
#         collection.insert_many(data)
#         st.success("Data uploaded to MongoDB!")
#     else:
#         st.info("Data already exists.")
# # for doc in collection.find():
# #     print(doc)
