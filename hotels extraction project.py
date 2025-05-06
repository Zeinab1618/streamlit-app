import pandas as pd  
import numpy as np   
import seaborn as sns 
import matplotlib.pyplot as plt  
import re
import streamlit as st

st.title("Hotel Data Analysis")

df = pd.read_csv('hotels_data.csv')

st.write(df.head())
st.write(df.info())

st.write("\nMissing values in the dataset:")
st.write(df.isnull().sum())

df.replace('N/A', pd.NA, inplace=True)

df['rating'] = df['rating'].fillna(df['rating'].median())
df['category'] = df['category'].fillna('Unknown')  

df.dropna(subset=['name', 'price', 'type'], inplace=True)

duplicates = df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicates}")

# Remove 'US$' and commas, convert price to float
df['price'] = df['price'].apply(lambda x: float(re.sub(r'[^\d.]', '', x)))
st.write("Sample Cleaned Prices:\n", df['price'].head())

# Extract numeric value from 'number of reviews' using Regex
def clean_reviews(review_str):
    if pd.isna(review_str):
        return 0
    match = re.search(r'(\d{1,3}(?:,\d{3})*)', review_str)
    return int(match.group(1).replace(',', '')) if match else 0

df['number of reviews'] = df['number of reviews'].apply(clean_reviews)
st.write("Sample Cleaned Number of Reviews:\n", df['number of reviews'].head())

# Extract numeric value and ensure it's in kilometers
def clean_distance(distance_str):
    if pd.isna(distance_str):
        return None
    match = re.search(r'(\d+\.?\d*)', distance_str)
    return float(match.group(1)) if match else None

df['Distance from Downtown'] = df['Distance from Downtown'].apply(clean_distance)
st.write("Sample Cleaned Distances:\n", df['Distance from Downtown'].head())

def clean_name(name):
    # Remove special characters, multiple spaces, and trim the name
    cleaned = re.sub(r'\s+', ' ', re.sub(r'[^\w\s-]', '', name)).strip()
    return cleaned.title()  # Capitalize first letter of each word

df['name'] = df['name'].apply(clean_name)
st.write("Sample Cleaned Names:\n", df['name'].head())

st.write("Unique values in 'category' before standardization:")
st.write(df['category'].unique())

category_counts = df['category'].value_counts()
st.write("Category Counts:")
st.write(category_counts)

# Ensure rating is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Change column types for consistency
df['price'] = df['price'].astype(float)
df['number of reviews'] = df['number of reviews'].astype(int)
df['Distance from Downtown'] = df['Distance from Downtown'].astype(float)
df['rating'] = df['rating'].astype(float)

# Display missing values after cleaning and data types
st.write("Missing Values After Cleaning:\n", df.isnull().sum())
st.write("Data Types:\n", df.dtypes)
st.write("Sample Cleaned Data:\n", df.head())

# Show price statistics
st.write("Price Statistics Before Cleaning:")
st.write(f"Mean: {df['price'].mean()}")
st.write(f"Median: {df['price'].median()}")
st.write(f"Min: {df['price'].min()}")
st.write(f"Max: {df['price'].max()}")
st.write(f"Standard Deviation: {df['price'].std()}")
st.write(f"Number of rows: {len(df)}")

# Box Plot to Check for Outliers
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

# Show row count before and after filtering outliers
st.write(f"Before: {len(df)} rows")
st.write(f"After: {len(dff)} rows")

# Box Plot After Removing Outliers
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(x=dff['price'])
ax.set_title('Box Plot of Hotel Prices (After Removing Outliers)')
ax.set_xlabel('Price (USD)')
st.pyplot(fig)

stats = dff[['price', 'rating', 'number of reviews', 'Distance from Downtown']].describe()
st.write(stats)

# Correlation matrix
correlation_matrix = dff[['price', 'rating', 'number of reviews', 'Distance from Downtown']].corr()
st.write("Correlation Matrix:")
st.write(correlation_matrix)

# Heatmap of correlations
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title('Correlation Heatmap of Variables')
st.pyplot(fig)

# Average price by category
avg_price_by_category = dff.groupby('category')['price'].mean().reset_index()
st.write("Average Price by Category:")
st.write(avg_price_by_category)

# Scatter plot showing Price vs. Rating with Distance and Reviews
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='rating', y='price', size='number of reviews', hue='Distance from Downtown',
                data=dff, sizes=(50, 500), palette='viridis', ax=ax)
ax.set_title('Price vs. Rating with Distance and Number of Reviews')
ax.set_xlabel('Rating')
ax.set_ylabel('Price')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# Bar plot for Average Price by Category
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='category', y='price', data=avg_price_by_category, palette='Blues', ax=ax)
ax.set_title('Average Price by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Average Price')
st.pyplot(fig)

# Histogram for Price Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(dff['price'], bins=20, kde=True, color='skyblue', ax=ax)
ax.set_title('Price Distribution')
ax.set_xlabel('Price')
ax.set_ylabel('Count')
st.pyplot(fig)

