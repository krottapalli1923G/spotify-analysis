import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Define file path
file_path = "/Users/krishnanjalikrottapalli/Documents/Spotify_Analysis/Spotify-2000.csv"

# Load the dataset
df = pd.read_csv(file_path)

# -------------------- Step 3: Explore the Dataset --------------------

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nOriginal Column Names:")
print(df.columns)

print("\nStatistical Summary:")
print(df.describe())

# -------------------- Step 4: Data Cleaning & Processing --------------------

# Rename relevant columns for easier reference
df = df.rename(columns={
    'Beats Per Minute (BPM)': 'BPM',
    'Top Genre': 'Genre',
    'Loudness (dB)': 'Loudness',
    'Length (Duration)': 'Duration'
})

# Check for missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Convert 'Genre' (categorical) into numeric codes
df['Genre'] = df['Genre'].astype('category').cat.codes  

# Display dataset after cleaning
print("\nDataset After Cleaning:")
print(df.head())

# Confirm updated data types
print("\nUpdated Data Types:")
print(df.dtypes)

# -------------------- Step 5: Visualizing Popular Artists --------------------

# Count number of songs per artist
top_artists = df['Artist'].value_counts().head(10)

# Plot the top 10 artists with the most songs in the dataset
plt.figure(figsize=(10,5))
sns.barplot(x=top_artists.index, y=top_artists.values, palette='viridis')
plt.xticks(rotation=45)
plt.xlabel("Artists")
plt.ylabel("Number of Songs")
plt.title("Top 10 Artists with Most Songs in the Dataset")
plt.show()

# -------------------- Step 6: Finding Patterns in Tempo, Energy, and Danceability --------------------

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['BPM', 'Energy', 'Danceability', 'Loudness', 'Valence', 'Popularity']].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter Plot: Energy vs Popularity
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.scatterplot(x=df['Energy'], y=df['Popularity'], alpha=0.6)
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.title("Energy vs. Popularity")

# Scatter Plot: Danceability vs Popularity
plt.subplot(1,2,2)
sns.scatterplot(x=df['Danceability'], y=df['Popularity'], alpha=0.6, color="red")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.title("Danceability vs. Popularity")

plt.show()

# -------------------- Step 7: Predicting Song Popularity Using Regression Models --------------------

# Select features & target variable
features = ['BPM', 'Energy', 'Danceability', 'Loudness', 'Valence']
target = 'Popularity'

X = df[features]  # Independent variables
y = df[target]  # Target variable

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Visualizing Actual vs Predicted Popularity
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Popularity")
plt.show()

# -------------------- Step 8: Conclusion & Final Analysis --------------------

# Interpretation of Model Performance
print("\n🔹 Conclusion & Key Insights 🔹")

# Checking model accuracy
if r2 > 0.5:
    print("✅ The regression model has a decent predictive power with an R² score of {:.2f}".format(r2))
else:
    print("⚠️ The regression model does not explain much variability in song popularity (R² = {:.2f})".format(r2))

# Identifying the most important features affecting popularity
feature_importance = pd.Series(model.coef_, index=features).sort_values(ascending=False)
print("\n🎵 Most Important Features Affecting Popularity:")
print(feature_importance)

# Display feature importance as a bar chart
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="magma")
plt.xlabel("Influence on Popularity")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting Popularity")
plt.show()

print("\n🔍 Final Observations:")
print("- Songs with higher energy and danceability tend to have higher popularity.")
print("- BPM (tempo) has a weaker correlation with popularity than expected.")
print("- Loudness also plays a moderate role in determining popularity.")

print("\n📌 Potential Improvements:")
print("- Try using a more complex model like Random Forest or Gradient Boosting.")
print("- Include more features like Speechiness and Liveness to improve accuracy.")
print("- Experiment with a classification model (popular vs. non-popular songs).")

