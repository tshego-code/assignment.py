# --------------------------------------------------------
# Assignment: Data Analysis and Visualization with Pandas & Matplotlib
# Dataset: Iris (via sklearn.datasets)
# --------------------------------------------------------

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Make plots look nicer
sns.set(style="whitegrid")

# Load dataset with error handling
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please check the dataset path.")
except Exception as e:
    print("❌ An error occurred while loading the dataset:", e)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore structure
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean missing values (not needed for Iris, but included for practice)
df = df.dropna()

# Add species names column (instead of numeric target)
df["species"] = df["target"].map(dict(enumerate(iris_data.target_names)))
python
Copy code
# Task 2: Basic Data Analysis

# Basic statistics
print("\nSummary Statistics (numerical columns):")
print(df.describe())

# Group by species and calculate mean
grouped = df.groupby("species").mean()
print("\nMean of numerical features per species:")
print(grouped)

# Observations
print("\nObservations:")
print("- Setosa has the smallest petal measurements overall.")
print("- Virginica tends to have the largest sepal and petal sizes.")
print("- Versicolor lies between Setosa and Virginica.")
python
Copy code
# Task 3: Data Visualization

# 1. Line Chart: Sepal length trend across samples
plt.figure(figsize=(8,4))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="teal")
plt.title("Line Chart: Sepal Length across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(6,4))
df.groupby("species")["petal length (cm)"].mean().plot(kind="bar", color=["skyblue","orange","green"])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal width distribution
plt.figure(figsize=(6,4))
plt.hist(df["sepal width (cm)"], bins=15, edgecolor="black", color="purple", alpha=0.7)
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()