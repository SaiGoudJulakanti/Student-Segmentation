
# Import NumPy to generate and process numerical data.
import numpy as np

# Import Pandas to create and analyze the student dataset.
import pandas as pd

# Import Matplotlib to display graphs.
import matplotlib.pyplot as plt

# Import StandardScaler to put features on comparable scales.
from sklearn.preprocessing import StandardScaler

# Import KMeans to perform student clustering.
from sklearn.cluster import KMeans

# Import silhouette_score to evaluate cluster separation.
from sklearn.metrics import silhouette_score

# Import PCA to reduce four features to two dimensions for visualization.
from sklearn.decomposition import PCA

# Set a random seed so the experiment can be reproduced.
np.random.seed(42)

# Define the number of students in each simulated academic group.
students_per_group = 60

# Define example average values for four simulated behavior patterns.
# The four columns represent attendance, study hours, assignments, and exam scores.
group_means = [
    [92, 4.5, 95, 88],  # Strong attendance and academic performance.
    [75, 2.8, 78, 68],  # Moderate academic behavior.
    [58, 1.5, 55, 48],  # Lower attendance and academic performance.
    [85, 2.0, 65, 60]   # Good attendance but less study and assignment completion.
]

# Define the variation around each group's average values.
group_stds = [6, 0.7, 9, 8]

# Create an empty list to store the simulated student records.
student_records = []

# Generate students for each simulated group.
for means in group_means:
    # Generate numerical features around the current group's average.
    generated = np.random.normal(
        loc=means,
        scale=group_stds,
        size=(students_per_group, 4)
    )

    # Limit attendance and assignment completion to valid percentages.
    generated[:, 0] = np.clip(generated[:, 0], 0, 100)
    generated[:, 2] = np.clip(generated[:, 2], 0, 100)

    # Limit study hours to a reasonable range of zero to twelve hours.
    generated[:, 1] = np.clip(generated[:, 1], 0, 12)

    # Limit examination scores to the range zero to one hundred.
    generated[:, 3] = np.clip(generated[:, 3], 0, 100)

    # Add the generated students to the complete dataset.
    student_records.extend(generated)

# Create a DataFrame with descriptive column names.
df = pd.DataFrame(
    student_records,
    columns=[
        "Attendance",
        "Study_Hours",
        "Assignment_Completion",
        "Exam_Score"
    ]
)

# Create a unique ID for every simulated student.
df.insert(0, "Student_ID", range(1, len(df) + 1))

# Display the first five student records.
print("First five student records:")
print(df.head())

# Display the number of rows and columns in the dataset.
print("\nDataset shape:", df.shape)

# Display summary statistics for all academic behavior features.
print("\nDataset summary:")
print(df.describe().round(2))

# Select the features that the clustering algorithm will use.
features = [
    "Attendance",
    "Study_Hours",
    "Assignment_Completion",
    "Exam_Score"
]

# Create the feature matrix without the student ID.
X = df[features]

# Standardize the features because they use different measurement scales.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set K to four to explore four possible student behavior clusters.
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Fit K-Means to the standardized features and assign each student a cluster.
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Calculate the silhouette score to measure cluster separation.
score = silhouette_score(X_scaled, df["Cluster"])

# Print the silhouette score.
print("\nSilhouette Score:", round(score, 3))

# Count the number of students assigned to each cluster.
print("\nStudents in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Calculate the average academic behavior features for each cluster.
cluster_profiles = df.groupby("Cluster")[features].mean().round(2)

# Display the cluster profiles for interpretation.
print("\nAverage academic behavior by cluster:")
print(cluster_profiles)

# Count students in each cluster and display the counts.
cluster_sizes = df["Cluster"].value_counts().sort_index()

# Create a bar chart to compare cluster sizes.
plt.figure(figsize=(8, 5))
cluster_sizes.plot(kind="bar")
plt.title("Number of Students in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Create PCA to reduce the four standardized features to two dimensions.
pca = PCA(n_components=2)

# Transform the standardized features into two principal components.
X_pca = pca.fit_transform(X_scaled)

# Plot each student using the two principal components.
plt.figure(figsize=(9, 6))
for cluster_id in sorted(df["Cluster"].unique()):
    # Select the students belonging to the current cluster.
    mask = df["Cluster"] == cluster_id

    # Plot the selected students in the two-dimensional PCA space.
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.7
    )

# Add a title and axis labels to explain the PCA visualization.
plt.title("Student Segmentation Using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Display a legend identifying the cluster IDs.
plt.legend()

# Adjust the layout and display the scatter plot.
plt.tight_layout()
plt.show()

# Display the proportion of variance explained by the two PCA components.
print(
    "\nVariance explained by two PCA components:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

# Save the student records and their cluster assignments to a CSV file.
df.to_csv("student_segmentation_results.csv", index=False)

# Confirm that the results have been saved.
print("\nResults saved to student_segmentation_results.csv")
