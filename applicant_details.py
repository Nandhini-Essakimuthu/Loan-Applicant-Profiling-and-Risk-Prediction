

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Applicant_details.csv")

# Select relevant features for clustering
features = data[['Annual_Income', 'Applicant_Age', 'Work_Experience']]

# Perform scaling if needed
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow method, select the optimal number of clusters
# For simplicity, let's say it's 3
n_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Add cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Display the clustering results
print(data[['Applicant_ID', 'Annual_Income', 'Applicant_Age', 'Work_Experience', 'Cluster']])

# Visualize defaulters based on work experience
plt.figure(figsize=(10, 6))
sns.countplot(data=data[data['Loan_Default_Risk'] == 1], x='Work_Experience', palette='pastel')
plt.title('Defaulters by Work Experience')
plt.xlabel('Work Experience')
plt.ylabel('Count')
plt.show()

# Visualize defaulters based on marital status
plt.figure(figsize=(10, 6))
sns.countplot(data=data[data['Loan_Default_Risk'] == 1], x='Marital_Status', palette='pastel')
plt.title('Defaulters by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

# Get counts of defaulters for each residence state
state_counts = data[data['Loan_Default_Risk'] == 1]['Residence_State'].value_counts().head(5)

# Plot defaulters based on the top 5 residence states
plt.figure(figsize=(10, 6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette='pastel')
plt.title('Top 5 Defaulters by Residence State')
plt.xlabel('Residence State')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Get counts of defaulters for each residence city
city_counts = data[data['Loan_Default_Risk'] == 1]['Residence_City'].value_counts().head(5)

# Plot defaulters based on the top 5 residence cities
plt.figure(figsize=(10, 6))
sns.barplot(x=city_counts.index, y=city_counts.values, palette='pastel')
plt.title('Top 5 Defaulters by Residence City')
plt.xlabel('Residence City')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Applicant_details.csv")

# Get counts of defaulters for each occupation
occupation_counts = data[data['Loan_Default_Risk'] == 1]['Occupation'].value_counts().head(5)

# Plot defaulters based on the top 5 occupations
plt.figure(figsize=(10, 6))
sns.barplot(x=occupation_counts.index, y=occupation_counts.values, palette='pastel')
plt.title('Top 5 Defaulters by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("Applicant_details.csv")

# Prepare data for predicting occupation
X_occupation = data.drop(['Occupation'], axis=1)  # Features for predicting occupation
y_occupation = data['Occupation']  # Target variable (Occupation)

# Prepare data for predicting marital status
X_marital = data.drop(['Marital_Status'], axis=1)  # Features for predicting marital status
y_marital = data['Marital_Status']  # Target variable (Marital Status)

# Prepare data for predicting loan default risk categories
X_loan = data.drop(['Loan_Default_Risk'], axis=1)  # Features for predicting loan default risk
y_loan = data['Loan_Default_Risk']  # Target variable (Loan Default Risk)

# Split the data into training and testing sets
X_train_occ, X_test_occ, y_train_occ, y_test_occ = train_test_split(X_occupation, y_occupation, test_size=0.2, random_state=42)
X_train_mar, X_test_mar, y_train_mar, y_test_mar = train_test_split(X_marital, y_marital, test_size=0.2, random_state=42)
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Define and train RandomForestClassifier for predicting occupation
model_occupation = RandomForestClassifier(n_estimators=100, random_state=42)
model_occupation.fit(X_train_occ, y_train_occ)

# Define and train RandomForestClassifier for predicting marital status
model_marital = RandomForestClassifier(n_estimators=100, random_state=42)
model_marital.fit(X_train_mar, y_train_mar)

# Define and train RandomForestClassifier for predicting loan default risk categories
model_loan = RandomForestClassifier(n_estimators=100, random_state=42)
model_loan.fit(X_train_loan, y_train_loan)

# Predictions for occupation
predictions_occupation = model_occupation.predict(X_test_occ)
print("Occupation Classification Report:")
print(classification_report(y_test_occ, predictions_occupation))
print("Occupation Confusion Matrix:")
print(confusion_matrix(y_test_occ, predictions_occupation))

# Predictions for marital status
predictions_marital = model_marital.predict(X_test_mar)
print("\nMarital Status Classification Report:")
print(classification_report(y_test_mar, predictions_marital))
print("Marital Status Confusion Matrix:")
print(confusion_matrix(y_test_mar, predictions_marital))

# Predictions for loan default risk categories
predictions_loan = model_loan.predict(X_test_loan)
print("\nLoan Default Risk Classification Report:")
print(classification_report(y_test_loan, predictions_loan))
print("Loan Default Risk Confusion Matrix:")
print(confusion_matrix(y_test_loan, predictions_loan))
