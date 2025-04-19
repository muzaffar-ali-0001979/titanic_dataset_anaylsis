import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load the dataset
# We assume the CSV file is named 'tested.csv'
df = pd.read_csv('titanic.csv')

# Step 2: Clean null values
# Check for null values to understand what's missing
print("Null values before cleaning:")
print(df.isnull().sum())

# Fill missing 'Age' with median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' with the most common value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' column due to too many missing values
df = df.drop(columns=['Cabin'])

# Drop any remaining rows with null values (if any)
df = df.dropna()

# Confirm null values are handled
print("\nNull values after cleaning:")
print(df.isnull().sum())

# Step 3: Analyze the data
# Total number of people
total_people = len(df)
print(f"\nTotal number of people: {total_people}")

# Count males and females
sex_counts = df['Sex'].value_counts()
print("\nNumber of males and females:")
print(sex_counts)

# Create age groups
bins = [0, 18, 30, 50, 100]  # Age ranges: 0-18, 19-30, 31-50, 51+
labels = ['0-18', '19-30', '31-50', '51+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_group_counts = df['AgeGroup'].value_counts().sort_index()
print("\nNumber of people in each age group:")
print(age_group_counts)

# Count people in each Pclass
pclass_counts = df['Pclass'].value_counts().sort_index()
print("\nNumber of people in each Pclass:")
print(pclass_counts)

# Step 4: Calculate survival rates
# Survival rate by sex
survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival rate by sex (%):")
print(survival_by_sex)

# Survival rate by age group
survival_by_age_group = df.groupby('AgeGroup')['Survived'].mean() * 100
print("\nSurvival rate by age group (%):")
print(survival_by_age_group)

# Survival rate by Pclass
survival_by_pclass = df.groupby('Pclass')['Survived'].mean() * 100
print("\nSurvival rate by Pclass (%):")
print(survival_by_pclass)

# Step 5: Visualizations
# Set Seaborn style for better-looking plots
sns.set(style="whitegrid")

# Plot 1: Male vs. Female counts
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df)
plt.title('Number of Males and Females')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('sex_counts.png')
plt.close()

# Plot 2: Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()

# Plot 3: Pclass counts
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Number of Passengers in Each Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.savefig('pclass_counts.png')
plt.close()

# Plot 4: Survival rate by sex
plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate (%)')
plt.savefig('survival_by_sex.png')
plt.close()

# Plot 5: Survival rate by age group
plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_age_group.index, y=survival_by_age_group.values)
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate (%)')
plt.savefig('survival_by_age_group.png')
plt.close()

# Plot 6: Survival rate by Pclass
plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_pclass.index, y=survival_by_pclass.values)
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate (%)')
plt.savefig('survival_by_pclass.png')
plt.close()

print("\nPlots have been saved as PNG files.")