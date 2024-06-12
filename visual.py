import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel('salarydataset.xlsx')

# Let's assume 'BaseSalary', 'DegreeFinished', 'Experience', 'SubjectsTaught', 'ClassTaught', 'StudentStrength' are the features and 'Salary' is the target
X = df[['BaseSalary', 'DegreeFinished', 'Experience', 'SubjectsTaught', 'ClassTaught', 'StudentStrength']]
y = df['Salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Plot the decision tree
fig, ax= plt.subplots(figsize=(15, 10))  # Adjust the size of the plot as needed
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.show()
