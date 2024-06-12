import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# Load your data
df = pd.read_excel('salarydataset.xlsx')

# Let's assume 'BaseSalary', 'DegreeFinished', 'Experience', 'SubjectsTaught', 'ClassTaught', 'StudentStrength' are the features and 'Salary' is the target
X = df[['BaseSalary', 'DegreeFinished', 'Experience', 'SubjectsTaught', 'ClassTaught', 'StudentStrength']]
y = df['Salary']

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

print ("Enter your number of degree finished ")
# Make prediction
predictions = model.predict(X_test)

#Print the predictions
print('Predictions:', predictions)
print("training score:{}".format(model.score(X_train, y_train)))
print("test score:{}".format( model.score(X_test, y_test)))
