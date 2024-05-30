import pandas as pd 
import os 

path = "./data/mlproject22" if os.path.exists("./data/mlproject22") else "."
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns = "Class")
y_train = train_data["Class"]

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", y_train.shape)
print(train_data.head())
print(train_data.info())
print(train_data.describe())
print(train_data['Class'].value_counts())

print(train_data)


