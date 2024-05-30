import os 
import pandas as pd

def get_data(): 
    # Correctly construct the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(current_dir, "mlproject22")
    path = project_dir if os.path.exists(project_dir) else current_dir
    csv_path = os.path.join(path, "transactions.csv.zip")

    
    # Check if the file exists at the path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    # Load the data
    train_data = pd.read_csv(csv_path)
    
    # Prepare features and labels
    X_train = train_data.drop(columns="Class")
    y_train = train_data["Class"]

    return X_train, y_train