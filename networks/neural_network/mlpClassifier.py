from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

def train_and_evaluate(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                        solver='adam', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=0.001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp.fit(X_train_scaled, y_train)

    # Save the trained model and scaler
    joblib.dump(mlp, 'mlp_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Predict on the test set
    y_pred = mlp.predict(X_test_scaled)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))