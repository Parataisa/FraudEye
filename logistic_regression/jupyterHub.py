import joblib 

def leader_board_predict_fn(values):
  model = joblib.load("./logistic_regression_model.pkl")
  return model.predict(values)
