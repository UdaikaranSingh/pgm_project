from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression



def MyBaggingRegressor():
    
    return BaggingRegressor(
        base_estimator = SVR(),
        n_estimators = 10, 
        random_state = 0
    )
        