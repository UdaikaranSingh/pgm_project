from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.datasets import make_regression



def MyBaggingRegressor(base_model):
    
    return BaggingRegressor(
        base_estimator = base_model,
        n_estimators = 10, 
        random_state = 0
    )

def MyAdaBoostRegressor(base_model):

    return AdaBoostRegressor(
        base_estimator=base_model,
        n_estimators=10,
        random_state=0)

def MyTreeBaggingRegressor(depth):

    return RandomForestRegressor(
        max_depth = depth, 
        random_state=0)

def MyGradientBoostedRegressor(depth):

    return GradientBoostingRegressor(
        max_depth=depth,
        random_state=0)
        