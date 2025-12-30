from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.utils import resample

def bootstrap_predictions(model, X_train, y_train, X_pred, n_iterations=100):
    predictions = []
    
    for i in range(n_iterations):
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        
        model.fit(X_boot, y_boot)

        pred = model.predict(X_pred)
        predictions.append(pred[0])
    
    predictions = np.array(predictions)
    
    lower = np.percentile(predictions, 10)
    median = np.percentile(predictions, 50)
    upper = np.percentile(predictions, 90)
    
    return lower, median, upper

def predict_category(stat, df, estimators):

    y = df[stat]

    X = df

    targets_to_drop = [stat]

    X = df.drop(columns=[col for col in targets_to_drop if col in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    model = RandomForestRegressor(
        n_estimators=estimators,      
        max_depth=4,          
        random_state=42,      
        verbose=0,            
        n_jobs=-1            
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    model_features = X.columns.tolist()

    return r2, rmse, model, model_features

def predict_category_xg(stat, df):

    y = df[stat]

    X = df

    targets_to_drop = [stat]

    X = df.drop(columns=[col for col in targets_to_drop if col in df.columns], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    model_features = X.columns.tolist()

    return r2, rmse, model, model_features, X_train, y_train
