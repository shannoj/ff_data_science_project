from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_category(stat, df):

    y = df[stat]

    X = df

    X = X.drop([stat], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    model = RandomForestRegressor(
        n_estimators=50,      
        max_depth=4,          
        random_state=42,      
        verbose=0,            
        n_jobs=-1            
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return r2, rmse, model
