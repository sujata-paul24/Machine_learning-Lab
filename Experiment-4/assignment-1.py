import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df =pd.read_csv("Housing.csv.xls")
X=df[["area"]]
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25,random_state=42
)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Slope (b1):",model.coef_[0])
print("Intercept (b0):",model.intercept_)
print("\n---Evaluation Metrics---")
print("MAE :",round(mae,2))
print("MSE :",round(mse,2))
print("RMSE:",round(rmse,2))
print("R2  :",round(r2,4))

plt.scatter(X,y,label="Actual Data")
plt.plot(X,model.predict(X),linewidth=2,label="Regression Line")
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("House Area vs House Price")
plt.legend()
plt.grid(True)
plt.show()