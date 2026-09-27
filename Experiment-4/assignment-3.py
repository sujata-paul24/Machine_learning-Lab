import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("Housing.csv.xls")
X = df[["area"]]
y = df["price"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25,random_state=42
)

linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
y_pred_linear=linear_model.predict(X_test)
r2_linear=r2_score(y_test,y_pred_linear)

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

poly_model=LinearRegression()
poly_model.fit(X_train_poly,y_train)
y_pred_poly = poly_model.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("---R2 Score Comparison---")
print("Linear Regression R2    :",round(r2_linear,4))
print("Polynomial Regression R2:",round(r2_poly,4))

X_sorted=X_test.sort_values(by="area")
y_linear_sorted=linear_model.predict(X_sorted)
y_poly_sorted=poly_model.predict(poly.transform(X_sorted))

plt.scatter(X_test,y_test,label="Actual Data")
plt.plot(
    X_sorted,
    y_linear_sorted,
    linewidth=2,
    label="Linear Regression"
)
plt.plot(
    X_sorted,
    y_poly_sorted,
    linewidth=2,
    label="Polynomial Regression"
)
plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()