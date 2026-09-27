import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data={
    "Age":[22,25,None,30,28],
    "Salary":[25000,35000,30000,None,45000],
    "Years_Experience":[1,3,2,None,5]
}
df=pd.DataFrame(data)

features=["Age","Salary","Years_Experience"]
imputer=SimpleImputer(strategy="median")
X=imputer.fit_transform(df[features])

standard_scaler=StandardScaler()
X_standard=standard_scaler.fit_transform(X)

#MinMaxScaler
minmax_scaler=MinMaxScaler()
X_minmax=minmax_scaler.fit_transform(X)

print("---Original Data After Imputation---")
print(X)
print("\n---StandardScaler---")
print(X_standard)
print("\n---MinMaxScaler---")
print(X_minmax)
print("\n---Numerical Range---")
print("\nStandardScaler:")
print("Minimum:",X_standard.min())
print("Maximum:",X_standard.max())
print("\nMinMaxScaler:")
print("Minimum:",X_minmax.min())
print("Maximum:",X_minmax.max())