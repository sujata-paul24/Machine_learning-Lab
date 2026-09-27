import pandas as pd
import numpy as np

data={
    "Age":[22,25,28,None,35,40,45,50],
    "Salary":[25000,30000,None,40000,50000,60000,70000,80000],
    "Department":["IT","HR","IT",None,"HR","IT","Finance","IT"],
    "Years of Experience":[1,2,4,5,8,12,None,20]
}

df=pd.DataFrame(data)
df[["Age","Salary","Years of Experience"]]=df[
    ["Age","Salary","Years of Experience"]
].fillna(
    df[["Age","Salary","Years of Experience"]].mean()
)

df["Department"] = df["Department"].fillna("Unknown")
print("Preprocessed Dataset:")
print(df)