import pandas as pd

data ={
    "Student Name": ["Rahul","Priya","Amit","Sneha","Riya"],
    "Roll Number": [101,102,103,104,105],
    "Marks": [93,89,77,68,42]
}

df = pd.DataFrame(data)

def calculate_grade(marks):
    if marks >=90:
        return "O"
    elif marks >=80:
        return "E"
    elif marks >=70:
        return "A"
    elif marks >=60:
        return "B"
    else:
        return "F"

df["Grade"] =df["Marks"].apply(calculate_grade)
print(df)