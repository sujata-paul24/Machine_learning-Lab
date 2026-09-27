import pandas as pd

data = {
    "Student Name": ["Rahul", "Priya", "Amit", "Sneha", "Riya"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [85, 76, 92, 68, 88],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
result=df[df["Marks"]>80]

print("\nStudents who scored above 80:")
print(result)