import numpy as np

marks = np.array([78, 35, 89, 72, 90, 56, 81, 42, 68, 85])
mean = np.mean(marks)
median = np.median(marks)
std = np.std(marks)
maximum = np.max(marks)
minimum = np.min(marks)

print("Internal Marks:", marks)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std)
print("Maximum:", maximum)
print("Minimum:", minimum)