import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
rows = []
with open("Implementation\Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)

'''
df = pd.read_csv("Implementation\Grading_Assignment.csv")
print(df.head())

teacher1 = df['sc1']
teacher2 = df['sc2']
plt.title('Grading Dataset')
plt.ylabel('Frequency')
plt.xlabel('Grades')
plt.hist(teacher1)
plt.hist(teacher2,color = 'green',alpha = 0.5)
plt.show()

