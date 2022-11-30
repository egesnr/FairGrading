import csv
import matplotlib.pyplot as plot
import numpy as np

rows = []
with open("Grading_Assignment.csv","r") as file:

    read = csv.reader(file)

    header = next(read)

    for row in read:
        rows.append(row)


plot.scatter(row)