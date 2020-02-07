from statistics import *
from datetime import datetime
import csv
import math

#for hw2 because its stupid and i need data

data = {}

n = 100
s_mean = 2
std = 9

def getSample(i):
    sT = datetime.now()
    return NormalDist(s_mean, std).samples(n, seed = i)



with open('hw2_100.csv', mode = '+w') as csv_file:
    fieldnames = ['equation1', 'equation2']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(100):
        y = getSample(i)
        eq1 = (mean(y) - 2)/(math.sqrt(std/n))
        eq2 = ((n-1)*stdev(y)) / std
        writer.writerow({'equation1': eq1, 'equation2' : eq2})

