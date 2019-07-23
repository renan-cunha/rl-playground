import matplotlib.pyplot as plt
import numpy as np

total = 0
gamma = 0.99
totals = []

for e in range(10000):
    total += 1*gamma**e
    totals.append(total)

plt.plot(totals)
plt.show()

