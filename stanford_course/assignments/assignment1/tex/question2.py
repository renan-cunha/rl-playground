import matplotlib.pyplot as plt

def action1(n, gamma):
    result = 0
    results = []
    for i in range(1, n+1):
        result += gamma**i
        results.append(result)
    return results

def action2(n, gamma):
    return [gamma**2/(1-gamma)]

n=10**3
results1 = action1(n, 0.5)
results2 = action2(n, 0.5)*n
plt.plot(results1, label="action1")
plt.plot(results2, label="action2")
plt.legend()
plt.show()
