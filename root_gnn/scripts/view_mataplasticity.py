import matplotlib.pyplot as plt
import numpy as np

def metaplasticity(m, W):
    x = np.tanh(m * W)
    return 1 - x * x


W = np.arange(-10, 10, step=0.01)
meta_values = [0, 0.2, 0.5, 0.8, 1.0]
y_vals = [metaplasticity(m, W) for m in meta_values]
[plt.plot(W, y, label='m={:.1f}'.format(m)) for m,y in zip(meta_values,y_vals)]
# plt.yscale('log')
plt.xlabel("weights", fontsize=14)
plt.ylabel("Metaplasticity", fontsize=14)
plt.title(r"meta = 1 - $\tanh^2$(m $\cdot$ W)", fontsize=14)
plt.legend(fontsize=12)
plt.savefig("metaplasticity.pdf")
# %%
