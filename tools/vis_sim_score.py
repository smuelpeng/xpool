import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(filepath):
    data = torch.load(filepath)
    scores = []
    for k,v in data.items():
        scores.append(v.reshape(-1))
    scores = np.array(scores)
    print(scores.shape)
    return scores

array = load_data(sys.argv[1])
# analyze the array
scores_max = np.max(array, axis=1)
scores_mean = np.mean(array, axis=1)
scores_max.sort()
scores_mean.sort()

scores_max = scores_max.reshape(-1, 1)
print('100:',scores_max[100], scores_mean[100])
print('200:',scores_max[200], scores_mean[200])
print('400:',scores_max[400], scores_mean[400])
print('500:',scores_max[500], scores_mean[500])
print('600:',scores_max[600], scores_mean[600])
print('700:',scores_max[700], scores_mean[700])
print('800:',scores_max[800], scores_mean[800])
print('900:',scores_max[900], scores_mean[900])


# Visualize the array
plt.figure(figsize=(10, 100))
plt.imshow(scores_max, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Visualization of test_itm.csv Array')
plt.savefig('test.png')



    
    