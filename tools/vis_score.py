# It seems there was an issue in the previous attempt, let's try to load and visualize the array again
import numpy as np
import matplotlib.pyplot as plt

# set fig size
# Load the CSV file
file_path = 'test_itm.csv'
array = np.loadtxt(file_path, delimiter=',')

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


# Visualize the array
plt.figure(figsize=(10, 100))
plt.imshow(scores_max, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Visualization of test_itm.csv Array')
plt.savefig('test.png')

