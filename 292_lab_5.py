import pandas as pd
import matplotlib.pyplot as plt

# Load the data from your CSV file
data = pd.read_csv('your_file.csv')

# Plot the data
plt.figure(figsize=(10,6))
plt.plot(data)