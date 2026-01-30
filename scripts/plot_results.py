import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("assets", exist_ok=True)
times = [1, 3600, 86400, 31536000]
time_labels = ['1s', '1h', '1d', '1y']

# DATA
wrn_hwa = [76.95, 76.87, 76.94, 76.94]
bert_hwa = [90.37, 90.37, 90.37, 90.37]
lstm_hwa = [259.05, 258.89, 258.65, 259.09]

# PLOTTING
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# (Code simplifié pour le template, tu as déjà les images)
plt.savefig("assets/summary.png")
print("Plot script executed.")
