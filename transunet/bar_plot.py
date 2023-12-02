import matplotlib.pyplot as plt
import numpy as np
import mplcursors

scores = [[0.6767, 0.7495, 0.6994, 0.7756, 0.6289, 0.7287], [0.7132, 0.7856, 0.7546, 0.8288, 0.6469, 0.7491],
          [0.6677, 0.7829, 0.7120, 0.8182, 0.6289, 0.7520], [0.6906, 0.7651, 0.7230, 0.7981, 0.6309, 0.7383],
          [0.7281, 0.8300, 0.7727, 0.8659, 0.6892, 0.7985], [0.7033, 0.7727, 0.7436, 0.8168, 0.6346, 0.7285],
          [0.7094, 0.8117, 0.7184, 0.8089, 0.7016, 0.8142], [0.6925, 0.7632, 0.7191, 0.7925, 0.6446, 0.7424],
          [0.7402, 0.8300, 0.7996, 0.8710, 0.6883, 0.7941], [0.6977, 0.7665, 0.7493, 0.8181, 0.6336, 0.7337],
          [0.7477, 0.8447, 0.8130, 0.8930, 0.6906, 0.8024], [0.7018, 0.7755, 0.7329, 0.8104, 0.6211, 0.7213],
          [0.7253, 0.8289, 0.7705, 0.8636, 0.6857, 0.7986]]



# Sample data (replace this with your actual data)
methods = ['Ind BU S', 'Ind BU M', 'Ind OB', 'Co BU', 'Co OB',
           'Co Med 3 BU', 'Co Med 3 OB', 'Co Med 5 BU', 'Co Med 5 OB',
           'Co Med 7 BU', 'Co Med 7 OB', 'Co Med 9 BU', 'Co Med 9 OB']
metrics = ['Overall IoU', 'Overall Dice', 'Benign IoU', 'Benign Dice', 'Malignant IoU', 'Malignant Dice']


# Replace the following with your actual performance data

performance_data = {}

for method, metric in zip(methods, scores):
    performance_data[method] = metric


# Set up the larger figure size
fig, ax = plt.subplots(figsize=(48, 36))

# Set up the bar plot
bar_width = 0.1
index = np.arange(len(methods))

# Plot each metric for each method
for i, metric in enumerate(metrics):
    ax.bar(index + i * bar_width, [performance_data[method][i] for method in methods], bar_width, label=metric)

# Set labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Performance')
ax.set_title('Performance of Different Methods on Multiple Metrics')
ax.set_xticks(index + (len(metrics) - 1) * bar_width / 2)
ax.set_xticklabels(methods)
ax.legend()

# Show the plot
plt.show()
fig.savefig('All.png')

# Calculate the number of figures needed
num_figures = len(metrics) // 3 + (len(metrics) % 3 > 0)

# Create separate figures for each set of 3 metrics
for figure_num in range(num_figures):
    start_metric = figure_num * 3
    end_metric = min((figure_num + 1) * 3, len(metrics))

    # Set up the figure with subplots stacked vertically
    fig, axes = plt.subplots(nrows=end_metric - start_metric, ncols=1, figsize=(16, 3 * (end_metric - start_metric)))

    # Plot each metric for each method in a separate subplot
    for i in range(start_metric, end_metric):
        ax = axes[i - start_metric]
        metric = metrics[i]
        ax.bar(methods, [performance_data[method][i] for method in methods])
        ax.set_title(f'{metric} Performance')
        ax.set_ylabel('Performance')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
    fig.savefig(str(figure_num) + '.png')


# Set up the larger figure size
fig, ax = plt.subplots(figsize=(48, 36))

# Set up the bar plot
bar_width = 0.1
index = np.arange(len(methods))

# Plot each metric for each method
for i, metric in enumerate(metrics):
    bars = ax.bar(index + i * bar_width, [performance_data[method][i] for method in methods], bar_width, label=metric)

    # Enable mplcursors for the current bars
    mplcursors.cursor(bars, hover=True)

# Set labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Performance')
ax.set_title('Performance of Different Methods on Multiple Metrics')
ax.set_xticks(index + (len(metrics) - 1) * bar_width / 2)
ax.set_xticklabels(methods)
ax.legend()

# Show the plot
plt.show()