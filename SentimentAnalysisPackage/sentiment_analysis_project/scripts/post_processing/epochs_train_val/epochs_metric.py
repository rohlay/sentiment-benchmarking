import matplotlib.pyplot as plt

# Replace these lists with your epochs and corresponding final accuracies
epochs_list = [5, 10, 15, 20, 30, 50]
final_accuracies = [0.754, 0.753, 0.754, 0.751, 0.756, 0.742]
# accuracy, MLP, sst2_2

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs_list, final_accuracies, '-o', color='orange')  # '-o' creates a line plot with circles at the data points

# Customize the plot
plt.title('Final Evaluation Accuracy by Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Final Evaluation Accuracy')
plt.grid(True)

plt.ylim(0, 1)

# Optionally, you can add a line for each point to make it easier to read the exact values.
for i, txt in enumerate(final_accuracies):
    plt.annotate(txt, (epochs_list[i], final_accuracies[i]))


plt.savefig(r"C:\Users\Usuario\Desktop\epoch_metric_plots.png", dpi=300)
# Show the plot
plt.show()
