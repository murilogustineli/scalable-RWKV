import matplotlib.pyplot as plt

def plot_losses(parsed_data1, parsed_data2, label1, label2):
    # Extracting the loss values for both datasets
    loss_values1 = parsed_data1[:, 0]
    loss_values2 = parsed_data2[:, 0]

    # Steps (assuming each entry is one step), divided by 2
    steps1 = [x / 2 for x in range(len(loss_values1))]
    steps2 = [x / 2 for x in range(len(loss_values2))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps1, loss_values1, label=label1)
    plt.plot(steps2, loss_values2, label=label2)
    plt.xlabel('Steps (halved)')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps for Two Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

