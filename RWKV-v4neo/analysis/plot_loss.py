import matplotlib.pyplot as plt

def plot_loss(parsed_data):
    # Extracting the loss values
    loss_values = parsed_data[:, 0]

    # Steps (assuming each entry is one step)
    steps = [x / 2 for x in range(len(loss_values))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label='Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Extracting the loss values
    lr_values = parsed_data[:, 1]

    # Steps (assuming each entry is one step)
    steps = [x / 2 for x in range(len(lr_values))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr_values, label='lr')
    plt.xlabel('Steps')
    plt.ylabel('lr')
    plt.title('lr vs Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
# plot_loss(parsed_array)
