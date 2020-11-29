import matplotlib.pyplot as plt
from IPython.display import clear_output

plt.ion()

def create_plot():
    fig, ax = plt.subplots()  # Create a figure and an axes.
    return fig, ax

def plot_data(ax, data_dict):
    clear_output(wait=True)
    ax.clear()
    for k in data_dict.keys():
        ax.plot(data_dict[k], label=k)
    ax.set_xlabel('episods')
    ax.set_ylabel('rewards')
    ax.legend(loc='lower right')
    plt.pause(0.0001)


