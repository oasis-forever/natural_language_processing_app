import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.use("Agg")

# Define font for JP lang
plt.rcParams["font.family"] = "IPAPGothic"

def draw_barcharts(value_table, x_labels, titles, method_name):
    """
    value_table: 2 dimensional array of values to show
    x_labels: List for assignment of x-axis labels
    titles: List for assignment of title for each graph
    """
    n_rows, n_dims = value_table.shape

    if n_rows != len(titles):
        raise ValueError("value_table.shape[0] != len(titles)")

    if n_dims != len(x_labels):
        raise ValueError("value_table.shape[1] != len(x_labels)")

    x_labels = [str(x_label) for x_label in x_labels]

    # Define range of x-axis to show
    xmin = -1
    xmax = n_dims

    # Define range of y-axis to show
    # Conditional statement based an whether all number are 0 or more, or less than 0 for visibility
    show_negative = np.min(value_table) < 0
    ymin = np.min(value_table) - 0.5 if show_negative else 0
    ymax = np.max(value_table) + 0.5

    # Create subplot in each column
    fig, axs = plt.subplots(n_rows, 1, figsize=(n_dims, n_rows * 1.5))

    # Describe bars in each subplot
    for i, values in enumerate(value_table):
        ax = axs[i]
        ax.set_title(titles[i])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.hlines([0], xmin, xmax, linestyle="-", linewidth=1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(left="off", right="off", bottom="off")

        # Show x-axis lables only fot the graph on the bottom
        if i == n_rows - 1:
            ax.set_xticklabels(x_labels, rotation=0)
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig("../barcharts/{}.png".format(method_name))
