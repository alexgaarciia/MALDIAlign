import matplotlib.pyplot as plt


def handle_save_show(save, path):
    """
    Save or show plot and close the figure.
    """

    if save and path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        

def plot_misclassified(ax, mis_points, sp=None, hosp=None):
    """
    Overlay misclassified samples on the given axes.
    Filters by species and/or hospital if provided.
    """

    if mis_points is None or len(mis_points) == 0:
            return
        
    mis = mis_points.copy()
    if sp is not None:
        mis = mis[mis["species"] == sp]
    if hosp is not None:
        mis = mis[mis["hospital"] == hosp]
            
    if len(mis) > 0:
        ax.scatter(mis["x"], mis["y"], marker="x", s=45, color="purple", alpha=0.9, zorder=15, label="Misclassified")


def plot_prior_star(ax, subset_prior, color_idx, label="Prior samples"):
    """
    Plot prior samples as large stars with black edges to make them stand out.
    """

    if len(subset_prior) > 0:
        ax.scatter(subset_prior["x"], subset_prior["y"], marker="*", s=150, color=plt.cm.tab10(color_idx), edgecolors="black", linewidths=1.2, zorder=10, label=label)
