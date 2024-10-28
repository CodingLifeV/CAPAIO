from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("dpc_cluster")


def plot_rho_delta(local_sentity, delta):
	'''
	Plot scatter diagram for rho-delta points

	Args:
		rho   : local sentity list
		delta : delta list
	'''
	logger.info("PLOT: rho-delta plot")

	plot_scatter_diagram(10, local_sentity, delta, x_label=r'$\rho_i$', y_label=r'$\delta_i$',
						  title = r'cutoff distance $d_c$ = top 10% of distances')


def plot_scatter_diagram(which_fig, x, y, x_label='x', y_label='y', title='title', style_list=None):
    '''
    Plot scatter diagram
    Args:
        which_fig  : which sub plot
        x          : x array
        y          : y array
        x_label    : label of x pixel
        y_label    : label of y pixel
        title      : title of the plot
    '''

    assert len(x) == len(y)
    if style_list is not None:
        assert len(x) == len(style_list)

    plt.figure(which_fig, figsize=(8, 6))
    plt.clf()

    if style_list is None:
        plt.scatter(x, y, s=20**2, marker='o', edgecolors='#6A9655', linewidths=2.0, facecolors='none', alpha=0.8)
        for i, txt in enumerate(range(len(x))):
            plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 0), ha='center', va='center',
                         fontsize=12, color='#000000')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.savefig(f'plot_{which_fig}.png', dpi=300)
    plt.show()
