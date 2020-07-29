import matplotlib.pyplot as plt
import numpy as np
import gym


def generate_learning_plot(rnds, scores, epsilon, fname, wndw):
    fig = plt.figure()
    axis = fig.add_subplot(111, label="1")
    axis2 = fig.add_subplot(111, label="2", frameon=False)

    axis.plot(rnds, epsilon, color="C0")
    axis.set_xlabel("Game", color="C0")
    axis.set_ylabel("Epsilon", color="C0")
    axis.tick_params(axis='x', color="C0")
    axis.tick_params(axis='x', color="C0")

    scores_store_length = len(scores)
    moving_average = np.empty(scores_store_length)
    for z in range(scores_store_length):
        moving_average[z] = np.mean(scores[max(0, z-wndw):(z+1)])

    axis2.scatter(rnds, moving_average, color="C1")
    axis2.axes.get_xaxis().set_visible(False)
    axis2.yaxis.tick_right()

    axis2.set_ylabel('Score', color="C1")
    axis2.yaxis.set_label_position('right')
    axis2.tick_params(axis='y', colors="C1")

    plt.savefig(fname)






