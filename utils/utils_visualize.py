import numpy as np
from matplotlib import pyplot as plt

from utils.utils_loading import get_model_data


def visualize_reconstruction(model_path, data_path, cuda=True, batch_size=10):
    model, model_config, dataset, loader = get_model_data(model_path, data_path, cuda, batch_size)

    image = model.visualize_reconstruction(
        next(iter(loader))[0].to('cuda' if cuda else 'cpu')
    )
    plt.imshow(image)
    plt.show()


def visualize_action_binding(model_path, data_path, cuda=True, batch_size=10):
    model, model_config, dataset, loader = get_model_data(model_path, data_path, cuda, batch_size)

    batch = next(iter(loader))
    obs = batch[0].to('cuda' if cuda else 'cpu')
    action = batch[1].to('cuda' if cuda else 'cpu')

    attention = model.get_action_binding(obs=obs, action=action)
    plt.imshow(attention, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


def plot_binding(info, dataset):
    # TODO abstract out the implementation of visualization and feeding information

    # > Use subplot (or switch to `plotly` if necessary)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6), constrained_layout=True)
    # [ax.set_visible(False) for ax in axes.flat]
    # [ax.axis('off') for ax in axes.flat]

    ax = axes[0, 0]
    cax = ax.matshow(
        info['Visualization/ActionMatrix'],
        cmap=plt.cm.Oranges, vmin=0, vmax=1
    )
    # ax.set_title('ActionMatrix')
    ax.set_title('Action')
    fig.colorbar(cax, ax=ax)

    # TODO add visualization for all objects
    # ax = axes[0, 8]
    ax = axes[1, 0]
    cax = ax.imshow(dataset.obj_vis.transpose(1, 2, 0))
    ax.set_title('ObjectLibrary')

    # ax = axes[1, 0]
    # ax = axes[1, 4]
    ax = axes[1, 3]
    cax = ax.matshow(
        info['Visualization/LatentActionMatrix'],
        cmap=plt.cm.RdYlGn,  # > Diverging colormap
        # vmin=0, vmax=1
    )
    # ax.set_title('LatentActionMatrix')
    ax.set_title('LatentAction')
    fig.colorbar(cax, ax=ax)

    ax = axes[0, 1]
    cax = ax.matshow(
        info['Visualization/ActionAttentionMatrix'],
        cmap=plt.cm.Blues, vmin=0, vmax=1
    )
    # ax.set_title('ActionAttentionMatrix')
    ax.set_title('ActionAttention')
    fig.colorbar(cax, ax=ax)

    ax = axes[1, 1]
    cax = ax.matshow(
        info['Visualization/ActionAttentionNextMatrix'],
        cmap=plt.cm.Blues, vmin=0, vmax=1
    )
    # ax.set_title('ActionAttentionNextMatrix')
    ax.set_title('ActionAttention-Next')
    fig.colorbar(cax, ax=ax)

    # > Compute shared min and max for consistency
    state_keys = ['Visualization/Embedding-(FullMDP)', 'Visualization/EmbeddingNext-(FullMDP)']
    state_max = np.max([info[k].max() for k in state_keys])
    state_min = np.min([info[k].min() for k in state_keys])

    ax = axes[0, 2]
    cax = ax.matshow(
        info['Visualization/Embedding-(FullMDP)'],
        cmap=plt.cm.Spectral,  # > Diverging colormap
        vmin=state_min, vmax=state_max
    )
    # ax.set_title('Embedding-(FullMDP)')
    ax.set_title('Embedding(FullMDP)')
    fig.colorbar(cax, ax=ax)

    ax = axes[1, 2]
    cax = ax.matshow(
        info['Visualization/EmbeddingNext-(FullMDP)'],
        cmap=plt.cm.Spectral,  # > Diverging colormap
        vmin=state_min, vmax=state_max
    )
    # ax.set_title('EmbeddingNext-(FullMDP)')
    ax.set_title('Embedding-Next')
    fig.colorbar(cax, ax=ax)

    ax = axes[0, 3]
    cax = ax.matshow(
        info['Visualization/Embedding-Difference-(FullMDP)'],
        cmap=plt.cm.RdBu,  # > Diverging colormap
        vmin=-(state_max - state_min) * 0.5, vmax=(state_max - state_min) * 0.5
    )
    # ax.set_title('Embedding-Difference-(FullMDP)')
    ax.set_title('Embedding-Difference')
    fig.colorbar(cax, ax=ax)

    # ax = axes[0, 4]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Difference-(SlotMDP)'],
    #     cmap=plt.cm.RdBu,  # > Diverging colormap
    #     # vmin=-(state_max - state_min) * 0.5, vmax=(state_max - state_min) * 0.5
    # )
    # ax.set_title('Slot-Difference-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)

    # ax = axes[1, 3]
    # cax = ax.matshow(
    #     info['Visualization/Embedding-Difference-Hard-(FullMDP)'],
    #     cmap=plt.cm.RdBu,  # > Diverging colormap
    #     vmin=-(state_max - state_min) * 0.5, vmax=(state_max - state_min) * 0.5
    # )
    # ax.set_title('Embedding-Difference-Hard-(FullMDP)')
    # fig.colorbar(cax, ax=ax)
    #
    # ax = axes[1, 4]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Difference-Hard-(SlotMDP)'],
    #     cmap=plt.cm.RdBu,  # > Diverging colormap
    #     # vmin=-(state_max - state_min) * 0.5, vmax=(state_max - state_min) * 0.5
    # )
    # ax.set_title('Slot-Difference-Hard-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)

    # > Prediction error
    error_keys = [
        'Visualization/Embedding-Error-(FullMDP)',
        'Visualization/Slot-Error-(SlotMDP)',
        'Visualization/Embedding-Error-Hard-(FullMDP)',
        'Visualization/Slot-Error-Hard-(SlotMDP)'
    ]
    error_max = np.max([info[k].max() for k in error_keys])
    # error_min = np.min([info[k].min() for k in error_keys])

    # # ax = axes[0, 5]
    # ax = axes[1, 3]
    # cax = ax.matshow(
    #     info['Visualization/Embedding-Error-(FullMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Embedding-Error-(FullMDP)')
    # fig.colorbar(cax, ax=ax)

    # ax = axes[0, 6]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Error-(SlotMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Slot-Error-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)
    #
    # ax = axes[1, 5]
    # cax = ax.matshow(
    #     info['Visualization/Embedding-Error-Hard-(FullMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Embedding-Error-Hard-(FullMDP)')
    # fig.colorbar(cax, ax=ax)
    #
    # ax = axes[1, 6]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Error-Hard-(SlotMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Slot-Error-Hard-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)
    #
    # # > T1 aligned slot error
    # ax = axes[0, 7]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Error-T1-(SlotMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Slot-Error-T1-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)
    #
    # ax = axes[1, 7]
    # cax = ax.matshow(
    #     info['Visualization/Slot-Error-Hard-T1-(SlotMDP)'],
    #     cmap=plt.cm.Oranges,
    #     # vmin=0, vmax=error_max
    # )
    # ax.set_title('Slot-Error-Hard-T1-(SlotMDP)')
    # fig.colorbar(cax, ax=ax)

    # TODO add recon - hard to plot through multiple subplots
    # cax = axes[2, :].imshow(last_info['Visualization/Reconstruction'])
    # TODO try this - https://matplotlib.org/stable/tutorials/provisional/mosaic.html

    # fig.set_tight_layout(True)  # > Or `plt.tight_layout()`
    # plt.tight_layout()

    info.update({
        'Visualization/BindingVisualization': fig
    })

    return info