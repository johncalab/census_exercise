def plot_explained_variance(data,fig_x=10,fig_y=10,save_path=None):
    """
    assumes 'instance weight' and 'label' aren't present
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")

    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA().fit(data)
    plt.figure(figsize=(fig_x,fig_y))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    sns.despine()

    if save_path:
        plt.savefig(save_path)


def pca_visual(data,y,fig_x=10,fig_y=10,save_path=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")

    plt.figure(figsize=(fig_x,fig_y))
#     plt.scatter(projected[:, 0], projected[:, 1],c=y)
    sns.scatterplot(projected[:, 0], projected[:, 1],hue=y, legend=False)
    plt.xlabel('component 1')
    plt.ylabel('component 2')

    sns.despine()

    if save_path:
        plt.savefig(save_path)
