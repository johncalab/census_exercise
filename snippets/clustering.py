# k means k-means minibatch mini-batch
from preprocess import load_for_training
path_to_data = 'census-income.data'
path_to_columns = 'census-income.columns'
df, var_cont,var_disc = load_for_training(path_to_data,path_to_columns,make_dummies=False)

w = df['instance weight']
x = df.drop(columns=['instance weight'])

from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def elbow_kmeans(x,w,num_batches=30,cluster_range=20,fig_size_x=8,fig_size_y=8,save_path=None):    

    inertias = []
    for n_clusters in range(1,cluster_range+1):
        model = MiniBatchKMeans(n_clusters=n_clusters,batch_size=batch_size)
        model.fit(x,kmean__sample_weight=w)
        inertias.append(model.named_steps['kmean'].inertia_)
    
    import seaborn as sns

    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")

    hor_axis = list(range(1,cluster_range+1))    
    plt.figure(figsize=(fig_size_x,fig_size_y))
    plt.plot(hor_axis,inertias)
    sns.despine()
    if save_path:
        plt.savefig(save_path)

col_pipe = ColumnTransformer(
       remainder='drop',
        transformers=[
            ('std_scal', StandardScaler() , list(var_cont)),
            ('one_hot', OneHotEncoder(handle_unknown='ignore') , list(var_disc))
        ])

x = col_pipe.fit_transform(df)

n_clusters=3
batch_size=30
elbow_kmeans(x,w)




# hierarchical clustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

model = AgglomerativeClustering()
model.fit(x)

plt.title('H clustering dendro')
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()