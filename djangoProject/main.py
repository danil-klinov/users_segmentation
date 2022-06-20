import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from kneed import KneeLocator
import statsmodels.api as sm
from statsmodels.formula.api import ols

from djangoProject.settings import CLUSTERING_FILES_PATH


def preprocess(df):
    scaler = StandardScaler()
    scaler.fit(df)
    norm = scaler.transform(df)
    return norm


def elbow_plot(df):
    norm = preprocess(df)

    sse = {}

    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_

    plt.title('Elbow plot for K selection')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()


def find_k(norm, increment=0, decrement=0):
    sse = {}

    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_

    kn = KneeLocator(
        x=list(sse.keys()),
        y=list(sse.values()),
        curve='convex',
        direction='decreasing'
    )
    k = kn.knee + increment - decrement
    return k


def run_kmeans(df, increment=0, decrement=0):
    norm = preprocess(df)
    k = find_k(norm, increment, decrement)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(norm)
    return df.assign(cluster=kmeans.labels_)


def run_mini_batch_kmeans(df, increment=0, decrement=0):
    norm = preprocess(df)
    k = find_k(norm, increment, decrement)
    mini_batch_kmeans = MiniBatchKMeans(n_clusters=k)
    mini_batch_kmeans.fit(norm)
    return df.assign(cluster=mini_batch_kmeans.labels_)


def run_agglomerative_clustering(df, increment=0, decrement=0):
    norm = preprocess(df)
    k = find_k(norm, increment, decrement)
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(norm)
    return df.assign(cluster=model.labels_)


def run_spectral_clustering(df, increment=0, decrement=0):
    norm = preprocess(df)
    k = find_k(norm, increment, decrement)
    model = SpectralClustering(n_clusters=k)
    model.fit(norm)
    return df.assign(cluster=model.labels_)


def draw_3d(dataframe, name):
    fig = px.scatter_3d(dataframe, x='recency', y='frequency', z='monetary', color='cluster')
    fig.write_image('images/' + name + '.png')


def run(file, cols):
    df = pd.read_csv(file, usecols=cols)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['LinePrice'] = df['Price'] * df['Quantity']
    df = df[df['LinePrice'] > 0]

    end_date = max(df['InvoiceDate']) + dt.timedelta(days=1)
    df_rfm = df.groupby('Customer ID').agg(
        recency=('InvoiceDate', lambda x: (end_date - x.max()).days),
        frequency=('Invoice', 'count'),
        monetary=('LinePrice', 'sum')
    ).reset_index()

    algorithms = {
        'mini_batch_kmeans': run_mini_batch_kmeans,
        'kmeans': run_kmeans,
        'agglomerative_clustering': run_agglomerative_clustering,
        'spectral_clustering': run_spectral_clustering
    }
    df_algorithms = {}
    max_anova = 0
    max_anova_algorithm = ''
    for algorithm_name in algorithms:
        df_algorithm = algorithms.get(algorithm_name)(df_rfm, decrement=2)
        df_algorithms[algorithm_name] = df_algorithm
        draw_3d(df_algorithm, algorithm_name)
        merged_df = pd.merge(df_algorithm, df, on='Customer ID')
        model = ols(formula='Price ~ C(cluster)', data=merged_df.head(1000)).fit()
        print(model.summary())
        print('----------------------')
        anova = sm.stats.anova_lm(model, typ=1)
        anova.to_excel(CLUSTERING_FILES_PATH + 'anova/' + algorithm_name + '.xlsx')
        anova_f = anova.iloc[0]['F']
        print(anova_f)
        max_anova = max(max_anova, anova_f)
        if max_anova == anova_f:
            max_anova_algorithm = algorithm_name

    if max_anova_algorithm != '':
        df_algorithms.get(max_anova_algorithm).to_excel(
            CLUSTERING_FILES_PATH + 'segmentation_' + max_anova_algorithm + '.xlsx'
        )
