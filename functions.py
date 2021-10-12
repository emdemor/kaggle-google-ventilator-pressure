import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold


def get_slope(y):

    x = np.ones(y.shape).cumsum()

    res = stats.linregress(x, y)

    return res[0]


def pred_pressure(x, temp):

    p0 = temp.iloc[0]["pressure"]

    p1 = temp[(temp["time_step"] > 0.85) & (temp["time_step"] < 0.98)].mean()[
        "pressure"
    ]

    p2 = temp["pressure"].iloc[-1]

    t_ref = x[3]

    kp = x[0] / 10
    ki = x[1] / 100
    kd = x[2] / 1000

    # c = x[3] * 100

    err = p0 - p1
    int_ = 0
    de = (temp["pressure"].iloc[2] - temp["pressure"].iloc[0]) / temp["dt"].iloc[1]

    error = [err]

    # target = p1 - (p1 - p2) / (1 + np.exp(-c * (temp["time_step"] - t_ref)))

    target = np.where(
        temp["time_step"] < 1,
        p1,
        np.where(
            temp["time_step"] < t_ref,
            (p2 - t_ref * p1 + (p1 - p2) * temp["time_step"]) / (1 - t_ref),
            p2,
        ),
    )

    for i in range(1, len(temp)):

        dt = temp["dt"].iloc[i]
        int_ = int_ + err * dt
        dif_ = de / dt

        de = kp * err + ki * int_ + kd * dif_ - err

        err = err + de

        error.append(err)

    pred = target + error

    return pred


def sum_squad(x, temp):

    pred = pred_pressure(x, temp)

    return ((pred - temp["pressure"]) ** 2).sum()


# sns.lineplot(data = temp,x='time_step', y = 'pred')
# sns.lineplot(data = temp,x='time_step', y = 'pressure')

# plt.show()


def eval_u_in_out_ks(df_model):
    selected_breaths = df_model["breath_id"].unique()
    temp = df_model[df_model["breath_id"].isin(selected_breaths)]
    n_breaths = len(selected_breaths)
    a = temp["u_in"].values.reshape(n_breaths, 80)
    b = temp["u_out"].values.reshape(n_breaths, 80)

    results = pd.DataFrame(
        {
            "breath_id": selected_breaths,
            "ks_u": [ks_2samp(a[i], b[i])[0] for i in range(n_breaths)],
        }
    )

    return results


def generate_folds(X, n_folds=5, shuffle=True, random_state=42):

    X = X.copy()

    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    folds = list(kf.split(X))

    for fold_index in range(n_folds):

        train_index, validation_index = folds[fold_index]

        X.loc[X[X.index.isin(validation_index)].index, "fold"] = fold_index

    return X["fold"].astype(int)


def get_breath_type(df, scaler, clusterer):

    grouped = df.groupby("breath_id")

    breaths = (
        grouped.mean()
        .drop(columns=["id", "time_step", "pressure"], errors="ignore")
        .reset_index()
    )

    breaths = breaths.drop(columns="cluster", errors="ignore")

    breaths_scaled = scaler.transform(breaths)

    labels = clusterer.predict(breaths_scaled)

    return labels


def preprocessing(df_model):
    def log_exp_return(series):
        return np.exp(np.log1p(series).diff(1).fillna(0))

    # ---------------------------
    mean_u_in = (
        df_model.groupby("breath_id").agg(u_in_mean=("u_in", "mean")).reset_index()
    )

    # -----------------------------
    ks_u = eval_u_in_out_ks(df_model)

    teste = (
        df_model.merge(mean_u_in, on="breath_id", how="left").merge(
            ks_u, on="breath_id", how="left"
        )
        # .merge(slope,on="breath_id", how="left")
        # .merge(pressure_init,on="breath_id", how="left")
    )

    # time diff
    teste["time_diff"] = (
        teste["time_step"].groupby(teste["breath_id"]).diff(1).fillna(0)
    )

    # u_in parameter
    teste["u_in_ratio"] = (
        teste["u_in"].groupby(teste["breath_id"]).apply(log_exp_return)
    )
    teste["last_value_u_in"] = (
        teste["u_in"].groupby(teste["breath_id"]).transform("last")
    )
    teste["first_value_u_in"] = (
        teste["u_in"].groupby(teste["breath_id"]).transform("first")
    )

    # u_in area
    teste["area"] = teste["time_step"] * teste["u_in"]
    teste["area"] = teste.groupby("breath_id")["area"].cumsum()
    teste["u_in_cumsum"] = (teste["u_in"]).groupby(teste["breath_id"]).cumsum()

    # u_in shift change
    for i in np.arange(1, 5, 1):
        teste[f"u_in_lag_fwrd{i}"] = (
            teste["u_in"].groupby(teste["breath_id"]).shift(i).fillna(0)
        )
        teste[f"u_in_lag_back{i}"] = (
            teste["u_in"].groupby(teste["breath_id"]).shift(int(-i)).fillna(0)
        )

    # R, C parameter
    teste["RC"] = teste["C"] * teste["R"]
    teste["R/C"] = teste["R"] / teste["C"]
    teste["C/R"] = teste["C"] / teste["R"]
    #teste["R"] = teste["R"].astype("category")
    #teste["C"] = teste["C"].astype("category")
    #teste["RC"] = teste["RC"].astype("category")
    #teste["R/C"] = teste["R/C"].astype("category")
    #teste["C/R"] = teste["C/R"].astype("category")


    # for col in teste.dtypes[teste.dtypes == "category"].index:
    #     teste = pd.concat([teste.drop(columns=col,errors='ignore'),pd.get_dummies(teste[col], prefix=col)], axis=1)
        
    return teste


class BreathClusterer:
    def __init__(
        self,
        scaler=StandardScaler(),
        n_clusters_list=range(4, 10),
        verbose=1,
        **kwargs,
    ):
        self.scaler = scaler
        self.n_clusters_list = n_clusters_list
        self.verbose = verbose
        self.kmean_parameters = kwargs
        self.best_score = -1
        self.clusterer = KMeans(**kwargs)

    def preprocess(self, X, y=None):

        grouped = X.groupby("breath_id")

        breaths = (
            grouped.mean()
            .drop(columns=["id", "time_step", "pressure"], errors="ignore")
            .reset_index()
        )

        breaths = breaths.drop(columns="cluster", errors="ignore")

        return breaths

    def transform(self, X, y=None):

        breaths = self.preprocess(X)

        breaths_scaled = self.scaler.transform(breaths)

        return breaths_scaled

    def predict(self, X, y=None):

        X = X.copy()

        breaths = self.preprocess(X)

        breaths_scaled = self.scaler.transform(breaths)

        labels_prediction = self.clusterer.predict(breaths_scaled)

        breaths["breath_type"] = labels_prediction

        return breaths[["breath_id", "breath_type"]]

    def fit(self, X, y=None):

        breaths = self.preprocess(X)

        self.scaler.fit(breaths)

        breaths_scaled = self.scaler.transform(breaths)

        n_cluster_scores = []

        for n_clusters in self.n_clusters_list:
            pars = self.clusterer.get_params()
            pars.update(dict(n_clusters=n_clusters))
            temp_clusterer = self.clusterer.__class__(**pars)
            temp_clusterer.fit(breaths_scaled)

            silhouette = silhouette_score(breaths_scaled, temp_clusterer.labels_)

            if silhouette > self.best_score:
                self.best_score = silhouette
                self.clusterer = temp_clusterer

            n_cluster_scores.append({"n_cluster": n_clusters, "score": silhouette})

            if self.verbose > 0:
                print(
                    "n_cluster: {} | silhouette: {:.3f}".format(n_clusters, silhouette)
                )

        return self
