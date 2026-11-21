"""Model fitting, persistence and prediction (protocol §3.2, stage 'fit').

The EN, EBM and fastai fitting code is copied from model_evaluation/model_evaluation.py
(same split, seeds, preprocessing, hyperparameters and RelMSE arithmetic).  Ablations and OLS
follow §3.2.  Every fitted pipeline exposes ``predict(X)`` on the raw duration matrix in the
original column order (all duration columns, constants included).
"""
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from . import common
from . import config as cfg


def reset_seeds(seed_value):
    """model_evaluation.reset_seeds (torch part only if torch is imported)."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)


def load_hyperparameters(repo, model_id):
    """Read the LHS row as model_evaluation.py does and check it against config.MODELS."""
    spec = cfg.MODELS[model_id]
    if spec["lhs"] is None:
        return {}
    path = os.path.join(repo, cfg.HYPERPARAMETER_DIR, "lhs_%s_parameters.csv" % spec["family"])
    params = pd.read_csv(path, index_col=0)
    hp = params.iloc[[spec["lhs"]]].to_dict(orient="records")[0]
    for k, v in spec["hyperparameters"].items():
        if hp.get(k) != v:
            raise RuntimeError("hyperparameter %s of %s differs from config: %r != %r"
                               % (k, model_id, hp.get(k), v))
    if set(hp) != set(spec["hyperparameters"]):
        raise RuntimeError("hyperparameter columns of %s differ from config" % model_id)
    return hp


def _n_threads():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# ------------------------------------------------------------------ sklearn-type pipelines
class SklearnPipeline:
    def __init__(self, model_id, model, scaler, poly=None, centre=None, keep=None, dcols=None):
        self.model_id = model_id
        self.model = model
        self.scaler = scaler
        self.poly = poly
        self.centre = centre      # training means (EN-36-centred) or None
        self.keep = keep          # boolean column mask used as input (OLS) or None
        self.dcols = list(dcols)

    def _frame(self, X):
        cols = self.dcols if self.keep is None else [c for c, k in zip(self.dcols, self.keep) if k]
        X = np.asarray(X, float)
        if self.keep is not None:
            X = X[:, self.keep]
        return pd.DataFrame(X, columns=cols)

    def transform(self, X):
        Xf = self._frame(X)
        if self.centre is not None:
            Xf = Xf - self.centre
        Xp = self.poly.transform(Xf) if self.poly is not None else Xf
        return self.scaler.transform(Xp)

    def predict(self, X, chunk=None):
        X = np.asarray(X, float)
        chunk = chunk or cfg.EXPLAIN["predict_chunk_rows"]
        if X.shape[0] <= chunk:
            return np.asarray(self.model.predict(self.transform(X)), float)
        return np.concatenate([np.asarray(self.model.predict(self.transform(X[i:i + chunk])), float)
                               for i in range(0, X.shape[0], chunk)])


def fit_sklearn(model_id, hp, df_train, df_test, dcols, const):
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    seed = cfg.SEEDS["training_seed"]
    spec = cfg.MODELS[model_id]
    family = spec["family"]
    hyper = dict(hp)
    hyper.update(spec.get("override", {}))
    n_threads = _n_threads()
    if family in ("ebm", "en"):
        hyper["random_state"] = seed
    if family == "ebm":
        hyper["n_jobs"] = n_threads
    if family == "en":
        hyper["max_iter"] = 50000

    Xtrain = df_train[dcols]
    Ytrain = df_train["actual_duration"]
    Xtest = df_test[dcols]
    Ytest = df_test["actual_duration"]
    reset_seeds(seed)
    start = time.time()
    poly = None
    centre = None
    keep = None
    if model_id == "EN-36":
        # --- copied from model_evaluation.py (mdl_name == "en")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        Xtrain_poly = poly.fit_transform(Xtrain)
        Xtest_poly = poly.transform(Xtest)
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain_poly)
        Xtest_scaled = scaler.transform(Xtest_poly)
        model = ElasticNet(**hyper)
    elif model_id == "EN-36-centred":
        centre = Xtrain.mean()
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        Xtrain_poly = poly.fit_transform(Xtrain - centre)
        Xtest_poly = poly.transform(Xtest - centre)
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain_poly)
        Xtest_scaled = scaler.transform(Xtest_poly)
        model = ElasticNet(**hyper)
    elif model_id in ("EN-36-main", "EBM-24", "EBM-24-noint"):
        # --- copied from model_evaluation.py (else branch, no product terms)
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xtest_scaled = scaler.transform(Xtest)
        model = ElasticNet(**hyper) if family == "en" else ExplainableBoostingRegressor(**hyper)
    elif model_id == "OLS":
        keep = ~np.asarray(const)
        cols = [c for c, k in zip(dcols, keep) if k]
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain[cols])
        Xtest_scaled = scaler.transform(Xtest[cols])
        model = LinearRegression()
    else:
        raise ValueError(model_id)
    model.fit(Xtrain_scaled, Ytrain)
    Ypred = model.predict(Xtest_scaled)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(Xtrain_scaled, Ytrain)
    Ybase = dummy.predict(Xtest_scaled)
    mse_base = mean_squared_error(Ytest, Ybase)
    mse_model = mean_squared_error(Ytest, Ypred)
    rel_mse = mse_model / mse_base if mse_base != 0 else np.inf
    elapsed = time.time() - start
    pipe = SklearnPipeline(model_id, model, scaler, poly=poly, centre=centre, keep=keep, dcols=dcols)
    info = {"RelMSE": rel_mse, "fit_time_s": elapsed, "hyperparameters_used": hyper,
            "test_pred_fit_path": np.asarray(Ypred, float)}
    if family == "ebm":
        info["n_terms"] = len(model.term_features_)
        info["n_pair_terms"] = int(sum(len(f) > 1 for f in model.term_features_))
    elif family in ("en", "ols"):
        coef = np.asarray(model.coef_)
        info["n_terms"] = int((coef != 0).sum())
        if poly is not None:
            info["n_pair_terms"] = int(((poly.powers_.sum(1) == 2) & (coef != 0)).sum())
        else:
            info["n_pair_terms"] = 0
    return pipe, info


def save_sklearn(pipe, path):
    import joblib
    tmp = "%s.tmp.%d" % (path, os.getpid())
    joblib.dump(pipe, tmp)
    os.replace(tmp, path)


def load_sklearn(path):
    import joblib
    return joblib.load(path)


# ------------------------------------------------------------------ fastai
def _fastai_imports():
    import torch
    if sys.platform == "darwin":
        torch.backends.mps.is_available = lambda: False   # force CPU on macOS
    import fastai.torch_core
    if sys.platform == "darwin":
        fastai.torch_core.defaults.use_cuda = False
    import fastai.tabular.all as fta
    return torch, fta


class FastaiPipeline:
    """Wraps a fastai Learner; ``predict`` reproduces dls.test_dl preprocessing with large batches."""

    def __init__(self, learn, dcols, keep_cols):
        self.learn = learn
        self.dcols = list(dcols)
        self.keep_cols = list(keep_cols)
        self.idx = np.array([self.dcols.index(c) for c in self.keep_cols])
        norm = learn.dls.train_ds.procs.normalize
        self.means = np.array([float(norm.means[c]) for c in self.keep_cols])
        self.stds = np.array([float(norm.stds[c]) for c in self.keep_cols])

    def predict_exact(self, df):
        """fastai test_dl/get_preds path (as model_evaluation.py)."""
        test_dl = self.learn.dls.test_dl(df, with_labels=False, shuffle=False)
        preds, _ = self.learn.get_preds(dl=test_dl)
        return preds.numpy().astype(float).ravel()

    def predict(self, X, chunk=100000):
        torch, _ = _fastai_imports()
        X = np.asarray(X, float)[:, self.idx]
        Z = ((X - self.means) / self.stds).astype(np.float32)
        model = self.learn.model
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Z.shape[0], chunk):
                xc = torch.from_numpy(Z[i:i + chunk])
                xcat = torch.zeros((xc.shape[0], 0), dtype=torch.long)
                out.append(model(xcat, xc).numpy().astype(float).ravel())
        return np.concatenate(out) if out else np.zeros(0)


def fit_fastai(model_id, hp, df_train, df_test, dcols):
    torch, fta = _fastai_imports()
    from sklearn.metrics import mean_squared_error
    seed = cfg.SEEDS["training_seed"]
    torch.set_num_threads(_n_threads())
    hyperparameters = dict(hp)
    reset_seeds(seed)
    start_time = time.time()
    # --- copied from model_evaluation.py (mdl_name == "fastai")
    fta.set_seed(seed, reproducible=True)
    const_cols = [c for c in df_train.columns if c.startswith("duration") and df_train[c].nunique() <= 1]
    df_train = df_train.drop(columns=const_cols)
    df_test = df_test.drop(columns=const_cols)
    splits = fta.RandomSplitter(valid_pct=0.1, seed=seed)(fta.range_of(df_train))
    to = fta.TabularPandas(
        df_train,
        procs=[fta.Categorify, fta.FillMissing, fta.Normalize],
        cat_names=[],
        cont_names=[col for col in df_train.columns if col.startswith("duration")],
        y_names="actual_duration",
        splits=splits,
    )
    dls = to.dataloaders(bs=64, num_workers=0)
    layers = [int((hyperparameters["layer1"] // 5) * 5), int((hyperparameters["layer2"] // 5) * 5)]
    wd = float(hyperparameters["wd"])
    lr_max = float(hyperparameters["lr_max"])
    learn = fta.tabular_learner(dls, layers=layers, metrics=fta.mse, config=fta.tabular_config(), wd=wd,
                                loss_func=fta.MSELossFlat(),
                                cbs=[fta.EarlyStoppingCallback(monitor="valid_loss", patience=10)])
    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(100, lr_max=lr_max)
        test_dl = dls.test_dl(df_test, with_labels=True, shuffle=False)
        test_preds, targs = learn.get_preds(dl=test_dl)
    mse_model = fta.mse(test_preds, targs).item()
    Ybase = np.repeat(df_train["actual_duration"].mean(), df_test.shape[0])
    mse_base = mean_squared_error(df_test["actual_duration"].values, Ybase)
    rel_mse = mse_model / mse_base if mse_base != 0 else np.inf
    elapsed = time.time() - start_time
    keep_cols = [c for c in dcols if c not in const_cols]
    pipe = FastaiPipeline(learn, dcols, keep_cols)
    n_epochs = len(learn.recorder.values)
    info = {"RelMSE": rel_mse, "fit_time_s": elapsed, "hyperparameters_used": dict(hp, layers=layers),
            "test_pred_fit_path": test_preds.numpy().astype(float).ravel(),
            "n_constant_dropped": len(const_cols), "epochs_run": n_epochs,
            "n_terms": None, "n_pair_terms": None}
    return pipe, info


def save_fastai(pipe, path):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    pipe.learn.export(os.path.abspath(tmp))
    os.replace(tmp, path)


def load_fastai(path, dcols, keep_cols):
    _, fta = _fastai_imports()
    learn = fta.load_learner(path, cpu=True)
    return FastaiPipeline(learn, dcols, keep_cols)


# ------------------------------------------------------------------ dispatch
def model_filename(model_id):
    return "model.pkl" if cfg.MODELS[model_id]["family"] == "fastai" else "model.joblib"


def fit(repo, model_id, df_train, df_test, dcols, const):
    hp = load_hyperparameters(repo, model_id)
    if cfg.MODELS[model_id]["family"] == "fastai":
        return fit_fastai(model_id, hp, df_train, df_test, dcols)
    return fit_sklearn(model_id, hp, df_train, df_test, dcols, const)


def save(pipe, model_id, path):
    if cfg.MODELS[model_id]["family"] == "fastai":
        save_fastai(pipe, path)
    else:
        save_sklearn(pipe, path)


def load(model_dir, model_id):
    import json
    with open(os.path.join(model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    path = os.path.join(model_dir, model_filename(model_id))
    if cfg.MODELS[model_id]["family"] == "fastai":
        return load_fastai(path, meta["column_order"], meta["input_columns"]), meta
    return load_sklearn(path), meta
