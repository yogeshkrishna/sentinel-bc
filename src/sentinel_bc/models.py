"""Small tabular models; all weights start from scratch."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import hashlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .preprocessing import transformer


def classical_candidates(seed: int, mode: str):
    grid = [0.3, 3.0] if mode == "full" else [1.0]
    for c in grid:
        yield "logistic", {"C": c}, make_pipeline(transformer(), LogisticRegression(C=c, max_iter=3000, random_state=seed))
        yield "rbf_svm", {"C": c}, make_pipeline(transformer(), SVC(C=c, kernel="rbf", probability=True, random_state=seed))
    if mode == "full":
        for depth in [3, None]:
            yield "random_forest", {"max_depth": depth}, make_pipeline(transformer(), RandomForestClassifier(
                n_estimators=200, max_depth=depth, min_samples_leaf=2, random_state=seed, n_jobs=2))
        for depth in [2, 3]:
            yield "xgboost", {"max_depth": depth}, make_pipeline(transformer(), XGBClassifier(
                n_estimators=150, max_depth=depth, learning_rate=.05, subsample=.85,
                colsample_bytree=.85, reg_lambda=2, eval_metric="logloss", random_state=seed, n_jobs=2))


def tensorflow():
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def train_network(x, y, seed: int, epochs: int, patience: int, epoch_callback=None):
    from sklearn.model_selection import train_test_split
    tf = tensorflow()
    tf.keras.utils.set_random_seed(seed)
    a, b = train_test_split(np.arange(len(y)), test_size=.18, stratify=y, random_state=seed)
    prep = transformer().fit(x.iloc[a])
    z = prep.transform(x).astype("float32")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(30,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(.20), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="binary_crossentropy", metrics=["accuracy"])
    # Explicit datasets bound thread pools even on small Windows laptops.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train = tf.data.Dataset.from_tensor_slices((z[a], y[a].astype("float32"))).shuffle(len(a), seed=seed).batch(32).with_options(options)
    valid = tf.data.Dataset.from_tensor_slices((z[b], y[b].astype("float32"))).batch(64).with_options(options)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if epoch_callback:
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=epoch_callback))
    history = model.fit(train, validation_data=valid, epochs=epochs, verbose=0, shuffle=False, callbacks=callbacks)
    return prep, model, history.history


class NeuralPredictor(ClassifierMixin, BaseEstimator):
    """Fitted sklearn-compatible adapter with explicit canonical class order."""
    def __init__(self, members=None):
        self.members = members
        if members:
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = 30

    def fit(self, x, y):
        raise RuntimeError("Use train_network; this adapter represents already fitted members")

    def predict_proba(self, x):
        p = self.member_probabilities(x).mean(axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= .5).astype(int)

    def member_probabilities(self, x):
        return np.asarray([np.asarray(m(p.transform(x).astype("float32"), training=False)).reshape(-1) for p, m in self.members])

    def mc_probabilities(self, x, passes=20):
        """Repeatable per-case Dropout draws, with every other layer in inference mode.

        Stable seeds make a row's uncertainty invariant to batching and row order.
        No BatchNorm is trained, but it would remain frozen in this implementation.
        """
        tf = tensorflow()
        prep, model = self.members[0]
        z = prep.transform(x).astype("float32")
        values = []
        for row in z:
            seed = int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:4], "little") % (2**31 - 1)
            out = tf.convert_to_tensor(np.tile(row, (passes, 1)))
            for i, layer in enumerate(model.layers):
                if isinstance(layer, tf.keras.layers.Dropout):
                    out = tf.nn.experimental.stateless_dropout(out, rate=layer.rate, seed=[seed, i])
                else:
                    out = layer(out, training=False)
            values.append(np.asarray(out).reshape(-1))
        return np.asarray(values).T

