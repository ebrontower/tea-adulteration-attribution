"""Self-contained readers for the four QIB/Quadram open FTIR corpora.

Only needs the public-deposition folder resolved by p5_paths (see data/README.md).
Handles the two file dialects seen in the deposit:
  - spectra-as-columns with a header row of labels (meats, purees)
  - transposed matrix with three metadata rows: Sample Number / Group Code / label (olive, coffee)

Yields tidy frames: id, label, session, extra fields, plus X (n_samples x n_channels)
and wavenumber axis. Values are taken as-is; the absorbance-vs-ratio check is done
separately by `axis_stats` so the caller can decide (cinnamon lesson: A = -log10 T).
"""
import re
import numpy as np
import pandas as pd
import p5_paths  # noqa: E402

QIB_ROOT = p5_paths.PUBLIC_DATA + "/qib"
NUM = re.compile(r"^-?\d+(\.\d+)?(e-?\d+)?$", re.I)


def _is_num(v):
    return isinstance(v, (int, float, np.floating)) or (isinstance(v, str) and NUM.match(v.strip()))


def load_columns_dialect(path, label_row_name=None):
    """First column = wavenumbers, remaining columns = spectra named by header."""
    df = pd.read_csv(path)
    axis_name = df.columns[0]
    ycol = df[axis_name].astype(str).str.strip()
    keep = np.array([bool(NUM.match(v)) for v in ycol])
    axis = ycol[keep].astype(float).to_numpy()
    X = df.loc[keep, df.columns[1:]].astype(float).to_numpy().T
    names = [str(c).strip() for c in df.columns[1:]]
    return axis, X, names


def load_transposed_dialect(path):
    """Rows 0..1 = metadata (sample number, group code), row 2 = label names, rows 3+ = spectra."""
    raw = pd.read_csv(path, header=None)
    sample_no = raw.iloc[0, 1:].tolist()
    group_code = raw.iloc[1, 1:].tolist()
    label = raw.iloc[2, 1:].tolist()
    body = raw.iloc[3:, :]
    ycol = body.iloc[:, 0].astype(str).str.strip().to_numpy()
    keep = np.array([bool(NUM.match(v)) for v in ycol])
    axis = ycol[keep].astype(float)
    X = body.loc[keep, body.columns[1:]].astype(float).to_numpy().T
    n = X.shape[0]
    return axis, X, [dict(sample=str(sample_no[i]).strip(), group=str(group_code[i]).strip(),
                          label=str(label[i]).strip()) for i in range(n)]


def axis_stats(X, axis, lo=None, hi=None):
    m = np.ones(len(axis), bool)
    if lo is not None:
        m &= axis >= lo
    if hi is not None:
        m &= axis <= hi
    sub = X[:, m]
    med = np.median(sub, axis=0)
    return dict(n_channels=int(m.sum()), first=float(axis[m][0]), last=float(axis[m][-1]),
                vmin=float(sub.min()), vmax=float(sub.max()),
                axis_median_min=float(med.min()), axis_median_max=float(med.max()),
                step=float(np.median(np.diff(axis[m]))))


def meats():
    axis, X, names = load_columns_dialect(f"{QIB_ROOT}/MIRFreshMeats.csv")
    rec = []
    for nm in names:
        p = nm.split("_")
        rec.append(dict(meat=p[0].replace("Fresh", ""), supplier=p[1], sample=p[2], run=p[3]))
    df = pd.DataFrame(rec)
    df["phys"] = df["meat"] + "|" + df["supplier"] + "|" + df["sample"]
    df["id"] = df["phys"] + "|" + df["run"]
    df["label"] = df["meat"]
    return axis, X, df


def olive():
    axis, X, metas = load_transposed_dialect(f"{QIB_ROOT}/FTIR_Spectra_olive_oils.csv")
    df = pd.DataFrame(metas)
    df["rep"] = df.groupby("sample").cumcount()
    df["phys"] = df["sample"]
    df["id"] = df["sample"] + "_r" + df["rep"].astype(str)
    return axis, X, df


def coffee():
    axis, X, metas = load_transposed_dialect(f"{QIB_ROOT}/FTIR_Spectra_instant_coffee.csv")
    df = pd.DataFrame(metas)
    df["phys"] = df["sample"]
    return axis, X, df


def purees():
    axis, X, names = load_columns_dialect(f"{QIB_ROOT}/MIR_Fruit_purees.csv")
    base = [re.sub(r"\.\d+$", "", str(n).strip()) for n in names]
    df = pd.DataFrame(dict(label=base))
    df["id"] = ["s%03d" % i for i in range(len(df))]
    return axis, X, df


# ---------------------------------------------------------------- preprocessing

def window(axis, X, lo, hi):
    m = (axis >= lo) & (axis <= hi)
    return axis[m], X[:, m]


def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)


def to_absorbance(X, unit):
    """unit='pctT' -> A = -log10(T/100); 'abs' or 'ratio' pass through (ratio already checked)."""
    if unit == "pctT":
        return -np.log10(np.clip(X, 1e-6, None) / 100.0)
    return X


def msc(X, reference=None):
    """Multiplicative scatter correction; reference defaults to the column mean of X."""
    ref = X.mean(axis=0) if reference is None else np.asarray(reference)
    G = np.column_stack([ref, np.ones(len(ref))])
    coef, *_ = np.linalg.lstsq(G, X.T, rcond=None)          # 2 x n
    a, b = coef[0][None, :], coef[1][None, :]
    corrected = (X.T - b) / (a + 1e-12) * ref[:, None]
    return corrected.T


def sg_derivative(X, axis=None, win=15, order=1):
    """Row-wise Savitzky-Golay first derivative on a (n, N) matrix."""
    from scipy.signal import savgol_filter
    return savgol_filter(X, window_length=win, polyorder=min(order + 1, win - 1), deriv=order, axis=1)


def coral(target, source):
    """Unsupervised covariance alignment of `target` onto `source` (no labels used)."""
    ct = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6
    cs = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    # symmetric square roots via eigendecomposition (stable for near-singular MIR covariances)
    def isqrt(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-10, None)
        return (V * w ** -0.5) @ V.T

    def sqrtm(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        return (V * np.sqrt(w)) @ V.T

    return (target - target.mean(axis=0)) @ isqrt(ct) @ sqrtm(cs) + source.mean(axis=0)
