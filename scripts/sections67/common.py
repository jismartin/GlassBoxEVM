"""Seeds, atomic I/O, tables, stamps and environment helpers (protocol §9.2, §9.3)."""
import datetime
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfg

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ seeds
def derive_seed(network_id, stage, replicate=0, root=None):
    """SHA-256 of 'root|network_id|stage|replicate'; first 8 bytes little-endian mod 2**32."""
    if root is None:
        root = cfg.SEEDS["analysis_root_seed"]
    if stage not in cfg.SEEDS["stages"]:
        raise ValueError("unknown seed stage %r" % stage)
    text = "%s|%s|%s|%s" % (root, network_id, stage, replicate)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def stage_rng(network_id, stage, replicate=0):
    return np.random.default_rng(derive_seed(network_id, stage, replicate))


# ------------------------------------------------------------------ hashing
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash():
    """Hash of the package source (tests excluded) and of scripts/tools.py."""
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(PKG_DIR, "*.py")))
    tools_py = os.path.join(os.path.dirname(PKG_DIR), "tools.py")
    if os.path.exists(tools_py):
        files.append(tools_py)
    for f in files:
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def git_commit(repo):
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20)
        commit = out.stdout.strip() or "unknown"
        st = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "scripts/sections67",
                             "scripts/tools.py"], capture_output=True, text=True, timeout=20)
        if st.stdout.strip():
            commit += "+dirty"
        return commit
    except Exception:  # noqa: BLE001 - git may be missing on compute nodes
        return "unknown"


def environment_info():
    info = {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "sklearn", "interpret", "torch", "fastai",
                 "networkx", "joblib"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except Exception:  # noqa: BLE001
                info[name] = "not installed"
                continue
        info[name] = getattr(mod, "__version__", "unknown")
    info["parquet"] = "pyarrow" if has_parquet() else "unavailable (csv.gz used)"
    for v in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK"):
        info[v] = os.environ.get(v, "-")
    return info


def peak_memory_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0
    except Exception:  # noqa: BLE001
        return float("nan")


# ------------------------------------------------------------------ atomic I/O
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_text(path, text):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_write_csv(df, path, **kw):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = "%s.tmp.%d" % (path, os.getpid())
    comp = "gzip" if path.endswith(".gz") else None
    df.to_csv(tmp, index=False, compression=comp, **kw)
    os.replace(tmp, path)


def atomic_save_npz(path, **arrays):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = "%s.tmp.%d.npz" % (path, os.getpid())
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def write_table(df, base):
    """Write a detail table as Parquet if pyarrow exists, else as gzip CSV. Returns path."""
    if has_parquet():
        path = base + ".parquet"
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        tmp = "%s.tmp.%d" % (path, os.getpid())
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    else:
        path = base + ".csv.gz"
        atomic_write_csv(df, path)
    return path


def read_table(base):
    if os.path.exists(base + ".parquet"):
        return pd.read_parquet(base + ".parquet")
    if os.path.exists(base + ".csv.gz"):
        return pd.read_csv(base + ".csv.gz")
    if os.path.exists(base + ".csv"):
        return pd.read_csv(base + ".csv")
    raise FileNotFoundError(base)


def table_exists(base):
    return any(os.path.exists(base + e) for e in (".parquet", ".csv.gz", ".csv"))


# ------------------------------------------------------------------ stamps / resumability
class Stamp:
    """A stage output directory is complete when ``_stamp.json`` exists and its hashes match.

    Outputs are written atomically first and the stamp last.  If an existing stamp does not
    match (different inputs, configuration or code), the old directory is moved aside to
    ``<dir>.superseded-<UTC time>`` instead of being overwritten silently.
    """

    def __init__(self, outdir, inputs, extra=None):
        self.outdir = outdir
        self.path = os.path.join(outdir, "_stamp.json")
        self.key = {"config_hash": cfg.config_hash(), "code_hash": code_hash(),
                    "inputs": inputs, "extra": extra or {}}
        self.t0 = None

    def is_current(self):
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                old = json.load(f)
        except Exception:  # noqa: BLE001
            return False
        return old.get("key") == json.loads(json.dumps(self.key, default=_json_default))

    def begin(self, force=False):
        """Return False if the stage can be skipped."""
        if self.is_current() and not force:
            return False
        if os.path.exists(self.outdir) and os.listdir(self.outdir):
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            shutil.move(self.outdir, "%s.superseded-%s" % (self.outdir.rstrip("/"), ts))
        ensure_dir(self.outdir)
        self.t0 = time.time()
        return True

    def finish(self, status="ok", info=None):
        rec = {"key": self.key, "status": status, "runtime_s": time.time() - (self.t0 or time.time()),
               "peak_memory_mb": peak_memory_mb(),
               "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
               "info": info or {}}
        atomic_write_json(self.path, rec)
        return rec


def utcnow():
    return datetime.datetime.utcnow().isoformat() + "Z"
