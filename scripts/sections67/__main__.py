"""Command-line entry point; limits BLAS/OpenMP threads before numpy is imported."""
import os
import sys

_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
for _a, _b in zip(sys.argv, sys.argv[1:]):
    if _a == "--threads":
        _threads = _b
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_threads)
os.environ.setdefault("SLURM_CPUS_PER_TASK", str(_threads))

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
