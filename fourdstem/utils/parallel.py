"""
fourdstem.utils.parallel
========================
Progress reporting and parallel execution helpers.

Per-file 4D-STEM work (load → mean pattern → RDF) is independent across files,
so it parallelizes cleanly. ``parallel_map`` fans a function over a list across
processes (via joblib's loky backend, which cloudpickles closures/lambdas so the
notebook's ``lambda f: ...`` works) and shows a live progress bar.

Everything degrades gracefully:
* no ``tqdm``  → a lightweight text counter is printed instead
* no ``joblib`` → falls back to sequential execution with a warning
* ``n_jobs=1`` → sequential (with progress), no process pool overhead

Set ``n_jobs=-1`` to use all cores, ``-2`` for all-but-one, or a specific number
(e.g. ``32``).
"""
from __future__ import annotations
import sys
import warnings
import contextlib


def _make_tqdm(total, desc, enable):
    """Return a tqdm instance (notebook-aware) or None."""
    if not enable:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc or "processing")
    except Exception:
        return None


class _TextBar:
    """Minimal fallback progress counter when tqdm is unavailable."""
    def __init__(self, total, desc):
        self.total = total
        self.n = 0
        self.desc = desc or "processing"
        self._emit()

    def update(self, k=1):
        self.n += k
        self._emit()

    def _emit(self):
        sys.stderr.write(f"\r{self.desc}: {self.n}/{self.total}")
        sys.stderr.flush()

    def close(self):
        sys.stderr.write("\n")
        sys.stderr.flush()


@contextlib.contextmanager
def _tqdm_joblib(bar):
    """Route joblib's per-task completion callbacks into a tqdm/text bar."""
    import joblib

    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if bar is not None:
                bar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old


def progress_iter(iterable, total=None, desc="", enable=True):
    """Wrap an iterable with a progress bar (tqdm if available, else text)."""
    items = iterable
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            items = list(iterable)
            total = len(items)
    bar = _make_tqdm(total, desc, enable) or (_TextBar(total, desc) if enable else None)
    for x in items:
        yield x
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()


def parallel_map(func, items, n_jobs=1, progress=True, desc=""):
    """Apply ``func`` to every element of ``items``, optionally in parallel.

    Parameters
    ----------
    func : callable
        ``func(item) -> result``. May be a closure/lambda (cloudpickled by loky).
    items : iterable
    n_jobs : int
        1 = sequential. ``-1`` = all cores, ``-2`` = all but one, or an explicit
        count (e.g. 32). Requires ``joblib`` for n_jobs != 1.
    progress : bool
        Show a progress bar.
    desc : str
        Progress bar label.

    Returns
    -------
    list of results, in input order.
    """
    items = list(items)
    n = len(items)

    if n_jobs in (1, None) or n <= 1:
        return [func(x) for x in progress_iter(items, total=n, desc=desc,
                                               enable=progress)]

    try:
        from joblib import Parallel, delayed
    except Exception:
        warnings.warn("joblib not available — falling back to sequential. "
                      "Install scikit-learn or joblib for parallelism.")
        return [func(x) for x in progress_iter(items, total=n, desc=desc,
                                               enable=progress)]

    bar = _make_tqdm(n, desc, progress) or (_TextBar(n, desc) if progress else None)
    try:
        with _tqdm_joblib(bar):
            out = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(func)(x) for x in items)
    finally:
        if bar is not None:
            bar.close()
    return out
