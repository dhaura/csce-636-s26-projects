#!/usr/bin/env python3
"""Generate exact-LP m-height labels with resumable, atomic checkpoints.

Requires numpy and scipy. Run --help; see GENERATION.md for Perlmutter usage.
HiGHS solves on CPUs. Floating-point LP optima are numerical approximations.
"""
from __future__ import annotations

import os

# Set before importing NumPy/SciPy, including in spawned worker processes.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import hashlib
from itertools import combinations
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import signal
import sys
import time
import warnings

import numpy as np
import scipy
from scipy.optimize import linprog, OptimizeWarning

N = 9
DEFAULT_NEW_SAMPLES = 3_833_203
FEATURES = "CSCE-636-Project-1-Train-n_k_m_P"
LABELS = "CSCE-636-Project-1-Train-mHeights"
EXT_FEATURES = "CSCE-636-Project-1-Train-Extended-n_k_m_P"
EXT_LABELS = "CSCE-636-Project-1-Train-Extended-mHeights"
FORMAT_VERSION = 2
K_VALUES = (4, 5, 6)
MAX_ENTRY = 100
STOP_REQUESTED = False
# Precompute the combinatorial structure once in each process.
SUBSETS = {
    m: [(s, tuple(t for t in range(N) if t not in s))
        for s in combinations(range(N), m)]
    for m in range(2, 6)
}
# SciPy forwards these native HiGHS options, with an expected warning.
warnings.filterwarnings("ignore", message="Unrecognized options detected:.*",
                        category=OptimizeWarning)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def compute_m_height(G, m):
    """Solve max G[:, j] @ u, with |G[:, t] @ u| <= 1 off S.

    u is free in sign (linprog's default nonnegative bounds are incorrect).
    The symmetric feasible region already covers both objective signs.
    An unbounded LP means the mathematical m-height is +infinity.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[1] != N or not np.isfinite(G).all():
        raise ValueError("G must be a finite k-by-9 matrix")
    k = G.shape[0]
    if k not in (4, 5, 6) or not 2 <= m <= N - k:
        raise ValueError("Invalid k or m")
    best = 0.0
    for selected, complement in SUBSETS[m]:
        rows = G[:, complement].T
        A_ub = np.concatenate((rows, -rows), axis=0)
        b_ub = np.ones(2 * len(complement))
        for j in selected:
            # Retry a failed solve without presolve. Never use a failed optimum.
            for presolve in (True, False):
                result = linprog(
                    -G[:, j], A_ub=A_ub, b_ub=b_ub,
                    bounds=[(None, None)] * k, method="highs-ds",
                    options={"threads": 1, "parallel": False,
                             "presolve": presolve, "time_limit": 30.0,
                             "primal_feasibility_tolerance": 1e-8,
                             "dual_feasibility_tolerance": 1e-8},
                )
                if result.status == 3:
                    # Confirm apparent unboundedness with presolve disabled.
                    if presolve:
                        continue
                    return float("inf")
                if result.status == 0:
                    value = float(-result.fun)
                    if (np.isfinite(value) and
                            np.max(A_ub @ result.x - b_ub) <= 1e-6):
                        best = max(best, value)
                        break
            else:
                raise RuntimeError(
                    f"LP failed: m={m}, S={selected}, j={j}, "
                    f"status={result.status}, message={result.message}"
                )
    return best


def matrix_k(matrix_id):
    """Cycle k=4,5,6 so every k gets the same number of sampled matrices."""
    return K_VALUES[matrix_id % 3]


def empirical_scales(features):
    """Count, per k, the per-matrix max|P| of the original data.

    Every original matrix has an m=2 row, so those rows enumerate the matrices.
    The original entries are integers in [-100, 100]; the scale is therefore an
    integer, and each new matrix draws one from this histogram.
    """
    counts = {k: [0] * (MAX_ENTRY + 1) for k in K_VALUES}
    for n, k, m, P in features:
        if m != 2:
            continue
        scale = float(np.max(np.abs(P)))
        if n != N or k not in counts or scale != round(scale) or not 0 <= scale <= MAX_ENTRY:
            raise ValueError("Original data must contain integer entries in [-100, 100]")
        counts[k][int(scale)] += 1
    if any(sum(row[1:]) == 0 for row in counts.values()):
        raise ValueError("Original data lacks nonzero matrices for some k")
    return {str(k): row for k, row in counts.items()}


def sample_matrix(rng, k, distribution, scale_counts):
    """Draw P, resampling if a column of G would be all zero (excluded by the spec)."""
    while True:
        if distribution == "empirical":
            weights = np.asarray(scale_counts, dtype=np.float64)
            weights[0] = 0.0  # a zero scale gives all-zero columns
            scale = int(rng.choice(MAX_ENTRY + 1, p=weights / weights.sum()))
            P = rng.integers(-scale, scale + 1, size=(k, N - k)).astype(np.float64)
        elif distribution == "integer":
            P = rng.integers(-MAX_ENTRY, MAX_ENTRY + 1, size=(k, N - k)).astype(np.float64)
        else:
            P = rng.uniform(-MAX_ENTRY, MAX_ENTRY, size=(k, N - k))
        if np.all(np.any(P != 0, axis=0)):
            return P


def generate_matrix(task):
    """Label m=2,3,... until the first infinite m-height.

    m-heights are nondecreasing in m, so every larger m is infinite as well.
    Like the original data and the test set, only finite rows are kept.
    A matrix whose 2-height is infinite therefore contributes no rows.
    """
    matrix_id, k, seed, distribution, scale_counts = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, matrix_id]))
    P = sample_matrix(rng, k, distribution, scale_counts)
    G = np.concatenate((np.eye(k), P), axis=1)
    features, labels = [], []
    for m in range(2, N - k + 1):
        try:
            height = compute_m_height(G, m)
        except Exception as exc:
            raise RuntimeError(f"Matrix {matrix_id}, k={k}, m={m}: {exc}") from exc
        if math.isinf(height):
            break
        features.append([N, k, m, P])
        labels.append(height)
    return matrix_id, features, labels


def fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, value, *, as_json=False):
    """A rename commits the complete file; interrupted .tmp files are ignored."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as file:
        if as_json:
            file.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        else:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_directory(path):
    path = path.expanduser().resolve()
    # Resolve symlinks so a link to HOME cannot accidentally pass this check.
    roots = [Path("/pscratch"), Path("/global/cfs")]
    roots += [Path(os.environ[name]).expanduser().resolve()
              for name in ("SCRATCH", "CFS") if os.environ.get(name)]
    home = Path.home().resolve()
    if path == home or home in path.parents:
        raise ValueError("Output must not be in HOME; choose SCRATCH or CFS")
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("Output must be inside $SCRATCH, $CFS, /pscratch, or /global/cfs")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def output_lock(directory):
    # Use a filesystem supporting flock (Perlmutter scratch does).
    with (directory / ".generation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another generator is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_config(args):
    inputs = []
    for name in (FEATURES, LABELS):
        path = (args.input_dir / name).resolve()
        log(f"Fingerprinting input {path}")
        inputs.append({"path": str(path), "sha256": sha256(path)})
    scale_counts = None
    if args.distribution == "empirical":
        log("Measuring per-matrix scales of the original data")
        with (args.input_dir / FEATURES).open("rb") as file:
            scale_counts = empirical_scales(pickle.load(file))
    return {
        "format": FORMAT_VERSION, "new_samples": args.new_samples,
        "seed": args.seed, "distribution": args.distribution,
        "scale_counts": scale_counts, "finite_only": True,
        "n": N, "k_order": list(K_VALUES), "matrix_range": [-MAX_ENTRY, MAX_ENTRY],
        "numpy": np.__version__, "scipy": scipy.__version__,
        "solver": "highs-ds", "lp_tolerance": 1e-8, "inputs": inputs,
    }


def prepare_run(directory, config):
    config_path = directory / "run.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise ValueError("Run configuration/input differs from checkpoint. "
                             "Restore the original options/versions or use a new output directory.")
    else:
        if (list(directory.glob("checkpoint-*.pkl")) or
                (directory / EXT_FEATURES).exists() or (directory / EXT_LABELS).exists()):
            raise ValueError("Output files exist without run.json; use a new output directory")
        atomic_write(config_path, config, as_json=True)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def checkpoint_paths(directory):
    return sorted(directory.glob("checkpoint-*.pkl"))


def read_checkpoint(path, run_id, expected_matrix):
    with path.open("rb") as file:
        data = pickle.load(file)
    if (data["run_id"] != run_id or data["start_matrix"] != expected_matrix or
            data["next_matrix"] <= expected_matrix or
            len(data["features"]) != len(data["labels"]) or
            len(data["features"]) != data["sample_count"] or
            len(data["group_sizes"]) != data["next_matrix"] - data["start_matrix"] or
            sum(data["group_sizes"]) != data["sample_count"]):
        raise ValueError(f"Invalid or noncontiguous checkpoint: {path}")
    if path.name != f"checkpoint-{expected_matrix:012d}.pkl":
        raise ValueError(f"Checkpoint name disagrees with content: {path}")
    return data


def resume_state(directory, run_id):
    matrix_id, samples = 0, 0
    for path in checkpoint_paths(directory):
        data = read_checkpoint(path, run_id, matrix_id)
        # Each matrix stores its finite rows m=2,3,...; the group may be empty.
        index = 0
        for size in data["group_sizes"]:
            k = matrix_k(matrix_id)
            group = data["features"][index:index + size]
            if not 0 <= size <= N - k - 1 or len(group) != size:
                raise ValueError(f"Invalid matrix group in {path}")
            for offset, (n, stored_k, m, matrix) in enumerate(group):
                if (n != N or stored_k != k or m != offset + 2 or
                        not isinstance(matrix, np.ndarray) or matrix.shape != (k, N - k) or
                        not np.isfinite(matrix).all() or
                        not np.array_equal(matrix, group[0][3])):
                    raise ValueError(f"Invalid feature group in {path}")
            labels = np.asarray(data["labels"][index:index + size], dtype=float)
            if not np.isfinite(labels).all() or (labels < 1 - 1e-6).any():
                raise ValueError(f"Invalid labels in {path}")
            index += size
            samples += size
            matrix_id += 1
    return matrix_id, samples


def generate(args, config, directory, run_id, next_matrix, saved_samples):
    """Bound the outstanding work and commit results in matrix-ID order."""
    start = time.monotonic()
    checkpoint_time = progress_time = start
    start_samples = saved_samples
    checkpoint_start = next_matrix
    buffer_features, buffer_labels, buffer_sizes = [], [], []
    submitted_matrix = next_matrix
    pending = deque()
    scale_counts = config["scale_counts"]

    def flush():
        nonlocal checkpoint_start, checkpoint_time
        if not buffer_sizes:
            return
        path = directory / f"checkpoint-{checkpoint_start:012d}.pkl"
        atomic_write(path, {
            "run_id": run_id, "start_matrix": checkpoint_start,
            "next_matrix": next_matrix, "sample_count": len(buffer_features),
            "group_sizes": buffer_sizes,
            "features": buffer_features, "labels": buffer_labels,
        })
        dropped = sum(N - matrix_k(i) - 1 - size
                      for i, size in enumerate(buffer_sizes, checkpoint_start))
        log(f"Checkpoint committed: {saved_samples:,}/{args.new_samples:,} new rows; "
            f"{dropped} infinite-height rows skipped in this checkpoint")
        buffer_features.clear()
        buffer_labels.clear()
        buffer_sizes.clear()
        checkpoint_start = next_matrix
        checkpoint_time = time.monotonic()

    pool = mp.get_context("spawn").Pool(
        args.workers, initializer=worker_init, maxtasksperchild=1000)
    try:
        while saved_samples < args.new_samples and not STOP_REQUESTED:
            now = time.monotonic()
            if args.max_seconds and now - start >= args.max_seconds:
                log("Run time budget reached; saving completed work")
                break
            # Rows per matrix are unknown in advance; stop after the matrix that
            # reaches the target. Extra in-flight results are discarded.
            while len(pending) < 2 * args.workers:
                k = matrix_k(submitted_matrix)
                counts = scale_counts[str(k)] if scale_counts else None
                task = (submitted_matrix, k, args.seed, args.distribution, counts)
                pending.append(pool.apply_async(generate_matrix, (task,)))
                submitted_matrix += 1
            try:
                matrix_id, features, labels = pending[0].get(timeout=1.0)
            except mp.TimeoutError:
                pass
            else:
                pending.popleft()
                if matrix_id != next_matrix:
                    raise RuntimeError("Worker results out of order")
                buffer_features.extend(features)
                buffer_labels.extend(labels)
                buffer_sizes.append(len(features))
                saved_samples += len(features)
                next_matrix += 1
            now = time.monotonic()
            if buffer_sizes and (len(buffer_features) >= args.batch_size or
                    now - checkpoint_time >= args.checkpoint_seconds):
                flush()
            if now - progress_time >= args.progress_seconds:
                rate = (saved_samples - start_samples) / max(now - start, 1e-9)
                remaining = (args.new_samples - saved_samples) / rate / 3600 if rate else math.inf
                log(f"Progress {saved_samples:,}/{args.new_samples:,}; "
                    f"{rate:.3f} rows/s; estimated remaining {remaining:.1f} hours")
                progress_time = now
    finally:
        # Discard bounded in-flight work. It is reproducible on the next run.
        pool.terminate()
        pool.join()
        flush()
    elapsed = time.monotonic() - start
    log(f"This run: {saved_samples - start_samples:,} new rows in {elapsed:.1f}s")
    return saved_samples


def merge_outputs(args, directory, run_id):
    """Load the original lists only after workers exit, then append shards."""
    marker = directory / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["run_id"] != run_id:
            raise ValueError("Completion record does not match this run")
        for name in (EXT_FEATURES, EXT_LABELS):
            if sha256(directory / name) != record["sha256"][name]:
                raise ValueError(f"Completed output failed checksum: {name}")
        log(f"Already complete: {record['total_samples']:,} combined samples")
        return True
    log("Loading original feature and label lists for merge")
    with (args.input_dir / FEATURES).open("rb") as file:
        features = pickle.load(file)
    with (args.input_dir / LABELS).open("rb") as file:
        labels = pickle.load(file)
    if not isinstance(features, list) or not isinstance(labels, list) or len(features) != len(labels):
        raise ValueError("Original pickles must contain aligned lists")
    original_count = len(features)
    next_matrix = 0
    for path in checkpoint_paths(directory):
        if STOP_REQUESTED:
            return False
        data = read_checkpoint(path, run_id, next_matrix)
        features.extend(data["features"])
        labels.extend(data["labels"])
        next_matrix = data["next_matrix"]
    expected = original_count + args.new_samples
    if len(features) < expected or len(labels) != len(features):
        raise ValueError("Merged count is below the requested total")
    # The last matrix may overshoot; dropping its largest-m rows keeps a valid prefix.
    del features[expected:], labels[expected:]
    if STOP_REQUESTED:
        return False
    atomic_write(directory / EXT_FEATURES, features)
    del features
    if STOP_REQUESTED:
        return False
    atomic_write(directory / EXT_LABELS, labels)
    del labels
    if STOP_REQUESTED:
        return False
    # Consumers should require this marker: two files cannot be renamed atomically together.
    atomic_write(marker, {
        "run_id": run_id, "original_samples": original_count,
        "new_samples": args.new_samples, "total_samples": expected,
        "sha256": {name: sha256(directory / name) for name in (EXT_FEATURES, EXT_LABELS)},
    }, as_json=True)
    log(f"Completed {expected:,} combined samples in {directory}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    root = os.environ.get("SCRATCH") or os.environ.get("CFS")
    default_output = Path(root) / "csce636-project1-generation-v2" if root else None
    parser.add_argument("--output-dir", type=Path, default=default_output,
                        help="SCRATCH/CFS directory; default $SCRATCH/csce636-project1-generation-v2")
    parser.add_argument("--new-samples", type=int, default=DEFAULT_NEW_SAMPLES)
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes; Slurm script uses 64")
    parser.add_argument("--batch-size", type=int, default=50_000,
                        help="Checkpoint after at least this many rows (complete matrix groups)")
    parser.add_argument("--checkpoint-seconds", type=float, default=300)
    parser.add_argument("--progress-seconds", type=float, default=60)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Stop generation after this many seconds, saving completed matrices")
    parser.add_argument("--seed", type=int, default=636)
    parser.add_argument("--distribution", choices=("empirical", "integer", "uniform"),
                        default="empirical",
                        help="empirical: per-matrix scale a from the original data, "
                             "then integers uniform in [-a, a]")
    args = parser.parse_args()
    if args.new_samples < 1 or args.workers < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("new-samples >= 1, workers/batch-size >= 1, and seed >= 0 are required")
    if (not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0 or
            not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0 or
            not math.isfinite(args.max_seconds) or args.max_seconds < 0):
        parser.error("Time options must be finite; intervals > 0, max-seconds >= 0")
    if args.output_dir is None:
        parser.error("Set SCRATCH/CFS or specify --output-dir")
    args.input_dir = args.input_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    directory = validate_output_directory(args.output_dir)
    with output_lock(directory):
        config = run_config(args)
        run_id = prepare_run(directory, config)
        next_matrix, saved_samples = resume_state(directory, run_id)
        log(f"Resume: {saved_samples:,} rows, next matrix {next_matrix}; {args.workers} CPU workers")
        if STOP_REQUESTED:
            return 75
        if saved_samples < args.new_samples:
            saved_samples = generate(args, config, directory, run_id, next_matrix, saved_samples)
        if saved_samples < args.new_samples or STOP_REQUESTED:
            log("Stopped safely. Submit the same command to resume; exit code 75.")
            return 75
        return 0 if merge_outputs(args, directory, run_id) else 75


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
