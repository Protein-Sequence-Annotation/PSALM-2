#!/usr/bin/env python3
"""InterPro ground-truth evaluation against scored domain predictions.

Loads a ground-truth pickle (plain seq_id → domains, or a dict containing
``domain_dict`` for the same mapping). Streams one or more prediction pickles
from a directory, scores InterPro-aware family / family* overlap sensitivities
by length bucket, and writes a summary table and pickle.

When predictions carry bitscores (index 3) or i-evalues (index 4), optionally
builds a ROC-style curve: quantile-spaced score thresholds, per-threshold
fam* hit counts (single- and double-overlap*), saved under the output path as
``roc_by_threshold.pkl``. If a negatives directory is given, every predicted
domain counts as a false positive at each threshold; results are written as
``roc_by_threshold_FULL.pkl`` with per-threshold and per-bucket FP totals.

CLI: ``--help`` lists required paths (ground truth, prediction directory, PFAM→clan
map, InterPro member map, output prefix) and optional ROC / negatives options.

In this repo the module is installed as ``psalm.test.evaluate_predictions``; run via
``python scripts/test/evaluate_predictions.py``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

# ─────────────────────────────── types & constants ───────────────────────────── #
DomainGT = Tuple[str, int, int]                        # (pfam, start, stop)
DomainPred = Tuple[str, int, int, Any, Any, Any, Any]  # allow 7-field tuple
Bucket = Tuple[int, int]                               # (lo_excl, hi_incl)
BUCKETS: List[Bucket] = [
    (0, 25), (25, 50), (50, 100), (100, 200),
    (200, 400), (400, 800), (800, 1600), (1600, 3200),
]

# ─────────────────────────────── helpers ─────────────────────────────────────── #
def load_pickle(path: str) -> Any:
    return pickle.load(open(path, "rb"))


def save_pickle(obj: Any, path: str) -> None:
    pickle.dump(obj, open(path, "wb"))


def atomic_pickle_dump(obj: Any, final_path: str) -> None:
    """Atomically write a pickle to avoid truncation on failures."""
    tmp_path = final_path + ".tmp"
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)


def core_pfam(acc: str) -> str:
    return acc.split(".")[0]


def midpoint(start: int, end: int) -> int:
    return (start + end) // 2


def bucket_id(dom_len: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo < dom_len <= hi:
            return i
    return len(BUCKETS) - 1


# ──────────────────────────── domain-matching rules ──────────────────────────── #
def double_midpoint_overlap(gt_dom: DomainGT, pred_dom: Tuple[Any, ...]) -> bool:
    """Overlap iff both midpoints are inside the opposite interval."""
    _, s_gt, e_gt = gt_dom
    _, s_p, e_p, *rest = pred_dom
    mid_gt = midpoint(s_gt, e_gt)
    mid_p = midpoint(s_p, e_p)
    return (s_p <= mid_gt <= e_p) and (s_gt <= mid_p <= e_gt)


def single_midpoint_overlap(gt_dom: DomainGT, pred_dom: Tuple[Any, ...]) -> bool:
    """Overlap iff either midpoint is inside the other interval."""
    _, s_gt, e_gt = gt_dom
    _, s_p, e_p, *rest = pred_dom
    mid_gt = midpoint(s_gt, e_gt)
    mid_p = midpoint(s_p, e_p)
    return (s_p <= mid_gt <= e_p) or (s_gt <= mid_p <= e_gt)


def famstar_match(gt_dom: DomainGT, pred_dom: Tuple[Any, ...], fam2clan: Dict[str, str]) -> bool:
    """fam* match: same family OR same clan (if GT clan defined)."""
    pf_gt, *_ = gt_dom
    pf_p, *_ = pred_dom
    core_gt = core_pfam(pf_gt)
    core_p = core_pfam(pf_p)
    if core_p == core_gt:
        return True
    clan_gt = fam2clan.get(core_gt)
    clan_p = fam2clan.get(core_p)
    return clan_gt not in (None, "None") and clan_p == clan_gt


# ───────────────────────────── data structures ───────────────────────────────── #
@dataclass
class Tallies:
    # denominators (all GT)
    gt_total_all: int
    b_gt_total_all: List[int]

    # denominators limited to sequences seen in preds (for --only-preds)
    gt_total_seen: int
    b_gt_total_seen: List[int]

    # hits (double-overlap)
    fam_hit_d: int
    famstar_hit_d: int
    b_fam_hit_d: List[int]
    b_famstar_hit_d: List[int]

    # hits (double-overlap*)
    fam_hit_dstar: int
    famstar_hit_dstar: int
    b_fam_hit_dstar: List[int]
    b_famstar_hit_dstar: List[int]

    # hits (single-overlap)
    fam_hit_s: int
    famstar_hit_s: int
    b_fam_hit_s: List[int]
    b_famstar_hit_s: List[int]


def init_tallies(n_buckets: int, gt_buckets: List[int]) -> Tallies:
    return Tallies(
        gt_total_all=sum(gt_buckets),
        b_gt_total_all=list(gt_buckets),
        gt_total_seen=0,
        b_gt_total_seen=[0] * n_buckets,
        fam_hit_d=0,
        famstar_hit_d=0,
        b_fam_hit_d=[0] * n_buckets,
        b_famstar_hit_d=[0] * n_buckets,
        fam_hit_dstar=0,
        famstar_hit_dstar=0,
        b_fam_hit_dstar=[0] * n_buckets,
        b_famstar_hit_dstar=[0] * n_buckets,
        fam_hit_s=0,
        famstar_hit_s=0,
        b_fam_hit_s=[0] * n_buckets,
        b_famstar_hit_s=[0] * n_buckets,
    )


# ──────────────────────────────── core logic ─────────────────────────────────── #
def compute_gt_denominators(gt: Dict[str, List[DomainGT]]) -> List[int]:
    """Per-bucket GT domain counts after any filtering."""
    b_counts = [0] * len(BUCKETS)
    for doms in gt.values():
        for pf, s, e in doms:
            dom_len = e - s + 1
            b_counts[bucket_id(dom_len)] += 1
    return b_counts


def maybe_trim_gt_tuples(gt: Dict[str, List[DomainGT]]) -> None:
    """Warn once if GT tuples have extra fields, then trim to first 3 elements."""
    first_dom = None
    for doms in gt.values():
        if doms:
            first_dom = doms[0]
            break
    if first_dom is None:
        return
    try:
        needs_trim = len(first_dom) > 3
    except Exception:
        return
    if not needs_trim:
        return
    print("[WARN] Ground-truth domain tuples have >3 fields; trimming to first 3.", flush=True)
    for sid, doms in gt.items():
        trimmed = []
        for d in doms:
            if isinstance(d, (list, tuple)):
                trimmed.append(tuple(d[:3]))
            else:
                trimmed.append(d)
        gt[sid] = trimmed


def iter_prediction_pickles(preds_dir: str) -> Iterable[str]:
    for fname in sorted(os.listdir(preds_dir)):
        path = os.path.join(preds_dir, fname)
        if not os.path.isfile(path):
            continue
        if not (fname.endswith(".pkl") or fname.endswith(".pickle")):
            continue
        yield path


def iter_normalized_items(raw: Dict[str, Any]) -> Iterable[Tuple[str, List[DomainPred]]]:
    """Yield (normalized_seq_id, preds_list) without materializing a new dict.

    - If key contains '|', use the field after the first '|'
    - If value is a tuple(list, int) with an int second element, take the list
    - Otherwise assume it's the list directly
    """
    for k, v in raw.items():
        key = k.split("|")[1] if "|" in k else k
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int):
            preds_list = v[0]
        else:
            preds_list = v
        yield key, preds_list


def scan_global_score_range(
    pred_files: List[str], *, scored_preds: bool, use_evalue: bool = False, reservoir_size: int = 500000
) -> Tuple[Optional[float], Optional[float], int, List[float]]:
    """Scan all prediction pickles to find global (min_score, max_score) and a sample of scores.

    Assumes scored predictions: bitscore at index 3, or with use_evalue i-evalue at index 4.
    Tuple layout: (pfam, start, stop, score, e_val, ...).

    Returns (min_score, max_score, count_scored, sample_scores_reservoir).
    """
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    count_scored = 0
    if not scored_preds:
        return None, None, 0, []
    score_idx = 4 if use_evalue else 3
    sample: List[float] = []
    for pred_path in tqdm(pred_files, desc="Scanning pred scores"):
        try:
            raw = load_pickle(pred_path)
        except Exception:
            continue
        for _sid, preds in iter_normalized_items(raw):
            for p in (preds or []):
                try:
                    if len(p) <= score_idx:
                        continue
                    sc = float(p[score_idx])
                except Exception:
                    continue
                count_scored += 1
                if min_score is None or sc < min_score:
                    min_score = sc
                if max_score is None or sc > max_score:
                    max_score = sc
                # Reservoir sampling for approximate quantiles
                if reservoir_size > 0:
                    if len(sample) < reservoir_size:
                        sample.append(sc)
                    else:
                        j = random.randint(0, count_scored - 1)
                        if j < reservoir_size:
                            sample[j] = sc
        # explicit drop to free memory
        del raw
    return min_score, max_score, count_scored, sample


def quantile_thresholds_from_sample(sample: List[float], n: int) -> List[float]:
    if not sample:
        return []
    if n <= 1:
        return [min(sample)]
    sample_sorted = sorted(sample)
    m = len(sample_sorted)
    thresholds: List[float] = []
    for i in range(n):
        # Quantiles inclusive from 0..1
        q = i / float(n - 1)
        idx = int(q * (m - 1))
        thresholds.append(sample_sorted[idx])
    return thresholds


def is_famstar_match(
    gt_dom: DomainGT,
    pf_core_pred: str,
    clan_pred: Optional[str],
    fam2clan: Dict[str, str],
    *,
    interpro_mode: bool = False,
    ipr_to_pf_cores: Optional[Dict[str, Set[str]]] = None,
    ipr_to_clans: Optional[Dict[str, Set[str]]] = None,
) -> bool:
    if interpro_mode:
        ipr_id = gt_dom[0]
        pf_set = ipr_to_pf_cores.get(ipr_id, set()) if ipr_to_pf_cores else set()
        clan_set = ipr_to_clans.get(ipr_id, set()) if ipr_to_clans else set()
        fam_match_ipr = pf_core_pred in pf_set
        clan_match_ipr = (clan_pred not in (None, "None")) and (clan_pred in clan_set)
        return fam_match_ipr or clan_match_ipr
    # PFAM mode
    pf_gt_core = core_pfam(gt_dom[0])
    if pf_core_pred == pf_gt_core:
        return True
    clan_gt = fam2clan.get(pf_gt_core)
    return (clan_gt not in (None, "None")) and (clan_pred == clan_gt)


def compute_max_scores_for_sequence(
    sid: str,
    gt_list: List[DomainGT],
    preds: List[DomainPred],
    fam2clan: Dict[str, str],
    *,
    use_evalue: bool = False,
    interpro_mode: bool = False,
    ipr_to_pf_cores: Optional[Dict[str, Set[str]]] = None,
    ipr_to_clans: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[List[float], List[float]]:
    """
    For a single sequence, compute per-GT best scores for:
      - fam* single-overlap (best over preds that pass fam/clan and single-midpoint overlap)
      - fam* double* (direct double fam/fam* OR assignment among single-only preds that overlap >1 GT)

    With use_evalue=False (bitscore): best = max; missing = -inf.
    With use_evalue=True (e-value): best = min (lower is better); missing = +inf.

    Returns two lists of length len(gt_list): (best_single, best_dstar).
    """
    n_gt = len(gt_list)
    if use_evalue:
        missing = float('inf')
    else:
        missing = float('-inf')
    if n_gt == 0 or not preds:
        return [missing] * n_gt, [missing] * n_gt

    score_idx = 4 if use_evalue else 3
    # Normalize preds: (pf_core, clan_pred, s_p, e_p, score_or_eval)
    norm_preds: List[Tuple[str, Optional[str], int, int, float]] = []
    for p in preds or []:
        try:
            if len(p) <= score_idx:
                continue
            pf, s, e = p[0], p[1], p[2]
            sc_f = float(p[score_idx])
            pf_core = core_pfam(str(pf))
            s_i, e_i = int(s), int(e)
            clan_p = fam2clan.get(pf_core)
            norm_preds.append((pf_core, clan_p, s_i, e_i, sc_f))
        except Exception:
            continue

    if not norm_preds:
        return [missing] * n_gt, [missing] * n_gt

    # Precompute overlaps for each pred
    single_overlaps: List[List[int]] = [[] for _ in norm_preds]
    double_overlaps: List[List[int]] = [[] for _ in norm_preds]
    for pidx, (pf_core, clan_p, s_p, e_p, sc) in enumerate(norm_preds):
        for gidx, gt_dom in enumerate(gt_list):
            if single_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                single_overlaps[pidx].append(gidx)
                if double_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                    double_overlaps[pidx].append(gidx)

    max_single = [missing] * n_gt
    max_dstar = [missing] * n_gt
    better = (lambda sc, current: sc < current) if use_evalue else (lambda sc, current: sc > current)

    # Direct single and direct double contributions
    for pidx, (pf_core, clan_p, s_p, e_p, sc) in enumerate(norm_preds):
        # single fam*
        for gidx in single_overlaps[pidx]:
            if is_famstar_match(gt_list[gidx], pf_core, clan_p, fam2clan,
                                 interpro_mode=interpro_mode,
                                 ipr_to_pf_cores=ipr_to_pf_cores,
                                 ipr_to_clans=ipr_to_clans):
                if better(sc, max_single[gidx]):
                    max_single[gidx] = sc
        # direct double fam* contributes to d*
        if double_overlaps[pidx]:
            for gidx in double_overlaps[pidx]:
                if is_famstar_match(gt_list[gidx], pf_core, clan_p, fam2clan,
                                     interpro_mode=interpro_mode,
                                     ipr_to_pf_cores=ipr_to_pf_cores,
                                     ipr_to_clans=ipr_to_clans):
                    if better(sc, max_dstar[gidx]):
                        max_dstar[gidx] = sc

    # Assignment among single-only preds that overlap >1 GT (and no doubles)
    for pidx, (pf_core, clan_p, s_p, e_p, sc) in enumerate(norm_preds):
        if double_overlaps[pidx]:
            continue
        gts = single_overlaps[pidx]
        if len(gts) <= 1:
            continue  # only assign when single-overlaps multiple GTs
        # pick best GT: maximize overlap length; tie by longer GT; then earlier start
        best = None
        best_key = None
        for g in gts:
            pf_g, sg, eg = gt_list[g]
            ov_len = max(0, min(e_p, eg) - max(s_p, sg) + 1)
            gt_len = eg - sg + 1
            key = (ov_len, gt_len, -sg)
            if best is None or key > best_key:
                best = g
                best_key = key
        if best is not None:
            if is_famstar_match(gt_list[best], pf_core, clan_p, fam2clan,
                                 interpro_mode=interpro_mode,
                                 ipr_to_pf_cores=ipr_to_pf_cores,
                                 ipr_to_clans=ipr_to_clans):
                if better(sc, max_dstar[best]):
                    max_dstar[best] = sc

    return max_single, max_dstar


def update_denominators_seen(
    sid: str,
    gt: Dict[str, List[DomainGT]],
    seen: Set[str],
    tallies: Tallies,
) -> None:
    if sid in seen:
        return
    doms = gt.get(sid, [])
    if not doms:
        seen.add(sid)
        return
    b_local = [0] * len(BUCKETS)
    for _, s, e in doms:
        b_local[bucket_id(e - s + 1)] += 1
    for i in range(len(BUCKETS)):
        tallies.b_gt_total_seen[i] += b_local[i]
    tallies.gt_total_seen += sum(b_local)
    seen.add(sid)


def evaluate_sequence(
    sid: str,
    gt_list: List[DomainGT],
    preds: List[DomainPred],
    fam2clan: Dict[str, str],
    filter_score: float,
    *,
    use_evalue: bool = False,
    interpro_mode: bool = False,
    ipr_to_pf_cores: Optional[Dict[str, Set[str]]] = None,
    ipr_to_clans: Optional[Dict[str, Set[str]]] = None,
    save_matches: Optional[List[DomainPred]] = None,
    save_match_keys: Optional[Set[Tuple[Any, ...]]] = None,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[Tuple[str, str, int, int]]]:
    """Evaluate one sequence, returning bucketed hits:
    Returns 4 lists (len = n_buckets):
      - b_fam_hit_d, b_famstar_hit_d, b_fam_hit_s, b_famstar_hit_s
    """
    n_b = len(BUCKETS)
    b_fam_hit_d = [0] * n_b
    b_famstar_hit_d = [0] * n_b
    b_fam_hit_dstar = [0] * n_b
    b_famstar_hit_dstar = [0] * n_b
    b_fam_hit_s = [0] * n_b
    b_famstar_hit_s = [0] * n_b

    so_do_diff = [] # So Do differences sotred for this sequence

    score_idx = 4 if use_evalue else 3
    # Score filter and vectorization
    # Keep a reference to the original prediction tuple for saving matches
    filt: List[Tuple[str, str | None, int, int, int, float, DomainPred]] = []
    for p in preds:
        try:
            if len(p) <= score_idx:
                continue
            pf, s, e = p[0], p[1], p[2]
            sc_f = float(p[score_idx])
            if use_evalue:
                if sc_f > filter_score:
                    continue
            else:
                if sc_f < filter_score:
                    continue
            core = core_pfam(pf)
            clan = fam2clan.get(core)
            s_i, e_i = int(s), int(e)
            filt.append((core, clan, s_i, e_i, midpoint(s_i, e_i), sc_f, p))
        except Exception:
            continue

    # Precompute per-pred overlap relationships for d* assignment
    # Map each pred idx to GT indices it overlaps (single) and the subset it double-overlaps
    pred_overlaps_single: List[List[int]] = [[] for _ in filt]
    pred_overlaps_double: List[List[int]] = [[] for _ in filt]

    for pidx, (pf_core, clan_p, s_p, e_p, mid_p, sc, orig_pred) in enumerate(filt):
        for gidx, gt_dom in enumerate(gt_list):
            if single_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                pred_overlaps_single[pidx].append(gidx)
                if double_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                    pred_overlaps_double[pidx].append(gidx)

    # Evaluate per-GT hits for double and single
    for gt_idx, gt_dom in enumerate(gt_list):
        pf_gt, s_gt, e_gt = gt_dom
        bidx = bucket_id(e_gt - s_gt + 1)

        fam_ok_d = False
        famstar_ok_d = False
        fam_ok_s = False
        famstar_ok_s = False

        for pidx, (pf_core, clan_p, s_p, e_p, mid_p, sc, orig_pred) in enumerate(filt):
            # double-overlap
            if not (fam_ok_d and famstar_ok_d):
                if double_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                    if interpro_mode:
                        ipr_id = gt_dom[0]
                        pf_set = ipr_to_pf_cores.get(ipr_id, set()) if ipr_to_pf_cores else set()
                        clan_set = ipr_to_clans.get(ipr_id, set()) if ipr_to_clans else set()
                        fam_match_ipr = pf_core in pf_set
                        clan_match_ipr = (clan_p not in (None, "None")) and (clan_p in clan_set)
                        if fam_match_ipr:
                            fam_ok_d = True
                        # fam* degrades to fam when no clan for GT or pred
                        if fam_match_ipr or clan_match_ipr:
                            famstar_ok_d = True
                    else:
                        if pf_core == core_pfam(pf_gt):
                            fam_ok_d = True
                            famstar_ok_d = True
                        elif famstar_match(gt_dom, (pf_core, s_p, e_p, sc, 0, ""), fam2clan):
                            famstar_ok_d = True

            # single-overlap
            if not (fam_ok_s and famstar_ok_s):
                if single_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                    if interpro_mode:
                        ipr_id = gt_dom[0]
                        pf_set = ipr_to_pf_cores.get(ipr_id, set()) if ipr_to_pf_cores else set()
                        clan_set = ipr_to_clans.get(ipr_id, set()) if ipr_to_clans else set()
                        fam_match_ipr = pf_core in pf_set
                        clan_match_ipr = (clan_p not in (None, "None")) and (clan_p in clan_set)
                        if fam_match_ipr:
                            fam_ok_s = True
                        if fam_match_ipr or clan_match_ipr:
                            famstar_ok_s = True
                            if save_matches is not None:
                                key = tuple(orig_pred)
                                if save_match_keys is None or key not in save_match_keys:
                                    save_matches.append(orig_pred)
                                    if save_match_keys is not None:
                                        save_match_keys.add(key)
                    else:
                        if pf_core == core_pfam(pf_gt):
                            fam_ok_s = True
                            famstar_ok_s = True
                            if save_matches is not None:
                                key = tuple(orig_pred)
                                if save_match_keys is None or key not in save_match_keys:
                                    save_matches.append(orig_pred)
                                    if save_match_keys is not None:
                                        save_match_keys.add(key)
                        elif famstar_match(gt_dom, (pf_core, s_p, e_p, sc, 0, ""), fam2clan):
                            famstar_ok_s = True
                            if save_matches is not None:
                                key = tuple(orig_pred)
                                if save_match_keys is None or key not in save_match_keys:
                                    save_matches.append(orig_pred)
                                    if save_match_keys is not None:
                                        save_match_keys.add(key)

            if fam_ok_d and famstar_ok_d and fam_ok_s and famstar_ok_s:
                break

        if fam_ok_d:
            b_fam_hit_d[bidx] += 1
        if famstar_ok_d:
            b_famstar_hit_d[bidx] += 1
        if fam_ok_s:
            b_fam_hit_s[bidx] += 1
        if famstar_ok_s:
            b_famstar_hit_s[bidx] += 1

        if famstar_ok_s and not famstar_ok_d:
            so_do_diff.append((sid, pf_gt, s_gt, e_gt - s_gt+1)) # Just save the domain where so and do differ

    # Compute double-overlap* per-GT using assignment: include doubles; for preds that only single-overlap
    # multiple GTs (and none doubles), assign to exactly one GT by maximum overlap length, ties by longer GT, then earlier start
    for gidx, gt_dom in enumerate(gt_list):
        pf_gt, s_gt, e_gt = gt_dom
        bidx = bucket_id(e_gt - s_gt + 1)
        fam_ok_dstar = False
        famstar_ok_dstar = False

        # If already double-overlap fam/fam* matched, d* must include it
        if b_fam_hit_d[bidx] > 0:
            pass  # we will recompute per-GT, not from bucket totals

        # Check direct double as baseline
        for pf_core, clan_p, s_p, e_p, mid_p, sc, orig_pred in filt:
            if double_midpoint_overlap(gt_dom, (pf_core, s_p, e_p, sc, 0, "")):
                if interpro_mode:
                    ipr_id = gt_dom[0]
                    pf_set = ipr_to_pf_cores.get(ipr_id, set()) if ipr_to_pf_cores else set()
                    clan_set = ipr_to_clans.get(ipr_id, set()) if ipr_to_clans else set()
                    fam_match_ipr = pf_core in pf_set
                    clan_match_ipr = (clan_p not in (None, "None")) and (clan_p in clan_set)
                    if fam_match_ipr:
                        fam_ok_dstar = True
                    if fam_match_ipr or clan_match_ipr:
                        famstar_ok_dstar = True
                else:
                    if pf_core == core_pfam(pf_gt):
                        fam_ok_dstar = True
                        famstar_ok_dstar = True
                    elif famstar_match(gt_dom, (pf_core, s_p, e_p, sc, 0, ""), fam2clan):
                        famstar_ok_dstar = True
        # If not established by direct double, apply assignment for preds that single-overlap multiple GTs but don't double any
        if not (fam_ok_dstar and famstar_ok_dstar):
            # For each pred: if it single-overlaps >1 GT and double-overlaps none, compute the chosen GT by overlap
            for pidx, (pf_core, clan_p, s_p, e_p, mid_p, sc, orig_pred) in enumerate(filt):
                if pred_overlaps_double[pidx]:
                    continue  # skip preds that have any double; already handled above
                gts = pred_overlaps_single[pidx]
                if len(gts) <= 1:
                    continue
                # pick best GT for this pred
                best = None
                best_key = None
                for g in gts:
                    pf_g, sg, eg = gt_list[g]
                    ov_len = max(0, min(e_p, eg) - max(s_p, sg) + 1)
                    gt_len = eg - sg + 1
                    key = (ov_len, gt_len, -sg)  # maximize ov_len, then gt_len, then earlier start (smaller sg)
                    if best is None or key > best_key:
                        best = g
                        best_key = key
                if best == gidx:
                    # Check identity for d*
                    if interpro_mode:
                        ipr_id = gt_dom[0]
                        pf_set = ipr_to_pf_cores.get(ipr_id, set()) if ipr_to_pf_cores else set()
                        clan_set = ipr_to_clans.get(ipr_id, set()) if ipr_to_clans else set()
                        fam_match_ipr = pf_core in pf_set
                        clan_match_ipr = (clan_p not in (None, "None")) and (clan_p in clan_set)
                        if fam_match_ipr:
                            fam_ok_dstar = True
                        if fam_match_ipr or clan_match_ipr:
                            famstar_ok_dstar = True
                    else:
                        if pf_core == core_pfam(pf_gt):
                            fam_ok_dstar = True
                            famstar_ok_dstar = True
                        elif famstar_match(gt_dom, (pf_core, s_p, e_p, sc, 0, ""), fam2clan):
                            famstar_ok_dstar = True

        if fam_ok_dstar:
            b_fam_hit_dstar[bidx] += 1
        if famstar_ok_dstar:
            b_famstar_hit_dstar[bidx] += 1

    return b_fam_hit_d, b_famstar_hit_d, b_fam_hit_dstar, b_famstar_hit_dstar, b_fam_hit_s, b_famstar_hit_s, so_do_diff


def print_bucket_table(
    tallies: Tallies,
    header_note: str,
    only_preds: bool,
) -> None:
    # Choose denominators
    b_denoms = tallies.b_gt_total_seen if only_preds else tallies.b_gt_total_all
    denom_total = tallies.gt_total_seen if only_preds else tallies.gt_total_all

    fam_s = (tallies.fam_hit_d / denom_total) if denom_total else 0.0
    famst_s = (tallies.famstar_hit_d / denom_total) if denom_total else 0.0
    fam_s_so = (tallies.fam_hit_s / denom_total) if denom_total else 0.0
    famst_s_so = (tallies.famstar_hit_s / denom_total) if denom_total else 0.0

    print(f"[PROGRESS] {header_note}")
    print(f"Evaluated {denom_total} ground-truth domains so far\n")
    print(f"Double-overlap   family-sensitivity   : {fam_s:.4f}")
    print(f"Double-overlap   family*-sensitivity  : {famst_s:.4f}")
    dstar_f = (tallies.fam_hit_dstar / denom_total) if denom_total else 0.0
    dstar_fs = (tallies.famstar_hit_dstar / denom_total) if denom_total else 0.0
    print(f"Double-overlap*  family-sensitivity   : {dstar_f:.4f}")
    print(f"Double-overlap*  family*-sensitivity  : {dstar_fs:.4f}")
    print(f"Single-overlap  family-sensitivity  : {fam_s_so:.4f}")
    print(f"Single-overlap  family*-sensitivity : {famst_s_so:.4f}\n")

    hdr = "Bucket      | #GT | Fam-d  | Fam*-d | Fam-d* | Fam*-d* | Fam-so | Fam*-so"
    print(hdr)
    print("-" * len(hdr))
    for i, (lo, hi) in enumerate(BUCKETS):
        tot = b_denoms[i]
        f_d = (tallies.b_fam_hit_d[i] / tot) if tot else 0.0
        fs_d = (tallies.b_famstar_hit_d[i] / tot) if tot else 0.0
        f_ds = (tallies.b_fam_hit_dstar[i] / tot) if tot else 0.0
        fs_ds = (tallies.b_famstar_hit_dstar[i] / tot) if tot else 0.0
        f_s = (tallies.b_fam_hit_s[i] / tot) if tot else 0.0
        fs_s = (tallies.b_famstar_hit_s[i] / tot) if tot else 0.0
        print(f"({lo:>4},{hi:<4}] | {tot:>4d} | {f_d:6.4f} | {fs_d:7.4f} | {f_ds:7.4f} | {fs_ds:8.4f} | {f_s:7.4f} | {fs_s:8.4f}")
    print("")

    # Counts table (hits) with identical layout
    print("Counts (hits):")
    print(hdr)
    print("-" * len(hdr))
    for i, (lo, hi) in enumerate(BUCKETS):
        tot = b_denoms[i]
        c_f_d = tallies.b_fam_hit_d[i]
        c_fs_d = tallies.b_famstar_hit_d[i]
        c_f_ds = tallies.b_fam_hit_dstar[i]
        c_fs_ds = tallies.b_famstar_hit_dstar[i]
        c_f_s = tallies.b_fam_hit_s[i]
        c_fs_s = tallies.b_famstar_hit_s[i]
        print(f"({lo:>4},{hi:<4}] | {tot:>4d} | {c_f_d:6d} | {c_fs_d:7d} | {c_f_ds:7d} | {c_fs_ds:8d} | {c_f_s:7d} | {c_fs_s:8d}")
    print("")


def write_summary_outputs(
    output_prefix: str,
    tallies: Tallies,
    only_preds: bool,
) -> None:
    b_denoms = tallies.b_gt_total_seen if only_preds else tallies.b_gt_total_all
    denom_total = tallies.gt_total_seen if only_preds else tallies.gt_total_all

    summary = {
        "config": {
            "only_preds": only_preds,
        },
        "overall_double": {
            "gt_total": denom_total,
            "fam_hit": tallies.fam_hit_d,
            "famstar_hit": tallies.famstar_hit_d,
        },
        "overall_double_star": {
            "gt_total": denom_total,
            "fam_hit": tallies.fam_hit_dstar,
            "famstar_hit": tallies.famstar_hit_dstar,
        },
        "overall_single": {
            "gt_total": denom_total,
            "fam_hit": tallies.fam_hit_s,
            "famstar_hit": tallies.famstar_hit_s,
        },
        "buckets_double": [
            {
                "range": BUCKETS[i],
                "gt_total": b_denoms[i],
                "fam_hit": tallies.b_fam_hit_d[i],
                "famstar_hit": tallies.b_famstar_hit_d[i],
            }
            for i in range(len(BUCKETS))
        ],
        "buckets_double_star": [
            {
                "range": BUCKETS[i],
                "gt_total": b_denoms[i],
                "fam_hit": tallies.b_fam_hit_dstar[i],
                "famstar_hit": tallies.b_famstar_hit_dstar[i],
            }
            for i in range(len(BUCKETS))
        ],
        "buckets_single": [
            {
                "range": BUCKETS[i],
                "gt_total": b_denoms[i],
                "fam_hit": tallies.b_fam_hit_s[i],
                "famstar_hit": tallies.b_famstar_hit_s[i],
            }
            for i in range(len(BUCKETS))
        ],
    }

    pkl_path = f"{output_prefix}_summary.pkl"
    save_pickle(summary, pkl_path)

    # human-readable text mirror
    fam_s = (tallies.fam_hit_d / denom_total) if denom_total else 0.0
    famst_s = (tallies.famstar_hit_d / denom_total) if denom_total else 0.0
    fam_s_so = (tallies.fam_hit_s / denom_total) if denom_total else 0.0
    famst_s_so = (tallies.famstar_hit_s / denom_total) if denom_total else 0.0

    lines = []
    lines.append(f"Evaluated {denom_total} ground-truth domains\n")
    lines.append(f"Double-overlap   family-sensitivity   : {fam_s:.4f}\n")
    lines.append(f"Double-overlap   family*-sensitivity  : {famst_s:.4f}\n")
    dstar_f = (tallies.fam_hit_dstar / denom_total) if denom_total else 0.0
    dstar_fs = (tallies.famstar_hit_dstar / denom_total) if denom_total else 0.0
    lines.append(f"Double-overlap*  family-sensitivity   : {dstar_f:.4f}\n")
    lines.append(f"Double-overlap*  family*-sensitivity  : {dstar_fs:.4f}\n")
    lines.append(f"Single-overlap  family-sensitivity  : {fam_s_so:.4f}\n")
    lines.append(f"Single-overlap  family*-sensitivity : {famst_s_so:.4f}\n\n")
    hdr = "Bucket      | #GT | Fam-sens | Fam*-sens | Fam-so | Fam*-so"
    lines.append(hdr + "\n")
    lines.append("-" * len(hdr) + "\n")
    for i, (lo, hi) in enumerate(BUCKETS):
        tot = b_denoms[i]
        f_d = (tallies.b_fam_hit_d[i] / tot) if tot else 0.0
        fs_d = (tallies.b_famstar_hit_d[i] / tot) if tot else 0.0
        f_s = (tallies.b_fam_hit_s[i] / tot) if tot else 0.0
        fs_s = (tallies.b_famstar_hit_s[i] / tot) if tot else 0.0
        lines.append(
            f"({lo:>4},{hi:<4}] | {tot:>4d} | {f_d:8.4f} | {fs_d:8.4f} | {f_s:7.4f} | {fs_s:8.4f}\n"
        )

    txt_path = f"{output_prefix}_summary.txt"
    with open(txt_path, "w") as f:
        f.writelines(lines)

    print(f"[SAVED] summary → {pkl_path}")
    print(f"[SAVED] summary table → {txt_path}")


def merge_false_positives_into_roc(
    roc_obj: Dict[Any, Any],
    negatives_dir: str,
    *,
    use_evalue: bool,
) -> None:
    """Attach false_positive counts per threshold (mutates roc_obj in place)."""
    thresholds = sorted([k for k in roc_obj.keys() if isinstance(k, (int, float))])
    if not thresholds:
        raise ValueError("ROC dict has no numeric thresholds to update.")
    n_thr = len(thresholds)
    overall_diff = [0] * (n_thr + 1)
    bucket_diff: Dict[int, List[int]] = {i: [0] * (n_thr + 1) for i in range(len(BUCKETS))}

    score_idx_neg = 4 if use_evalue else 3
    neg_files = list(iter_prediction_pickles(negatives_dir))
    if not neg_files:
        print("[WARN] No .pkl files found in --negatives; applying false_positives=0 at all thresholds.", flush=True)

    for pred_path in tqdm(neg_files, desc="Negatives pickles"):
        raw = load_pickle(pred_path)
        for _sid, preds in iter_normalized_items(raw):
            for p in preds or []:
                try:
                    if len(p) <= score_idx_neg:
                        continue
                    _pf, s, e = p[0], p[1], p[2]
                    s_i, e_i = int(s), int(e)
                    sc_f = float(p[score_idx_neg])
                except Exception:
                    continue
                b = bucket_id(max(1, e_i - s_i + 1))
                if use_evalue:
                    j = bisect_left(thresholds, sc_f)
                    if j < n_thr:
                        overall_diff[j] += 1
                        overall_diff[n_thr] -= 1
                        bd = bucket_diff[b]
                        bd[j] += 1
                        bd[n_thr] -= 1
                else:
                    j = bisect_right(thresholds, sc_f) - 1
                    if j >= 0:
                        overall_diff[0] += 1
                        if j + 1 < n_thr:
                            overall_diff[j + 1] -= 1
                        bd = bucket_diff[b]
                        bd[0] += 1
                        if j + 1 < n_thr:
                            bd[j + 1] -= 1
        del raw

    overall_counts = [0] * n_thr
    run = 0
    for t in range(n_thr):
        run += overall_diff[t]
        overall_counts[t] = run
    bucket_counts: Dict[int, List[int]] = {i: [0] * n_thr for i in range(len(BUCKETS))}
    for i in range(len(BUCKETS)):
        run = 0
        diff = bucket_diff[i]
        for t in range(n_thr):
            run += diff[t]
            bucket_counts[i][t] = run

    for ti, thr in enumerate(thresholds):
        entry = roc_obj.get(thr)
        if not isinstance(entry, dict):
            continue
        if "overall" not in entry:
            entry["overall"] = {}
        entry["overall"]["false_positives"] = overall_counts[ti]
        if "buckets" not in entry or not isinstance(entry["buckets"], list) or len(entry["buckets"]) != len(BUCKETS):
            entry["buckets"] = [
                {
                    "range": BUCKETS[i],
                    "gt_total": 0,
                    "famstar_hit_dstar": 0,
                    "famstar_hit_s": 0,
                }
                for i in range(len(BUCKETS))
            ]
        for i in range(len(BUCKETS)):
            if not isinstance(entry["buckets"][i], dict):
                entry["buckets"][i] = {"range": BUCKETS[i]}
            entry["buckets"][i]["false_positives"] = bucket_counts[i][ti]
        roc_obj[thr] = entry


def main() -> None:
    p = argparse.ArgumentParser(
        description="InterPro scored evaluation + ROC + optional FP merge from negatives (single process)."
    )
    p.add_argument("--groundtruth", required=True, help="GT pickle (seq -> domains) or dict with ['domain_dict'].")
    p.add_argument("--preds-dir", required=True, help="Directory of prediction pickles.")
    p.add_argument(
        "--fam-clan",
        "--fam_clan",
        required=True,
        dest="fam_clan",
        help="Pickle: PFAM accession → clan id.",
    )
    p.add_argument("--interpro-map", required=True, help="InterPro IPR→members pickle.")
    p.add_argument(
        "--output",
        required=True,
        help="Output directory/prefix (summary → <output>_summary.pkl; ROC → <output>/roc_by_threshold.pkl).",
    )
    p.add_argument("--negatives", default=None, help="Optional directory of negative prediction pickles.")
    p.add_argument("--filter-score", type=float, default=-1000.0)
    p.add_argument("--use-evalue", action="store_true")
    p.add_argument("--only-preds", action="store_true")
    p.add_argument("--progress-every", type=int, default=50000)
    p.add_argument("--roc-n", type=int, default=1000)
    p.add_argument("--roc-seed", type=int, default=100)
    args = p.parse_args()

    if not os.path.isdir(args.preds_dir):
        raise ValueError(f"--preds-dir is not a directory: {args.preds_dir}")
    if args.negatives is not None and not os.path.isdir(args.negatives):
        raise ValueError(f"--negatives is not a directory: {args.negatives}")

    effective_filter = 1e10 if (args.use_evalue and args.filter_score < 0) else args.filter_score

    print("Loading ground-truth domains...", flush=True)
    gt_obj = load_pickle(args.groundtruth)
    if isinstance(gt_obj, dict) and "domain_dict" in gt_obj:
        gt: Dict[str, List[DomainGT]] = gt_obj["domain_dict"]
    else:
        gt = gt_obj
    print(f"Loaded ground-truth domains from {len(gt)} sequences.", flush=True)
    maybe_trim_gt_tuples(gt)

    print("Loading PFAM→clan mapping...", flush=True)
    fam2clan = load_pickle(args.fam_clan)
    print(f"Loaded PFAM→clan mapping from {len(fam2clan)} PFAMs.", flush=True)

    b_gt_total_all = compute_gt_denominators(gt)
    tallies = init_tallies(len(BUCKETS), b_gt_total_all)

    pred_files = list(iter_prediction_pickles(args.preds_dir))
    if not pred_files:
        print("[WARN] No .pkl files found in inputs; saving empty summary.", flush=True)
        write_summary_outputs(args.output, tallies, args.only_preds)
        return

    interpro_mode = True
    if args.roc_seed is not None:
        random.seed(args.roc_seed)

    threshold_scan_files = pred_files
    min_s, max_s, cnt, sample = scan_global_score_range(
        threshold_scan_files,
        scored_preds=True,
        use_evalue=args.use_evalue,
    )
    roc_thresholds: Optional[List[float]] = None
    if cnt and min_s is not None and max_s is not None:
        n_thr = max(1, int(args.roc_n))
        roc_thresholds = quantile_thresholds_from_sample(sample, n_thr)
        if roc_thresholds:
            roc_thresholds[0] = min_s
            roc_thresholds[-1] = max_s
    else:
        print("[WARN] No numeric scores found; skipping ROC.", flush=True)

    roc_diff_single: Optional[Dict[int, List[int]]] = None
    roc_diff_dstar: Optional[Dict[int, List[int]]] = None
    if roc_thresholds is not None:
        n_thr = len(roc_thresholds)
        roc_diff_single = {i: [0] * (n_thr + 1) for i in range(len(BUCKETS))}
        roc_diff_dstar = {i: [0] * (n_thr + 1) for i in range(len(BUCKETS))}

    seen_for_denoms: Set[str] = set()
    low_tpr_root = args.preds_dir
    low_tpr_path = os.path.join(low_tpr_root, "low_tpr.txt")
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception:
        pass
    so_do_diff_path = os.path.join(args.output, "so_do_diff.txt")
    so_do_diff_agg: List[Tuple[Any, ...]] = []
    sodo_file = open(so_do_diff_path, "w")
    try:
        open(low_tpr_path, "w").close()
        print(f"[INIT] low_tpr sink → {low_tpr_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not initialize low_tpr.txt: {e}")

    print("Loading InterPro map for evaluation...", flush=True)
    ipr_map_obj = load_pickle(args.interpro_map)
    ipr_to_pf_cores: Dict[str, Set[str]] = {}
    ipr_to_clans: Dict[str, Set[str]] = {}
    for ipr, members in ipr_map_obj.items():
        pf_cores = {core_pfam(m) for m in members if isinstance(m, str) and m.startswith("PF")}
        if pf_cores:
            ipr_to_pf_cores[ipr] = pf_cores
            ipr_to_clans[ipr] = {fam2clan.get(core) for core in pf_cores if fam2clan.get(core) not in (None, "None")}

    total_streamed = 0
    total_gt_streamed = 0
    t0 = time.time()
    for idx, pred_path in enumerate(tqdm(pred_files, desc="Prediction pickles")):
        print(f"\n[FILE {idx + 1}/{len(pred_files)}] {os.path.basename(pred_path)}", flush=True)
        raw = load_pickle(pred_path)
        pr_iter = iter_normalized_items(raw)
        shard_seen_for_denoms: Set[str] = set()
        shard_tallies = init_tallies(len(BUCKETS), [0] * len(BUCKETS))
        print("Evaluating sequences (streaming)...", flush=True)
        for sid, preds in pr_iter:
            total_streamed += 1
            if args.progress_every > 0 and (total_streamed % args.progress_every) == 0:
                elapsed = max(1e-9, time.time() - t0)
                rate = total_streamed / elapsed
                print(
                    f"[PROGRESS] file={os.path.basename(pred_path)} "
                    f"streamed={total_streamed} gt_streamed={total_gt_streamed} "
                    f"elapsed_s={elapsed:.1f} rate_rec_per_s={rate:.1f}",
                    flush=True,
                )
            gt_list = gt.get(sid)
            if not gt_list:
                continue
            total_gt_streamed += 1

            if args.only_preds:
                update_denominators_seen(sid, gt, seen_for_denoms, tallies)
            update_denominators_seen(sid, gt, shard_seen_for_denoms, shard_tallies)

            b_f_d, b_fs_d, b_f_ds, b_fs_ds, b_f_s, b_fs_s, so_do_diff = evaluate_sequence(
                sid,
                gt_list,
                preds,
                fam2clan,
                effective_filter,
                use_evalue=args.use_evalue,
                interpro_mode=interpro_mode,
                ipr_to_pf_cores=ipr_to_pf_cores,
                ipr_to_clans=ipr_to_clans,
                save_matches=None,
                save_match_keys=None,
            )

            if len(so_do_diff) > 0:
                so_do_diff_agg.extend(so_do_diff)
            if len(so_do_diff_agg) > 10000:
                for pair in so_do_diff_agg:
                    sodo_file.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")
                so_do_diff_agg = []

            tallies.fam_hit_d += sum(b_f_d)
            tallies.famstar_hit_d += sum(b_fs_d)
            tallies.fam_hit_dstar += sum(b_f_ds)
            tallies.famstar_hit_dstar += sum(b_fs_ds)
            tallies.fam_hit_s += sum(b_f_s)
            tallies.famstar_hit_s += sum(b_fs_s)

            for i in range(len(BUCKETS)):
                tallies.b_fam_hit_d[i] += b_f_d[i]
                tallies.b_famstar_hit_d[i] += b_fs_d[i]
                tallies.b_fam_hit_dstar[i] += b_f_ds[i]
                tallies.b_famstar_hit_dstar[i] += b_fs_ds[i]
                tallies.b_fam_hit_s[i] += b_f_s[i]
                tallies.b_famstar_hit_s[i] += b_fs_s[i]

            shard_tallies.fam_hit_d += sum(b_f_d)
            shard_tallies.famstar_hit_d += sum(b_fs_d)
            shard_tallies.fam_hit_dstar += sum(b_f_ds)
            shard_tallies.famstar_hit_dstar += sum(b_fs_ds)
            shard_tallies.fam_hit_s += sum(b_f_s)
            shard_tallies.famstar_hit_s += sum(b_fs_s)
            for i in range(len(BUCKETS)):
                shard_tallies.b_fam_hit_d[i] += b_f_d[i]
                shard_tallies.b_famstar_hit_d[i] += b_fs_d[i]
                shard_tallies.b_fam_hit_dstar[i] += b_f_ds[i]
                shard_tallies.b_famstar_hit_dstar[i] += b_fs_ds[i]
                shard_tallies.b_fam_hit_s[i] += b_f_s[i]
                shard_tallies.b_famstar_hit_s[i] += b_fs_s[i]

            if roc_thresholds is not None and roc_diff_single is not None and roc_diff_dstar is not None:
                max_single_seq, max_dstar_seq = compute_max_scores_for_sequence(
                    sid,
                    gt_list,
                    preds,
                    fam2clan,
                    use_evalue=args.use_evalue,
                    interpro_mode=interpro_mode,
                    ipr_to_pf_cores=ipr_to_pf_cores,
                    ipr_to_clans=ipr_to_clans,
                )
                n_thr_local = len(roc_thresholds)
                roc_sentinel = float("inf") if args.use_evalue else float("-inf")
                for gidx, gt_dom in enumerate(gt_list):
                    _pf, s_gt, e_gt = gt_dom[0], gt_dom[1], gt_dom[2]
                    b = bucket_id(e_gt - s_gt + 1)
                    ms = max_single_seq[gidx]
                    if ms != roc_sentinel:
                        if args.use_evalue:
                            j = bisect_left(roc_thresholds, ms)
                            if j < n_thr_local:
                                roc_diff_single[b][j] += 1
                                roc_diff_single[b][n_thr_local] -= 1
                        else:
                            j = bisect_right(roc_thresholds, ms) - 1
                            if j >= 0:
                                roc_diff_single[b][0] += 1
                                if j + 1 < n_thr_local:
                                    roc_diff_single[b][j + 1] -= 1
                    md = max_dstar_seq[gidx]
                    if md != roc_sentinel:
                        if args.use_evalue:
                            j = bisect_left(roc_thresholds, md)
                            if j < n_thr_local:
                                roc_diff_dstar[b][j] += 1
                                roc_diff_dstar[b][n_thr_local] -= 1
                        else:
                            j = bisect_right(roc_thresholds, md) - 1
                            if j >= 0:
                                roc_diff_dstar[b][0] += 1
                                if j + 1 < n_thr_local:
                                    roc_diff_dstar[b][j + 1] -= 1

        print_bucket_table(
            shard_tallies,
            header_note=f"shard-only after {os.path.basename(pred_path)}",
            only_preds=True,
        )
        print_bucket_table(
            tallies,
            header_note=f"after {os.path.basename(pred_path)}",
            only_preds=args.only_preds,
        )
        del raw

    if not args.only_preds:
        tallies.gt_total_seen = tallies.gt_total_all
        tallies.b_gt_total_seen = list(tallies.b_gt_total_all)

    if so_do_diff_agg:
        for pair in so_do_diff_agg:
            sodo_file.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")
    sodo_file.close()

    write_summary_outputs(args.output, tallies, args.only_preds)

    if roc_thresholds is not None and roc_diff_single is not None and roc_diff_dstar is not None:
        print("[FINALIZE] Computing ROC counts via prefix sums...", flush=True)
        n_thr = len(roc_thresholds)
        counts_single_per_bucket: Dict[int, List[int]] = {i: [0] * n_thr for i in range(len(BUCKETS))}
        counts_dstar_per_bucket: Dict[int, List[int]] = {i: [0] * n_thr for i in range(len(BUCKETS))}
        for i in range(len(BUCKETS)):
            run = 0
            diff = roc_diff_single[i]
            for t in range(n_thr):
                run += diff[t]
                counts_single_per_bucket[i][t] = run
            run = 0
            diff = roc_diff_dstar[i]
            for t in range(n_thr):
                run += diff[t]
                counts_dstar_per_bucket[i][t] = run

        print("[FINALIZE] Building ROC summary and saving...", flush=True)
        roc_summary: Dict[float, Dict[str, Any]] = {}
        for ti, thr in enumerate(roc_thresholds):
            fs_s_hits = 0
            fs_ds_hits = 0
            b_fs_s_hits = [0] * len(BUCKETS)
            b_fs_ds_hits = [0] * len(BUCKETS)
            for i in range(len(BUCKETS)):
                s_cnt = counts_single_per_bucket[i][ti]
                d_cnt = counts_dstar_per_bucket[i][ti]
                b_fs_s_hits[i] = s_cnt
                b_fs_ds_hits[i] = d_cnt
                fs_s_hits += s_cnt
                fs_ds_hits += d_cnt
            roc_summary[thr] = {
                "threshold": thr,
                "overall": {
                    "gt_total": tallies.gt_total_all,
                    "famstar_hit_dstar": fs_ds_hits,
                    "famstar_hit_s": fs_s_hits,
                },
                "buckets": [
                    {
                        "range": BUCKETS[i],
                        "gt_total": b_gt_total_all[i],
                        "famstar_hit_dstar": b_fs_ds_hits[i],
                        "famstar_hit_s": b_fs_s_hits[i],
                    }
                    for i in range(len(BUCKETS))
                ],
            }

        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        roc_pkl = os.path.join(out_dir, "roc_by_threshold.pkl")
        atomic_pickle_dump(roc_summary, roc_pkl)
        print(f"[SAVED] ROC-by-threshold (sensitivities only) → {roc_pkl}", flush=True)

        if args.negatives:
            print("[INIT] Merging false positives from negatives...", flush=True)
            roc_full = load_pickle(roc_pkl)
            merge_false_positives_into_roc(roc_full, args.negatives, use_evalue=args.use_evalue)
            roc_full_path = os.path.join(out_dir, "roc_by_threshold_FULL.pkl")
            atomic_pickle_dump(roc_full, roc_full_path)
            print(f"[SAVED] ROC dict with FPs → {roc_full_path}", flush=True)

    elif args.negatives:
        print("[WARN] --negatives ignored: ROC was not computed (no scores in predictions).", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
