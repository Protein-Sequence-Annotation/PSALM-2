#!/usr/bin/env python3
"""One-shot InterPro GT: IPR map → global 10% pass-1 → passing_iprs.txt → pass-2 consistent-with-global.

Standalone subset of build_ipr_groundtruth.py for the BY_GLOBAL_AVG / consistent-with-global workflow.
Published as ``psalm.data.build_ipr_expanded_test``; run via ``python scripts/data/build_ipr_expanded_test.py``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Match, Optional, Pattern, Set, Tuple


LINE_PATTERN_FULL: Pattern[str] = re.compile(
    r"^(?P<seq_id>\S+)\s+"
    r"(?P<ipr>\S+)\s+"
    r"(?P<desc>.*?)\s+"
    r"(?P<member>\S+)\s+"
    r"(?P<start>\d+)\s+"
    r"(?P<stop>\d+)\s*$"
)

LINE_PATTERN_FALLBACK: Pattern[str] = re.compile(
    r"^(?P<seq_id>\S+)\s+"
    r"(?P<ipr>\S+)\s+"
    r"(?P<desc>.*?)\s+"
    r"(?P<member>\S+)\s*$"
)

IPR_PATTERN: Pattern[str] = re.compile(r"^IPR\d{6}$")
PFAM_PATTERN: Pattern[str] = re.compile(r"^PF\d{5}$")

@dataclass
class Pass1Counters:
    total_lines: int = 0
    parse_fail_lines: int = 0
    invalid_interval_lines: int = 0
    eligible_rows: int = 0


@dataclass
class Pass2Counters:
    total_lines: int = 0
    parse_fail_lines: int = 0
    invalid_interval_lines: int = 0
    eligible_rows: int = 0
    sequences_seen: int = 0
    sequences_with_candidates: int = 0
    sequences_output: int = 0
    candidates_total: int = 0
    placed_total: int = 0
    group_discard_overlap_with_placed: int = 0
    group_discard_single_not_double: int = 0
    group_discard_global_inconsistent: int = 0
    simplified_id_collisions: int = 0


@dataclass
class PlacementStats:
    candidates_total: int = 0
    placed_total: int = 0
    group_discard_overlap_with_placed: int = 0
    group_discard_single_not_double: int = 0
    group_discard_global_inconsistent: int = 0


# Worker-global pooled averages for pass-2 parallel mode.
_WORKER_POOLED_AVG: Dict[str, float] = {}
_WORKER_MODE = "default"
_WORKER_MAX_DIFF_FRAC = 0.10


def iter_lines_with_metrics(path: str) -> Iterator[Tuple[str, Optional[int], Optional[int]]]:
    """
    Yield (line, bytes_read, total_bytes).
    For plain-text input: bytes_read/total_bytes are available.
    For .gz input: bytes_read/total_bytes are None (ETA is not reported).
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line, None, None
        return

    total_bytes = os.path.getsize(path)
    bytes_read = 0
    with open(path, "rb") as fh:
        for raw in fh:
            bytes_read += len(raw)
            line = raw.decode("utf-8", errors="ignore")
            yield line, bytes_read, total_bytes


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def simplify_id(seq_id: str) -> str:
    # Repository convention: middle pipe field if >=3; else strip version suffix.
    try:
        if "|" in seq_id:
            parts = seq_id.split("|")
            if len(parts) >= 3:
                return parts[1]
        if "." in seq_id:
            return seq_id.split(".", 1)[0]
        return seq_id
    except Exception:
        return seq_id


def midpoint(start: int, stop: int) -> int:
    return (start + stop) // 2


def interval_length(start: int, stop: int) -> int:
    return stop - start + 1


def is_pfam_accession(acc: str) -> bool:
    return PFAM_PATTERN.match(acc) is not None


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def log_progress(
    stage: str,
    started_at: float,
    lines: int,
    eligible_rows: Optional[int],
    bytes_read: Optional[int],
    total_bytes: Optional[int],
) -> None:
    elapsed = max(1e-9, time.perf_counter() - started_at)
    lines_per_sec = lines / elapsed
    msg = f"[{stage}] lines={lines:,} elapsed={format_seconds(elapsed)} rate={lines_per_sec:,.0f} lines/s"
    if eligible_rows is not None:
        msg += f" eligible={eligible_rows:,}"
    if bytes_read is not None and total_bytes is not None and total_bytes > 0:
        frac = bytes_read / total_bytes
        bytes_per_sec = bytes_read / elapsed
        if bytes_per_sec > 0 and frac < 1.0:
            eta_sec = (total_bytes - bytes_read) / bytes_per_sec
            msg += f" progress={frac*100:.1f}% eta={format_seconds(eta_sec)}"
        else:
            msg += f" progress={frac*100:.1f}%"
    print(msg, flush=True)


def coordinate_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    return max(a_start, b_start) <= min(a_stop, b_stop)


def single_midpoint_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    mid_a = midpoint(a_start, a_stop)
    mid_b = midpoint(b_start, b_stop)
    return (b_start <= mid_a <= b_stop) or (a_start <= mid_b <= a_stop)


def double_midpoint_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    mid_a = midpoint(a_start, a_stop)
    mid_b = midpoint(b_start, b_stop)
    return (b_start <= mid_a <= b_stop) and (a_start <= mid_b <= a_stop)


def parse_line_full(line: str) -> Optional[Tuple[str, str, str, int, int]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m: Optional[Match[str]] = LINE_PATTERN_FULL.match(s)
    if m is None:
        return None
    ipr = m.group("ipr")
    if not IPR_PATTERN.match(ipr):
        return None
    return (
        m.group("seq_id"),
        ipr,
        m.group("member"),
        int(m.group("start")),
        int(m.group("stop")),
    )


def parse_line_for_map(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m: Optional[Match[str]] = LINE_PATTERN_FULL.match(s)
    if m is None:
        m = LINE_PATTERN_FALLBACK.match(s)
        if m is None:
            return None
    ipr = m.group("ipr")
    if not IPR_PATTERN.match(ipr):
        return None
    member = m.group("member")
    if not member:
        return None
    return ipr, member


def sort_members_with_pfam_first(members: Iterable[str]) -> List[str]:
    uniq = set(members)
    return sorted(uniq, key=lambda x: (0 if is_pfam_accession(x) else 1, x))


def normalize_ipr_map(obj: object) -> Dict[str, List[str]]:
    if not isinstance(obj, dict):
        raise TypeError("IPR map must be a dict[str, iterable[str]].")
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        ipr = str(k)
        if not IPR_PATTERN.match(ipr):
            continue
        if isinstance(v, (set, list, tuple)):
            members = [str(x) for x in v]
        else:
            continue
        members = sort_members_with_pfam_first(members)
        if not members:
            continue
        if any(is_pfam_accession(m) for m in members):
            out[ipr] = members
    return out


def build_ipr_map_with_pfam(dat_file: str, progress_every: int) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    ipr_to_members: Dict[str, Set[str]] = defaultdict(set)
    stats = {"total_lines": 0, "parse_fail_lines": 0}
    t0 = time.perf_counter()
    for line, bytes_read, total_bytes in iter_lines_with_metrics(dat_file):
            stats["total_lines"] += 1
            parsed = parse_line_for_map(line)
            if parsed is None:
                stats["parse_fail_lines"] += 1
                continue
            ipr, member = parsed
            ipr_to_members[ipr].add(member)
            if progress_every > 0 and (stats["total_lines"] % progress_every) == 0:
                log_progress(
                    stage="map",
                    started_at=t0,
                    lines=stats["total_lines"],
                    eligible_rows=len(ipr_to_members),
                    bytes_read=bytes_read,
                    total_bytes=total_bytes,
                )
    filtered: Dict[str, List[str]] = {}
    for ipr, members in ipr_to_members.items():
        if any(is_pfam_accession(m) for m in members):
            filtered[ipr] = sort_members_with_pfam_first(members)
    return filtered, stats


def load_or_build_ipr_map(
    dat_file: str,
    ipr_map_pkl: Optional[str],
    save_map_pkl: Optional[str],
    progress_every: int,
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    if ipr_map_pkl:
        with open(ipr_map_pkl, "rb") as fh:
            obj = pickle.load(fh)
        ipr_map = normalize_ipr_map(obj)
        return ipr_map, {"source": "loaded", "loaded_iprs": len(ipr_map)}
    ipr_map, stats = build_ipr_map_with_pfam(dat_file, progress_every)
    if save_map_pkl:
        ensure_parent_dir(save_map_pkl)
        with open(save_map_pkl, "wb") as fh:
            pickle.dump(ipr_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
    stats["source"] = "built"
    stats["built_iprs"] = len(ipr_map)
    return ipr_map, stats


def run_pass1_filter_iprs(
    dat_file: str,
    ipr_to_members: Dict[str, List[str]],
    max_diff_frac: float,
    progress_every: int,
) -> Tuple[Set[str], Dict[str, float], Pass1Counters]:
    members_by_ipr: Dict[str, Set[str]] = {ipr: set(members) for ipr, members in ipr_to_members.items()}
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lengths: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ctr = Pass1Counters()
    t0 = time.perf_counter()

    for line, bytes_read, total_bytes in iter_lines_with_metrics(dat_file):
            ctr.total_lines += 1
            parsed = parse_line_full(line)
            if parsed is None:
                ctr.parse_fail_lines += 1
                continue
            _seq, ipr, member, start, stop = parsed
            if start > stop:
                ctr.invalid_interval_lines += 1
                continue
            allowed_members = members_by_ipr.get(ipr)
            if not allowed_members:
                continue
            if member not in allowed_members:
                continue
            l = interval_length(start, stop)
            counts[ipr][member] += 1
            lengths[ipr][member] += l
            ctr.eligible_rows += 1
            if progress_every > 0 and (ctr.total_lines % progress_every) == 0:
                log_progress(
                    stage="pass1",
                    started_at=t0,
                    lines=ctr.total_lines,
                    eligible_rows=ctr.eligible_rows,
                    bytes_read=bytes_read,
                    total_bytes=total_bytes,
                )

    passing_iprs: Set[str] = set()
    pooled_avg: Dict[str, float] = {}

    for ipr in sorted(ipr_to_members):
        ipr_counts = counts.get(ipr, {})
        ipr_lengths = lengths.get(ipr, {})
        member_avgs: List[float] = []
        used_members: List[str] = []
        for member in ipr_to_members[ipr]:
            c = ipr_counts.get(member, 0)
            if c <= 0:
                continue
            t = ipr_lengths.get(member, 0)
            if t <= 0:
                continue
            member_avgs.append(t / c)
            used_members.append(member)
        if not member_avgs:
            continue

        keep = True
        for i in range(len(member_avgs)):
            for j in range(i + 1, len(member_avgs)):
                mx = member_avgs[i] if member_avgs[i] >= member_avgs[j] else member_avgs[j]
                mn = member_avgs[j] if mx == member_avgs[i] else member_avgs[i]
                frac = (mx - mn) / mx if mx > 0 else 0.0
                if frac > max_diff_frac:
                    keep = False
                    break
            if not keep:
                break
        if not keep:
            continue

        total_len = 0
        total_cnt = 0
        for member in used_members:
            total_len += ipr_lengths[member]
            total_cnt += ipr_counts[member]
        if total_cnt <= 0:
            continue

        passing_iprs.add(ipr)
        pooled_avg[ipr] = total_len / total_cnt
    return passing_iprs, pooled_avg, ctr


def write_passing_iprs_txt(out_path: str, pooled_avg: Dict[str, float], precision: int) -> None:
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("ipr_id\tglobal_avg_len\n")
        for ipr in sorted(pooled_avg):
            fh.write(f"{ipr}\t{pooled_avg[ipr]:.{precision}f}\n")


def _choose_group_winner(group: List[Tuple[int, str, int, int]], pooled_avg: Dict[str, float]) -> int:
    # group entries: (local_idx, ipr, start, stop)
    def rank_key(item: Tuple[int, str, int, int]) -> Tuple[float, int, int, int]:
        local_idx, ipr, start, stop = item
        avg = pooled_avg.get(ipr, 0.0)
        length = interval_length(start, stop)
        return (abs(length - avg), -length, start, stop)

    return min(group, key=rank_key)[0]


def _choose_group_winner_longest(group: List[Tuple[int, str, int, int]]) -> int:
    # Stable tie-break: max() returns first among equals, preserving group insertion order.
    return max(group, key=lambda item: interval_length(item[2], item[3]))[0]


def _within_global_length(length: int, global_avg: float, max_diff_frac: float) -> bool:
    denom = max(float(length), float(global_avg))
    if denom <= 0.0:
        return False
    return (abs(float(length) - float(global_avg)) / denom) <= max_diff_frac


def _build_same_ipr_component(
    seed_idx: int,
    ordered: List[Tuple[str, int, int]],
    blocked: Set[int],
) -> List[Tuple[int, str, int, int]]:
    seed_ipr, _, _ = ordered[seed_idx]
    seen: Set[int] = {seed_idx}
    stack: List[int] = [seed_idx]
    while stack:
        u = stack.pop()
        _ipr_u, s_u, e_u = ordered[u]
        for v, (ipr_v, s_v, e_v) in enumerate(ordered):
            if v in blocked or v in seen:
                continue
            if ipr_v != seed_ipr:
                continue
            if coordinate_overlap(s_u, e_u, s_v, e_v):
                seen.add(v)
                stack.append(v)
    return [(idx, ordered[idx][0], ordered[idx][1], ordered[idx][2]) for idx in sorted(seen)]


def _validate_placed_intervals(placed: List[Tuple[str, int, int]]) -> None:
    n = len(placed)
    for i in range(n):
        _, s1, e1 = placed[i]
        for j in range(i + 1, n):
            _, s2, e2 = placed[j]
            if coordinate_overlap(s1, e1, s2, e2):
                raise RuntimeError(f"Validation failure: coordinate overlap in placed intervals: {(s1, e1)} vs {(s2, e2)}")
            if single_midpoint_overlap(s1, e1, s2, e2):
                raise RuntimeError(f"Validation failure: single-midpoint overlap in placed intervals: {(s1, e1)} vs {(s2, e2)}")


def place_domains_for_sequence(
    candidates: List[Tuple[str, int, int]],
    pooled_avg: Dict[str, float],
    placement_mode: str = "default",
    max_diff_frac: float = 0.10,
) -> Tuple[List[Tuple[str, int, int]], PlacementStats]:
    # Candidate tuple: (ipr, start, stop)
    stats = PlacementStats(candidates_total=len(candidates))
    if not candidates:
        return [], stats

    # Deterministic candidate order.
    ordered = sorted(candidates, key=lambda t: (t[1], t[2], t[0]))
    placed_idx: Set[int] = set()
    non_placeable: Set[int] = set()
    placed_list: List[Tuple[str, int, int]] = []

    for i, (ipr_c, s_c, e_c) in enumerate(ordered):
        if i in placed_idx or i in non_placeable:
            continue

        if placement_mode != "consistent_with_global":
            # Step 3 gate (strict overlap against placed), early reject.
            overlaps_placed = any(
                coordinate_overlap(s_c, e_c, s_p, e_p) for (_ipr_p, s_p, e_p) in placed_list
            )
            if overlaps_placed:
                non_placeable.add(i)
                continue

        if placement_mode == "consistent_with_global":
            blocked = placed_idx | non_placeable
            G = _build_same_ipr_component(i, ordered, blocked)
        else:
            # Build local same-IPR group G (direct overlaps with c, non-transitive).
            G = []
            for j, (ipr_j, s_j, e_j) in enumerate(ordered):
                if j in placed_idx or j in non_placeable:
                    continue
                if ipr_j != ipr_c:
                    continue
                if j == i:
                    G.append((j, ipr_j, s_j, e_j))
                    continue
                if single_midpoint_overlap(s_c, e_c, s_j, e_j) or double_midpoint_overlap(s_c, e_c, s_j, e_j):
                    G.append((j, ipr_j, s_j, e_j))

        # If any domain in G overlaps already placed, discard entire G.
        bad_g = False
        for gidx, _gipr, gs, ge in G:
            if any(coordinate_overlap(gs, ge, ps, pe) for (_pipr, ps, pe) in placed_list):
                bad_g = True
                break
        if bad_g:
            for gidx, _gipr, _gs, _ge in G:
                non_placeable.add(gidx)
            stats.group_discard_overlap_with_placed += 1
            continue

        candidate_idx = i
        if len(G) > 1:
            if placement_mode == "default":
                has_single_not_double = False
                for a, b in combinations(G, 2):
                    _ia, _ipra, sa, ea = a
                    _ib, _iprb, sb, eb = b
                    single = single_midpoint_overlap(sa, ea, sb, eb)
                    double = double_midpoint_overlap(sa, ea, sb, eb)
                    if single and (not double):
                        has_single_not_double = True
                        break
                if has_single_not_double:
                    for gidx, _gipr, _gs, _ge in G:
                        non_placeable.add(gidx)
                    stats.group_discard_single_not_double += 1
                    continue
                winner = _choose_group_winner(G, pooled_avg)
            elif placement_mode == "consistent_with_global":
                has_not_double = False
                for a, b in combinations(G, 2):
                    _ia, _ipra, sa, ea = a
                    _ib, _iprb, sb, eb = b
                    if not double_midpoint_overlap(sa, ea, sb, eb):
                        has_not_double = True
                        break
                if has_not_double:
                    for gidx, _gipr, _gs, _ge in G:
                        non_placeable.add(gidx)
                    stats.group_discard_single_not_double += 1
                    continue

                has_global_inconsistent = False
                for _gidx, gipr, gs, ge in G:
                    gavg = pooled_avg.get(gipr)
                    if gavg is None:
                        has_global_inconsistent = True
                        break
                    glen = interval_length(gs, ge)
                    if not _within_global_length(glen, gavg, max_diff_frac):
                        has_global_inconsistent = True
                        break
                if has_global_inconsistent:
                    for gidx, _gipr, _gs, _ge in G:
                        non_placeable.add(gidx)
                    stats.group_discard_global_inconsistent += 1
                    continue
                winner = _choose_group_winner(G, pooled_avg)
            else:
                winner = _choose_group_winner_longest(G)
            for gidx, _gipr, _gs, _ge in G:
                if gidx != winner:
                    non_placeable.add(gidx)
            if winner != i:
                if placement_mode != "consistent_with_global":
                    non_placeable.add(i)
                    continue
                candidate_idx = winner
        elif placement_mode == "consistent_with_global":
            gidx, gipr, gs, ge = G[0]
            gavg = pooled_avg.get(gipr)
            if gavg is None or (not _within_global_length(interval_length(gs, ge), gavg, max_diff_frac)):
                non_placeable.add(gidx)
                stats.group_discard_global_inconsistent += 1
                continue

        # Global strict overlap gate again (candidate or winner from G).
        ipr_final, s_final, e_final = ordered[candidate_idx]
        if any(coordinate_overlap(s_final, e_final, s_p, e_p) for (_ipr_p, s_p, e_p) in placed_list):
            non_placeable.add(candidate_idx)
            continue

        placed_idx.add(candidate_idx)
        placed_list.append((ipr_final, s_final, e_final))
        stats.placed_total += 1

    # Deterministic output order.
    placed_sorted = sorted(placed_list, key=lambda t: (t[1], t[2], t[0]))
    _validate_placed_intervals(placed_sorted)
    return placed_sorted, stats


def _worker_init(pooled_avg: Dict[str, float], placement_mode: str, max_diff_frac: float) -> None:
    global _WORKER_POOLED_AVG
    global _WORKER_MODE
    global _WORKER_MAX_DIFF_FRAC
    _WORKER_POOLED_AVG = pooled_avg
    _WORKER_MODE = placement_mode
    _WORKER_MAX_DIFF_FRAC = max_diff_frac


def _worker_place(task: Tuple[int, str, List[Tuple[str, int, int]]]) -> Tuple[int, str, List[Tuple[str, int, int]], Dict[str, int]]:
    seq_idx, raw_seq_id, candidates = task
    placed, stats = place_domains_for_sequence(
        candidates,
        _WORKER_POOLED_AVG,
        placement_mode=_WORKER_MODE,
        max_diff_frac=_WORKER_MAX_DIFF_FRAC,
    )
    return (
        seq_idx,
        raw_seq_id,
        placed,
        {
            "candidates_total": stats.candidates_total,
            "placed_total": stats.placed_total,
            "group_discard_overlap_with_placed": stats.group_discard_overlap_with_placed,
            "group_discard_single_not_double": stats.group_discard_single_not_double,
            "group_discard_global_inconsistent": stats.group_discard_global_inconsistent,
        },
    )


def process_pass2_sequential(
    dat_file: str,
    ipr_to_members: Dict[str, List[str]],
    passing_iprs: Set[str],
    pooled_avg: Dict[str, float],
    max_diff_frac: float,
    progress_every: int,
    placement_mode: str = "default",
) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Pass2Counters]:
    members_by_ipr: Dict[str, Set[str]] = {ipr: set(members) for ipr, members in ipr_to_members.items()}
    domain_dict: Dict[str, List[Tuple[str, int, int]]] = {}
    ctr = Pass2Counters()
    t0 = time.perf_counter()

    current_seq: Optional[str] = None
    current_candidates: List[Tuple[str, int, int]] = []

    def flush_sequence() -> None:
        nonlocal current_seq, current_candidates
        if current_seq is None:
            return
        ctr.sequences_seen += 1
        if not current_candidates:
            return
        ctr.sequences_with_candidates += 1
        placed, pst = place_domains_for_sequence(
            current_candidates,
            pooled_avg,
            placement_mode=placement_mode,
            max_diff_frac=max_diff_frac,
        )
        ctr.candidates_total += pst.candidates_total
        ctr.placed_total += pst.placed_total
        ctr.group_discard_overlap_with_placed += pst.group_discard_overlap_with_placed
        ctr.group_discard_single_not_double += pst.group_discard_single_not_double
        ctr.group_discard_global_inconsistent += pst.group_discard_global_inconsistent
        if not placed:
            return
        sid = simplify_id(current_seq)
        if sid in domain_dict:
            ctr.simplified_id_collisions += 1
        domain_dict[sid] = placed
        ctr.sequences_output += 1

    for line, bytes_read, total_bytes in iter_lines_with_metrics(dat_file):
            ctr.total_lines += 1
            parsed = parse_line_full(line)
            if parsed is None:
                ctr.parse_fail_lines += 1
                continue
            seq_id, ipr, member, start, stop = parsed
            if start > stop:
                ctr.invalid_interval_lines += 1
                continue

            if current_seq is None:
                current_seq = seq_id
            elif seq_id != current_seq:
                flush_sequence()
                current_seq = seq_id
                current_candidates = []

            if ipr in passing_iprs and (not members_by_ipr or member in members_by_ipr.get(ipr, set())):
                current_candidates.append((ipr, start, stop))
                ctr.eligible_rows += 1

            if progress_every > 0 and (ctr.total_lines % progress_every) == 0:
                log_progress(
                    stage="pass2",
                    started_at=t0,
                    lines=ctr.total_lines,
                    eligible_rows=ctr.eligible_rows,
                    bytes_read=bytes_read,
                    total_bytes=total_bytes,
                )
                print(
                    f"[pass2] seq_seen={ctr.sequences_seen:,} seq_out={ctr.sequences_output:,}",
                    flush=True,
                )

    flush_sequence()
    return domain_dict, ctr


def process_pass2_parallel(
    dat_file: str,
    ipr_to_members: Dict[str, List[str]],
    passing_iprs: Set[str],
    pooled_avg: Dict[str, float],
    max_diff_frac: float,
    workers: int,
    queue_size: int,
    progress_every: int,
    placement_mode: str = "default",
) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Pass2Counters]:
    members_by_ipr: Dict[str, Set[str]] = {ipr: set(members) for ipr, members in ipr_to_members.items()}
    domain_dict: Dict[str, List[Tuple[str, int, int]]] = {}
    ctr = Pass2Counters()
    t0 = time.perf_counter()

    current_seq: Optional[str] = None
    current_candidates: List[Tuple[str, int, int]] = []
    submitted_count = 0
    next_flush_idx = 0
    pending_results: Dict[int, Tuple[str, List[Tuple[str, int, int]], Dict[str, int]]] = {}

    def flush_ready() -> None:
        nonlocal next_flush_idx
        while next_flush_idx in pending_results:
            raw_seq_id, placed, st = pending_results.pop(next_flush_idx)
            ctr.candidates_total += st["candidates_total"]
            ctr.placed_total += st["placed_total"]
            ctr.group_discard_overlap_with_placed += st["group_discard_overlap_with_placed"]
            ctr.group_discard_single_not_double += st["group_discard_single_not_double"]
            ctr.group_discard_global_inconsistent += st["group_discard_global_inconsistent"]
            if placed:
                sid = simplify_id(raw_seq_id)
                if sid in domain_dict:
                    ctr.simplified_id_collisions += 1
                domain_dict[sid] = placed
                ctr.sequences_output += 1
            next_flush_idx += 1

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(pooled_avg, placement_mode, max_diff_frac),
    ) as ex:
        futures = {}

        def submit_sequence(seq_id: str, cand: List[Tuple[str, int, int]]) -> None:
            nonlocal submitted_count
            fut = ex.submit(_worker_place, (submitted_count, seq_id, cand))
            futures[fut] = submitted_count
            submitted_count += 1

        for line, bytes_read, total_bytes in iter_lines_with_metrics(dat_file):
                ctr.total_lines += 1
                parsed = parse_line_full(line)
                if parsed is None:
                    ctr.parse_fail_lines += 1
                    continue
                seq_id, ipr, member, start, stop = parsed
                if start > stop:
                    ctr.invalid_interval_lines += 1
                    continue

                if current_seq is None:
                    current_seq = seq_id
                elif seq_id != current_seq:
                    ctr.sequences_seen += 1
                    if current_candidates:
                        ctr.sequences_with_candidates += 1
                        submit_sequence(current_seq, current_candidates)
                        current_candidates = []
                        # Bounded in-flight futures.
                        while len(futures) >= max(1, queue_size):
                            done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                            for d in done:
                                _idx, raw_id, placed, st = d.result()
                                pending_results[_idx] = (raw_id, placed, st)
                                futures.pop(d, None)
                            flush_ready()
                    current_seq = seq_id

                if ipr in passing_iprs and (not members_by_ipr or member in members_by_ipr.get(ipr, set())):
                    current_candidates.append((ipr, start, stop))
                    ctr.eligible_rows += 1

                if progress_every > 0 and (ctr.total_lines % progress_every) == 0:
                    log_progress(
                        stage="pass2-par",
                        started_at=t0,
                        lines=ctr.total_lines,
                        eligible_rows=ctr.eligible_rows,
                        bytes_read=bytes_read,
                        total_bytes=total_bytes,
                    )
                    print(
                        f"[pass2-par] submitted={submitted_count:,} seq_out={ctr.sequences_output:,}",
                        flush=True,
                    )

        if current_seq is not None:
            ctr.sequences_seen += 1
            if current_candidates:
                ctr.sequences_with_candidates += 1
                submit_sequence(current_seq, current_candidates)

        # Drain remaining futures
        while futures:
            done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for d in done:
                _idx, raw_id, placed, st = d.result()
                pending_results[_idx] = (raw_id, placed, st)
                futures.pop(d, None)
            flush_ready()

    flush_ready()
    return domain_dict, ctr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build InterPro GT: map from .dat → pass-1 global 10% filter → passing_iprs.txt → "
            "pass-2 placement with consistent-with-global (empty member map)."
        )
    )
    p.add_argument(
        "--dat-file",
        required=True,
        help="Path to protein2ipr.dat (whitespace-delimited; optionally .gz).",
    )
    p.add_argument(
        "--ipr-map-pkl",
        default=None,
        help="Optional precomputed IPR->members pickle. If absent, map is built from --dat-file.",
    )
    p.add_argument(
        "--save-map-pkl",
        default=None,
        help="Optional output path to save built map when --ipr-map-pkl is not provided.",
    )
    p.add_argument(
        "--passing-iprs-out",
        required=True,
        help="Output txt: ipr_id\\tglobal_avg_len (pass-1).",
    )
    p.add_argument(
        "--domain-dict-out",
        required=True,
        help="Output pickle for final domain dict.",
    )
    p.add_argument(
        "--max-diff-frac",
        type=float,
        default=0.10,
        help="All-pairs max fractional difference threshold (pass-1) and global length check (pass-2).",
    )
    p.add_argument("--workers", type=int, default=1, help="Workers for pass-2 per-sequence placement.")
    p.add_argument("--queue-size", type=int, default=128, help="Max in-flight sequence tasks in parallel pass-2.")
    p.add_argument("--progress-every", type=int, default=1_000_000, help="Progress log interval in lines (0 disables).")
    p.add_argument("--precision", type=int, default=6, help="Decimal precision for passing_iprs.txt global averages.")
    p.add_argument("--report-json", default=None, help="Optional JSON report output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    workers = max(1, int(args.workers))
    queue_size = max(1, int(args.queue_size))
    progress_every = max(0, int(args.progress_every))
    placement_mode = "consistent_with_global"

    print("[stage1] loading/building IPR->members map...", flush=True)
    ipr_map_pass1, map_stats = load_or_build_ipr_map(
        dat_file=args.dat_file,
        ipr_map_pkl=args.ipr_map_pkl,
        save_map_pkl=args.save_map_pkl,
        progress_every=progress_every,
    )
    print(f"[stage1] iprs_with_pfam={len(ipr_map_pass1):,} source={map_stats.get('source', 'unknown')}", flush=True)

    print("[stage2] applying global 10% all-pairs filter...", flush=True)
    passing_iprs, pooled_avg, pass1 = run_pass1_filter_iprs(
        dat_file=args.dat_file,
        ipr_to_members=ipr_map_pass1,
        max_diff_frac=float(args.max_diff_frac),
        progress_every=progress_every,
    )
    write_passing_iprs_txt(args.passing_iprs_out, pooled_avg, precision=int(args.precision))
    print(f"[stage2] passing_iprs={len(passing_iprs):,} wrote={args.passing_iprs_out}", flush=True)

    ipr_map_pass2: Dict[str, List[str]] = {}
    print("[stage3] building final domain dict (consistent-with-global)...", flush=True)
    if workers <= 1:
        domain_dict, pass2 = process_pass2_sequential(
            dat_file=args.dat_file,
            ipr_to_members=ipr_map_pass2,
            passing_iprs=passing_iprs,
            pooled_avg=pooled_avg,
            max_diff_frac=float(args.max_diff_frac),
            progress_every=progress_every,
            placement_mode=placement_mode,
        )
    else:
        domain_dict, pass2 = process_pass2_parallel(
            dat_file=args.dat_file,
            ipr_to_members=ipr_map_pass2,
            passing_iprs=passing_iprs,
            pooled_avg=pooled_avg,
            max_diff_frac=float(args.max_diff_frac),
            workers=workers,
            queue_size=queue_size,
            progress_every=progress_every,
            placement_mode=placement_mode,
        )

    ensure_parent_dir(args.domain_dict_out)
    with open(args.domain_dict_out, "wb") as fh:
        pickle.dump(domain_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[stage3] wrote domain_dict keys={len(domain_dict):,} -> {args.domain_dict_out}", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        (
            "[summary] "
            f"pass1_lines={pass1.total_lines:,} pass2_lines={pass2.total_lines:,} "
            f"pass1_eligible={pass1.eligible_rows:,} pass2_eligible={pass2.eligible_rows:,} "
            f"placed_total={pass2.placed_total:,} collisions={pass2.simplified_id_collisions:,} "
            f"elapsed_s={elapsed:.2f}"
        ),
        flush=True,
    )

    if args.report_json:
        report = {
            "inputs": {
                "dat_file": args.dat_file,
                "ipr_map_pkl": args.ipr_map_pkl,
                "save_map_pkl": args.save_map_pkl,
                "placement_mode": placement_mode,
                "passing_iprs_out": args.passing_iprs_out,
                "max_diff_frac": float(args.max_diff_frac),
                "workers": workers,
                "queue_size": queue_size,
                "progress_every": progress_every,
                "precision": int(args.precision),
            },
            "outputs": {
                "passing_iprs_out": args.passing_iprs_out,
                "domain_dict_out": args.domain_dict_out,
            },
            "map_stats": map_stats,
            "pass1": {
                "total_lines": pass1.total_lines,
                "parse_fail_lines": pass1.parse_fail_lines,
                "invalid_interval_lines": pass1.invalid_interval_lines,
                "eligible_rows": pass1.eligible_rows,
                "passing_iprs": len(passing_iprs),
            },
            "pass2": {
                "total_lines": pass2.total_lines,
                "parse_fail_lines": pass2.parse_fail_lines,
                "invalid_interval_lines": pass2.invalid_interval_lines,
                "eligible_rows": pass2.eligible_rows,
                "sequences_seen": pass2.sequences_seen,
                "sequences_with_candidates": pass2.sequences_with_candidates,
                "sequences_output": pass2.sequences_output,
                "candidates_total": pass2.candidates_total,
                "placed_total": pass2.placed_total,
                "group_discard_overlap_with_placed": pass2.group_discard_overlap_with_placed,
                "group_discard_single_not_double": pass2.group_discard_single_not_double,
                "group_discard_global_inconsistent": pass2.group_discard_global_inconsistent,
                "simplified_id_collisions": pass2.simplified_id_collisions,
            },
            "elapsed_seconds": elapsed,
        }
        ensure_parent_dir(args.report_json)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"[summary] wrote report json -> {args.report_json}", flush=True)


if __name__ == "__main__":
    main()
