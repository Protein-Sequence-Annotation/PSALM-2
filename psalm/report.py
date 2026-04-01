from __future__ import annotations

import contextlib
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, TextIO, Tuple

from psalm.terminal_ui import TERMINAL_WIDTH, ellipsize, frame_top, kv_line, result_bottom, section_header


DomainTuple = Tuple[str, int, int, float, float, float, float, str]


@dataclass(frozen=True)
class HitRow:
    sequence: str
    evalue: float
    score: float
    pfam: str
    start: int
    stop: int
    model: str
    len_frac: float
    status: str
    bit_score: float
    bias: float
    domain: DomainTuple


def capitalize_status(status: str) -> str:
    if not status:
        return status
    return status[:1].upper() + status[1:]


def format_evalue(value: float) -> str:
    return f"{value:.1e}"


def format_score(value: float) -> str:
    return f"{value:.2f}"


def build_hit_rows(
    *,
    seq_id: str,
    domains: List[DomainTuple],
    dataset_size: float,
    evalue_for_domain: Callable[[DomainTuple, float], float],
    label_mapping: dict,
) -> List[HitRow]:
    rows: List[HitRow] = []
    for domain in domains:
        pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status = domain
        rows.append(
            HitRow(
                sequence=seq_id,
                evalue=evalue_for_domain(domain, dataset_size),
                score=cbm_score,
                pfam=pfam,
                start=start,
                stop=stop,
                model=label_mapping[pfam]["family_name"],
                len_frac=len_ratio,
                status=capitalize_status(status),
                bit_score=bit_score,
                bias=bias,
                domain=domain,
            )
        )
    rows.sort(key=lambda row: (row.evalue, -row.score, row.start, row.stop, row.pfam))
    return rows


def build_hit_rows_from_evalues(
    *,
    seq_id: str,
    domains: List[DomainTuple],
    evalues: List[float],
    label_mapping: dict,
) -> List[HitRow]:
    rows: List[HitRow] = []
    for domain, evalue in zip(domains, evalues):
        pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status = domain
        rows.append(
            HitRow(
                sequence=seq_id,
                evalue=evalue,
                score=cbm_score,
                pfam=pfam,
                start=start,
                stop=stop,
                model=label_mapping[pfam]["family_name"],
                len_frac=len_ratio,
                status=capitalize_status(status),
                bit_score=bit_score,
                bias=bias,
                domain=domain,
            )
        )
    rows.sort(key=lambda row: (row.evalue, -row.score, row.start, row.stop, row.pfam))
    return rows


def write_hits_tsv(path: str, hit_rows: List[HitRow]) -> None:
    header = ["Sequence", "E-value", "Score", "Pfam", "Start", "Stop", "Model", "Len Frac", "Status"]
    with open(path, "w", encoding="utf-8") as handle:
        write_hits_tsv_header(handle, header=header)
        append_hits_tsv_rows(handle, hit_rows)


def write_hits_tsv_header(handle: TextIO, *, header: List[str] | None = None) -> None:
    columns = header or ["Sequence", "E-value", "Score", "Pfam", "Start", "Stop", "Model", "Len Frac", "Status"]
    handle.write("\t".join(columns) + "\n")


def append_hits_tsv_rows(handle: TextIO, hit_rows: List[HitRow]) -> None:
    for row in hit_rows:
        values = [
            row.sequence,
            format_evalue(row.evalue),
            format_score(row.score),
            row.pfam,
            str(row.start),
            str(row.stop),
            row.model,
            f"{row.len_frac:.2f}",
            row.status,
        ]
        handle.write("\t".join(values) + "\n")


def render_scan_output(
    *,
    sequence: str,
    seq_id: str,
    hit_rows: List[HitRow],
    score_thresh: float,
    evalue_thresh: float,
    score_pass: int,
    total_domains_raw: int,
    total_fams: int,
    kept_families: int,
    domains_original_scored: List[DomainTuple],
    gamma,
    best_path,
    verbose: bool,
    refine_extended: bool,
    beam_size: int,
    inf_dt: float,
    dec_dt: float,
    model_label: str,
    model_source_label: str,
    device_label: str,
    psalm_version: str,
    label_mapping: dict,
    output: Optional[TextIO] = None,
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stream = output if output is not None else sys.stdout
    with contextlib.redirect_stdout(stream):
        print(frame_top())
        print(section_header("QUERY"))
        print(kv_line("ID", seq_id))
        print(kv_line("Length", f"{len(sequence)} aa"))
        print(kv_line("Families", f"{kept_families}/{total_fams} families passed"))
        print(kv_line("Score Filter", f">={score_thresh:.2f}: {score_pass}/{total_domains_raw} domains passed"))
        print(kv_line("E-value Filter", f"<={format_evalue(evalue_thresh)}: {len(hit_rows)}/{score_pass} domains passed"))
        print(kv_line("Hits", f"{len(hit_rows)} domains"))
        print()

        print(section_header("HITS  (Pfam domain hits)"))
        ev_w = 12
        score_w = 7
        pfam_w = 7
        start_w = 5
        stop_w = 5
        model_w = 19
        len_w = 8
        header = (
            f"{'E-value':>{ev_w}}  {'Score':>{score_w}}  {'Pfam':<{pfam_w}}  "
            f"{'Start':>{start_w}}  {'Stop':>{stop_w}}  {'Model':<{model_w}}  {'Len Frac':>{len_w}}"
        )
        print(header)
        print("─" * min(TERMINAL_WIDTH, len(header)))
        if not hit_rows:
            print("(no domains passed score/E-value filters)")
        for row in hit_rows:
            print(
                f"{format_evalue(row.evalue):>{ev_w}}  {format_score(row.score):>{score_w}}  {row.pfam:<{pfam_w}}  "
                f"{row.start:>{start_w}}  {row.stop:>{stop_w}}  {ellipsize(row.model, model_w):<{model_w}}  "
                f"{row.len_frac:>{len_w}.2f}"
            )
        print()

        print(section_header("DOMAINS  (annotation + alignments)"))
        if not hit_rows:
            print("No domain annotations after filtering.")
            print()
        elif not verbose:
            print("Run with -v/--verbose to show per-domain alignments.")
            print()
        else:
            family_hits: OrderedDict[str, List[HitRow]] = OrderedDict()
            for row in hit_rows:
                family_hits.setdefault(row.pfam, []).append(row)

            nested_components = {}
            for row in hit_rows:
                pfam, start_combined, stop_combined, *_rest, status = row.domain
                if status == "full (merged)":
                    components = []
                    for pfam_orig, start_orig, stop_orig, cbm_o, bit_o, len_o, bias_o, status_o in domains_original_scored:
                        if (
                            pfam_orig == pfam
                            and start_orig >= start_combined
                            and stop_orig <= stop_combined
                            and status_o in [
                                "partial (no stop)",
                                "partial (no start)",
                                "partial (no start or stop)",
                            ]
                        ):
                            components.append(
                                (start_orig, stop_orig, cbm_o, bit_o, len_o, bias_o, capitalize_status(status_o))
                            )
                    if len(components) >= 2:
                        components.sort(key=lambda item: item[0])
                        nested_components[(pfam, start_combined, stop_combined)] = components

            for pfam, rows in family_hits.items():
                print(f"{pfam}  {ellipsize(label_mapping[pfam]['family_name'], 24)}")
                print(ellipsize(label_mapping[pfam]["family_desc"], max(0, TERMINAL_WIDTH - 2)))
                print()
                print("    E-value    Score  Start   Stop  Len Frac  Status")
                print("────────────  ─────  ─────  ─────  ────────  ─────────────────────────────────")

                for row in rows:
                    print(
                        f"{format_evalue(row.evalue):>12}  {format_score(row.score):>5}  {row.start:>5}  {row.stop:>5}  "
                        f"{row.len_frac:>8.2f}  {ellipsize(row.status, 32)}"
                    )
                print()

                indent = " " * 5
                for i, row in enumerate(rows, start=1):
                    pfam_name, start, stop, _cbm_score, bit_score, len_ratio, bias, _status = row.domain
                    nested_key = (pfam_name, start, stop)
                    if nested_key in nested_components:
                        components = nested_components[nested_key]
                        print(
                            f"Domain {i}: score={row.score:.2f} bit={bit_score:.2f} "
                            f"bias={bias:.2f} len={len_ratio:.2f} [nested]"
                        )

                        for comp_idx, (comp_start, comp_stop, cbm_c, bit_c, len_c, bias_c, status_c) in enumerate(components, 1):
                            print(
                                f"  Component {comp_idx}: score={cbm_c:.2f} bit={bit_c:.2f} "
                                f"bias={bias_c:.2f} len={len_c:.2f} {status_c}"
                            )
                            region_seq = sequence[comp_start - 1:comp_stop]
                            length = len(region_seq)

                            prob_chars = []
                            for offset in range(length):
                                t = comp_start - 1 + offset
                                p = float(gamma[t, best_path[t]]) if gamma is not None else 0.0
                                if p < 0.05:
                                    pc = "0"
                                elif p < 0.95:
                                    pc = str(min(9, int(p * 10)))
                                else:
                                    pc = "*"
                                prob_chars.append(pc)
                            prob_str = "".join(prob_chars)

                            for off in range(0, length, 70):
                                seq_chunk = region_seq[off:off + 70]
                                p_chunk = prob_str[off:off + 70]
                                abs_s = comp_start + off
                                abs_e = comp_start + off + len(seq_chunk) - 1
                                print(f"{abs_s:>4} {seq_chunk:<70} {abs_e:>4}")
                                print(indent + p_chunk)
                            print()
                    else:
                        print(
                            f"Domain {i}: score={row.score:.2f} bit={bit_score:.2f} "
                            f"bias={bias:.2f} len={len_ratio:.2f}"
                        )
                        region_seq = sequence[start - 1:stop]
                        length = len(region_seq)

                        prob_chars = []
                        for offset in range(length):
                            t = start - 1 + offset
                            p = float(gamma[t, best_path[t]]) if gamma is not None else 0.0
                            if p < 0.05:
                                pc = "0"
                            elif p < 0.95:
                                pc = str(min(9, int(p * 10)))
                            else:
                                pc = "*"
                            prob_chars.append(pc)
                        prob_str = "".join(prob_chars)

                        for off in range(0, length, 70):
                            seq_chunk = region_seq[off:off + 70]
                            p_chunk = prob_str[off:off + 70]
                            abs_s = start + off
                            abs_e = start + off + len(seq_chunk) - 1
                            print(f"{abs_s:>4} {seq_chunk:<70} {abs_e:>4}")
                            print(indent + p_chunk)
                        print()
                print()

        print(section_header("RUN"))
        print(kv_line("Date", stamp))
        print(kv_line("PSALM", psalm_version))
        print(kv_line("Model", model_label))
        print(kv_line("Source", model_source_label))
        print(kv_line("Device", device_label))
        print(kv_line("Refinement", "ON" if refine_extended else "OFF"))
        print(kv_line("Beam", str(beam_size)))
        print(kv_line("Embed", f"{inf_dt * 1000:.2f} ms"))
        print(kv_line("Decode", f"{dec_dt * 1000:.2f} ms"))
        print(result_bottom())
