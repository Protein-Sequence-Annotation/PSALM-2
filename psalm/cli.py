from __future__ import annotations

import argparse
from typing import Optional

from psalm import __version__ as PSALM_VERSION
from psalm.psalm_model import PSALM
from psalm.terminal_ui import (
    TERMINAL_WIDTH,
    center_text,
    divider,
    frame_bottom,
    frame_line,
    frame_top,
    kv_line,
    section_header,
)

DEFAULT_MODEL_NAME = "ProteinSequenceAnnotation/PSALM-2"
DEFAULT_FAST_MAX_BATCH_SIZE = 4096
DEFAULT_FAST_MAX_QUEUE_SIZE = 128


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.default in (None, False, argparse.SUPPRESS):
            return help_text
        return f"{help_text} (default: %(default)s)"


def _print_startup_banner() -> None:
    art = [
        "██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗",
        "██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║",
        "██████╔╝███████╗███████║██║     ██╔████╔██║",
        "██╔═══╝ ╚════██║██╔══██║██║     ██║╚██╔╝██║",
        "██║     ███████║██║  ██║███████╗██║ ╚═╝ ██║",
        "╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝",
    ]
    print(frame_top())
    for line in art:
        print(frame_line(line))
    print(frame_line("Protein Sequence Annotation using a Language Model"))
    print(frame_bottom())
    print(center_text(f"PSALM CLI (v{PSALM_VERSION})"))


def _format_model_source(source: str | None) -> str:
    mapping = {
        "local_path": "local path",
        "repo_models_dir": "repo models directory",
        "hf_cache": "Hugging Face cache",
        "hf_download": "Hugging Face download",
    }
    return mapping.get(str(source), str(source))


def _uses_fast_mode(args: argparse.Namespace) -> bool:
    return bool(args.fasta is not None and not getattr(args, "serial", False) and not args.verbose)


def add_scan_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "-s",
        default=None,
        metavar="SEQUENCE",
        dest="sequence",
        help="Raw amino-acid sequence string.",
    )
    input_group.add_argument(
        "-f",
        default=None,
        metavar="FASTA",
        dest="fasta",
        help="Path to a FASTA file containing one or more protein sequences.",
    )

    scoring_group = parser.add_argument_group("scoring")
    scoring_group.add_argument(
        "-T",
        type=float,
        default=0.85,
        metavar="SCORE",
        dest="score_thresh",
        help="Minimum CBM score required to keep a domain hit.",
    )
    scoring_group.add_argument(
        "-b",
        type=int,
        default=64,
        metavar="BEAM",
        dest="beam_size",
        help="Beam size used during decoding.",
    )
    scoring_group.add_argument(
        "-Z",
        type=float,
        default=None,
        metavar="DATASET",
        dest="dataset_size",
        help="Dataset size used in E-value scaling. If omitted, PSALM uses 1 for --sequence or the number of FASTA records for --fasta.",
    )
    scoring_group.add_argument(
        "-E",
        type=float,
        default=0.1,
        metavar="EVALUE",
        dest="evalue_thresh",
        help="Keep only domains with E-value less than or equal to this threshold.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--to-tsv",
        default=None,
        metavar="PATH",
        help="Write machine-readable hit output to a TSV file.",
    )
    output_group.add_argument(
        "--to-txt",
        default=None,
        metavar="PATH",
        help="Write the human-readable terminal report to a text file.",
    )

    display_group = parser.add_argument_group("display")
    display_group.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable the extended refinement pass.",
    )
    display_group.add_argument(
        "-v",
        action="store_true",
        dest="verbose",
        help="Show detailed per-domain tables and alignment output.",
    )
    display_group.add_argument(
        "-q",
        action="store_true",
        dest="quiet",
        help="Suppress scan result output in the terminal. Startup and status messages still print, and multi-sequence scans still show their progress bar.",
    )

    fast_group = parser.add_argument_group("fast scan")
    fast_group.add_argument(
        "--fast",
        action="store_true",
        help="Use batched FASTA scanning with optional CPU decode helpers. This is the default for FASTA input unless --serial is passed.",
    )
    fast_group.add_argument(
        "--serial",
        action="store_true",
        help="Use the legacy serial FASTA scan path instead of the default fast batching.",
    )
    fast_group.add_argument(
        "--sort",
        action="store_true",
        help="Sort FASTA sequences from longest to shortest before fast-mode batching.",
    )
    fast_group.add_argument(
        "-c",
        "--cpu-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of CPU decode helper processes for fast FASTA scans. "
            "Default behavior is equivalent to 0 unless reusing an existing warmed shell pool."
        ),
    )
    fast_group.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_FAST_MAX_BATCH_SIZE,
        metavar="TOKENS",
        help="Maximum fast-mode embedding batch size budget in tokens/amino acids.",
    )
    fast_group.add_argument(
        "--max-queue-size",
        type=int,
        default=DEFAULT_FAST_MAX_QUEUE_SIZE,
        metavar="N",
        help="Maximum fast-mode decode queue size in sequences.",
    )
    fast_group.add_argument(
        "--adaptive-fast",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def validate_scan_args(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None) -> None:
    def _fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if args.sequence is None and args.fasta is None:
        _fail("No input provided. Try `psalm-scan -h` for usage and examples.")
    if args.sequence is not None and args.fasta is not None:
        _fail(
            "Provide exactly one of -s or -f (not both). "
            "Try `psalm-scan -h` for usage and examples."
        )
    if args.fast and args.serial:
        _fail("Choose at most one of `--fast` or `--serial`.")
    if args.fast and args.sequence is not None:
        _fail("`--fast` requires `-f/--fasta` input and does not support `-s/--sequence`.")
    if args.serial and args.sequence is not None:
        _fail("`--serial` is only available with `-f/--fasta`.")
    if args.fast and args.verbose:
        _fail("`--fast` matches the non-verbose report only. Omit `-v/--verbose`.")
    if args.cpu_workers is not None and args.cpu_workers < 0:
        _fail("`--cpu-workers` must be >= 0.")
    uses_fast_mode = _uses_fast_mode(args)
    if not uses_fast_mode and args.sort:
        _fail("`--sort` is only available with fast FASTA scans.")
    if not uses_fast_mode and args.cpu_workers is not None:
        _fail("`--cpu-workers` is only available with fast FASTA scans.")
    if args.max_batch_size <= 0:
        _fail("`--max-batch-size` must be > 0.")
    if not uses_fast_mode and args.max_batch_size != DEFAULT_FAST_MAX_BATCH_SIZE:
        _fail("`--max-batch-size` is only available with fast FASTA scans.")
    if args.max_queue_size <= 0:
        _fail("`--max-queue-size` must be > 0.")
    if not uses_fast_mode and args.max_queue_size != DEFAULT_FAST_MAX_QUEUE_SIZE:
        _fail("`--max-queue-size` is only available with fast FASTA scans.")
    if not uses_fast_mode and getattr(args, "adaptive_fast", False):
        _fail("`--adaptive-fast` is only available with fast FASTA scans.")


def build_scan_parser(
    *,
    prog: Optional[str] = None,
    add_help: bool = True,
    exit_on_error: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        add_help=add_help,
        description=(
            "Run one PSALM scan inside the interactive shell.\n"
            "Provide exactly one of -s or -f."
        ),
        epilog=(
            "Examples:\n"
            "  scan -s MSTNPKPQRIT...\n"
            "  scan -f path/to/proteins.fa\n"
            "  scan --sort -f path/to/proteins.fa -c 4 --max-queue-size 128\n"
            "  scan -f path/to/proteins.fa -E 1e-3 -v\n"
            "  scan -q -f path/to/proteins.fa --to-tsv hits.tsv\n"
            "  scan --serial -f path/to/proteins.fa\n"
            "  scan -f path/to/proteins.fa --to-tsv hits.tsv --to-txt report.txt\n"
            "  scan -f path/to/proteins.fa -T 0.20 -b 128"
        ),
        formatter_class=_HelpFormatter,
        exit_on_error=exit_on_error,
    )
    add_scan_arguments(parser)
    return parser


def load_model_with_startup(
    *,
    model_name: str,
    device: str,
    extra_setup_callback=None,
    show_banner: bool = True,
) -> PSALM:
    if show_banner:
        _print_startup_banner()
    print(section_header("LOADING"), flush=True)
    print(
        "   " + kv_line("Requested device:", device, label_width=18, width=TERMINAL_WIDTH - 3),
        flush=True,
    )
    print(
        "   " + kv_line("Model:", model_name, label_width=18, width=TERMINAL_WIDTH - 3),
        flush=True,
    )

    warming_section_printed = False

    def _status(msg: str) -> None:
        nonlocal warming_section_printed
        clean = msg.rstrip(".")
        if clean.lower().startswith("warmup"):
            if not warming_section_printed:
                print(section_header("WARMING UP"), flush=True)
                warming_section_printed = True
            print(f"   {clean}", flush=True)
        else:
            print(f"   {clean}", flush=True)

    model = PSALM(
        model_name=model_name,
        device=device,
        status_callback=_status,
    )

    print(section_header("SETUP"), flush=True)
    resolved_device = getattr(model, "resolved_device", None)
    if resolved_device is not None:
        print(
            "   " + kv_line("Resolved device:", str(resolved_device), label_width=18, width=TERMINAL_WIDTH - 3),
            flush=True,
        )
    model_source = _format_model_source(getattr(model, "model_source", None))
    if model_source not in {"None", ""}:
        print(
            "   " + kv_line("Source:", model_source, label_width=18, width=TERMINAL_WIDTH - 3),
            flush=True,
        )
    if extra_setup_callback is not None:
        setup_messages: list[str] = []
        extra_setup_callback(model, setup_messages.append)
        if setup_messages:
            print(section_header("FAST WORKERS"), flush=True)
            for msg in setup_messages:
                print(f"   {msg}", flush=True)
    if not getattr(model, "warmup_executed", False):
        if not warming_section_printed:
            print(section_header("WARMING UP"), flush=True)
            warming_section_printed = True
        print("   Warmup skipped", flush=True)
    print(section_header("READY"), flush=True)
    print(divider(), flush=True)
    return model


def run_scan_from_args(model: PSALM, args: argparse.Namespace):
    if _uses_fast_mode(args):
        return model.scan_fast(
            fasta=args.fasta,
            score_thresh=args.score_thresh,
            beam_size=args.beam_size,
            dataset_size=args.dataset_size,
            evalue_thresh=args.evalue_thresh,
            refine_extended=(not args.no_refine),
            to_tsv=args.to_tsv,
            to_txt=args.to_txt,
            sort=args.sort,
            cpu_workers=args.cpu_workers,
            max_batch_size=args.max_batch_size,
            max_queue_size=args.max_queue_size,
            adaptive_fast=args.adaptive_fast,
            _print_output=(not args.quiet),
            _show_progress=True,
        )
    return model.scan(
        sequence=args.sequence,
        fasta=args.fasta,
        score_thresh=args.score_thresh,
        beam_size=args.beam_size,
        dataset_size=args.dataset_size,
        evalue_thresh=args.evalue_thresh,
        refine_extended=(not args.no_refine),
        verbose=args.verbose,
        to_tsv=args.to_tsv,
        to_txt=args.to_txt,
        _print_output=(not args.quiet),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PSALM domain scanning for one sequence or a FASTA file.\n"
            "Use psalm for a persistent interactive shell (load once, scan many times)."
        ),
        epilog=(
            "Examples:\n"
            "  psalm-scan -f Q09870.fa\n"
            "  psalm-scan -v -f Q09870.fa --to-tsv out.tsv\n"
            "  psalm-scan -q --no-refine -f Q09870.fa\n"
            "  psalm-scan -f Q09870.fa -E 1e-3 -T 0.20\n"
            "  psalm-scan --sort -f proteins.fa -c 4 --max-queue-size 128\n"
            "  psalm-scan --serial -f proteins.fa\n"
            "  psalm -d auto\n"
            "  psalm -d auto -c 4\n"
            "  psalm help"
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model bundle path or Hugging Face repo id.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        help="Device to use: auto, cpu, mps, cuda, cuda:<index>.",
    )
    add_scan_arguments(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PSALM_VERSION}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_scan_args(args, parser)

    model = load_model_with_startup(
        model_name=args.model_name,
        device=args.device,
        show_banner=False,
    )
    run_scan_from_args(model, args)


if __name__ == "__main__":
    main()
