from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm

if "TRANSFORMERS_CACHE" in os.environ:
    cache_path = Path(os.environ["TRANSFORMERS_CACHE"]).expanduser()
    if "HF_HOME" not in os.environ:
        hf_home = cache_path.parent if cache_path.name == "transformers" else cache_path
        os.environ["HF_HOME"] = str(hf_home)
    os.environ.pop("TRANSFORMERS_CACHE", None)

from transformers import AutoConfig, AutoTokenizer

from psalm.config import get_model_config
from psalm.inference.cbm_score import add_cbm_scores
from psalm.inference.decoder import annotate_domains
from psalm.inference.evalue import EValueCurve, load_default_curve
from psalm.report import (
    DomainTuple,
    HitRow,
    build_hit_rows,
    build_hit_rows_from_evalues,
    render_scan_output,
    write_hits_tsv,
)

SCAN_PROGRESS_BAR_FORMAT = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or str(device).strip().lower() == "auto":
        return _auto_device()

    try:
        requested = torch.device(str(device).strip().lower())
    except Exception as exc:
        raise ValueError(
            f"Invalid device '{device}'. Use one of: auto, cpu, mps, cuda, cuda:<index>."
        ) from exc

    if requested.type == "cuda" and not torch.cuda.is_available():
        fallback = _auto_device()
        warnings.warn(
            f"Requested device '{device}' is unavailable; falling back to '{fallback.type}'."
        )
        return fallback

    if requested.type == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            fallback = _auto_device()
            warnings.warn(
                f"Requested device '{device}' is unavailable; falling back to '{fallback.type}'."
            )
            return fallback

    return requested


def _resolve_model_dir_with_source(model_name: str) -> Tuple[Path, str]:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve(), "local_path"
    repo_root = Path(__file__).resolve().parents[1]
    local_candidate = repo_root / "models" / model_name
    if local_candidate.exists():
        return local_candidate.resolve(), "repo_models_dir"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model path not found locally and huggingface_hub is not available."
        ) from exc
    allow_patterns = [
        "config.json",
        "model.safetensors",
        "model.bin",
        "label_mapping.json",
        "pfam_label_mapping.json",
        "transitions.json",
        "pfam_seed_transitions.json",
        "score.cbm",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "tokenizer.json",
    ]
    try:
        cache_dir = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        return Path(cache_dir), "hf_cache"
    except Exception:
        cache_dir = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
        )
        return Path(cache_dir), "hf_download"


def _resolve_model_dir(model_name: str) -> Path:
    path, _ = _resolve_model_dir_with_source(model_name)
    return path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _load_label_mapping(model_dir: Path) -> dict:
    for name in ("label_mapping.json", "pfam_label_mapping.json"):
        path = model_dir / name
        if path.exists():
            return _load_json(path)
    raise FileNotFoundError("label_mapping.json not found in model bundle.")


def _load_transitions(model_dir: Path) -> dict:
    for name in ("transitions.json", "pfam_seed_transitions.json"):
        path = model_dir / name
        if path.exists():
            raw = _load_json(path)
            return {
                int(i): {int(j): float(lp) for j, lp in row.items()}
                for i, row in raw.items()
            }
    raise FileNotFoundError("transitions.json not found in model bundle.")


def _infer_output_size(label_mapping: dict) -> int:
    max_idx = -1
    for pfam, info in label_mapping.items():
        if isinstance(info, dict):
            max_idx = max(max_idx, int(info["start"]), int(info["middle"]), int(info["stop"]))
        else:
            max_idx = max(max_idx, int(info))
    return max_idx + 1


def _load_state_dict(model_dir: Path) -> dict:
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load model.safetensors."
            ) from exc
        return load_file(str(weights_path))
    bin_path = model_dir / "model.bin"
    if bin_path.exists():
        try:
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError("model.safetensors or model.bin not found in model bundle.")


def _load_model_state_with_fallback(model: nn.Module, state_dict: dict, context: str) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as err:
        msg = str(err)
        known_key = "esm_model.embeddings.position_embeddings.weight"
        if known_key not in msg:
            raise

    incompatible = model.load_state_dict(state_dict, strict=False)
    dropped = [k for k in incompatible.unexpected_keys if k == known_key]
    remaining_unexpected = [k for k in incompatible.unexpected_keys if k != known_key]
    if remaining_unexpected:
        raise RuntimeError(
            f"{context}: unexpected checkpoint keys remain after fallback: {remaining_unexpected}"
        )
    if incompatible.missing_keys:
        warnings.warn(
            f"{context}: missing keys during fallback load: {incompatible.missing_keys}"
        )
    # Intentionally suppress the known dropped-key warning by default
    # to keep CLI startup output clean.
    _ = dropped


def _get_psalm_version() -> str:
    try:
        return pkg_version("protein-sequence-annotation")
    except PackageNotFoundError:
        return "0+local"


def _format_model_source_display(source: Optional[str]) -> str:
    mapping = {
        "local_path": "Local path",
        "repo_models_dir": "Repo models directory",
        "hf_cache": "Hugging Face cache",
        "hf_download": "Hugging Face download",
    }
    key = str(source)
    return mapping.get(key, key)


def _load_cbm_model(model_dir: Path):
    cbm_path = model_dir / "score.cbm"
    if not cbm_path.exists():
        raise FileNotFoundError("score.cbm not found in model bundle.")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required for CBM scoring. Install catboost to use scan()."
        ) from exc
    model = CatBoostClassifier()
    model.load_model(str(cbm_path))
    return model


def _build_decoder_resources(label_mapping: dict, transitions: dict):
    none_label = label_mapping.get("None")
    if none_label is None:
        raise KeyError("label_mapping.json missing 'None' label.")

    inverse_label_mapping = {}
    for pfam, info in label_mapping.items():
        if pfam == "None":
            inverse_label_mapping[int(info)] = (pfam, None)
        else:
            inverse_label_mapping[int(info["start"])] = (pfam, "start")
            inverse_label_mapping[int(info["middle"])] = (pfam, "middle")
            inverse_label_mapping[int(info["stop"])] = (pfam, "stop")

    starts_full: List[int] = []
    middles_full: List[int] = []
    stops_full: List[int] = []
    for idx, (pf, role) in inverse_label_mapping.items():
        if role == "start":
            starts_full.append(idx)
        elif role == "middle":
            middles_full.append(idx)
        elif role == "stop":
            stops_full.append(idx)

    role_map = {
        "start": np.array(starts_full, dtype=np.int64),
        "middle": np.array(middles_full, dtype=np.int64),
        "stop": np.array(stops_full, dtype=np.int64),
    }

    E = sum(len(row) for row in transitions.values())
    from_list = np.empty(E, dtype=np.int64)
    to_list = np.empty(E, dtype=np.int64)
    lp_list = np.empty(E, dtype=np.float64)
    idx = 0
    for i, row in transitions.items():
        for j, lp in row.items():
            from_list[idx] = i
            to_list[idx] = j
            lp_list[idx] = lp
            idx += 1

    all_pfams = sorted({pf for pf, _ in inverse_label_mapping.values() if pf != "None"})
    pfam2id = {pf: i for i, pf in enumerate(all_pfams)}
    id_to_pfam = all_pfams

    C = max(inverse_label_mapping.keys()) + 1
    state_pfam_ids_full = np.full(C, -1, dtype=np.int64)
    state_role_ids_full = np.full(C, -1, dtype=np.int8)
    role2id = {"start": 0, "middle": 1, "stop": 2}
    for idx, (pf, role) in inverse_label_mapping.items():
        state_pfam_ids_full[idx] = pfam2id.get(pf, -1)
        state_role_ids_full[idx] = role2id.get(role, -1)

    return {
        "none_label": int(none_label),
        "label_mapping": label_mapping,
        "inverse_label_mapping": inverse_label_mapping,
        "role_map": role_map,
        "from_list": from_list,
        "to_list": to_list,
        "lp_list": lp_list,
        "id_to_pfam": id_to_pfam,
        "state_pfam_ids_full": state_pfam_ids_full,
        "state_role_ids_full": state_role_ids_full,
    }


def _resolve_use_fa(use_fa: bool, device: torch.device) -> bool:
    if not use_fa:
        return False
    if device.type != "cuda":
        warnings.warn("FlashAttention requires CUDA; disabling use_fa.")
        return False
    try:
        import faesm  # noqa: F401
    except ImportError:
        warnings.warn("faesm not installed; disabling use_fa.")
        return False
    return True


_VALID_SEQUENCE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_sequence_text(sequence: str, source: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if normalized == "":
        raise ValueError(f"{source} is empty after removing whitespace.")
    invalid_chars = sorted({ch for ch in normalized if ch not in _VALID_SEQUENCE_CHARS})
    if invalid_chars:
        joined = ", ".join(repr(ch) for ch in invalid_chars)
        raise ValueError(
            f"{source} contains invalid characters: {joined}. "
            "Sequences may contain letters only."
        )
    return normalized


def _load_fasta_records_checked(fasta: str) -> List[SeqRecord]:
    fasta_path = Path(fasta).expanduser()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta}")
    if not fasta_path.is_file():
        raise ValueError(f"FASTA path is not a file: {fasta}")
    try:
        with fasta_path.open("r", encoding="utf-8") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
    except Exception as exc:
        raise ValueError(f"Could not parse FASTA file: {fasta}") from exc
    if not records:
        raise ValueError(f"FASTA file is empty or contains no sequences: {fasta}")
    return records


def _iter_fasta_records_checked(fasta: str) -> Iterator[SeqRecord]:
    fasta_path = Path(fasta).expanduser()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta}")
    if not fasta_path.is_file():
        raise ValueError(f"FASTA path is not a file: {fasta}")

    saw_any = False
    try:
        with fasta_path.open("r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                saw_any = True
                yield record
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse FASTA file: {fasta}") from exc

    if not saw_any:
        raise ValueError(f"FASTA file is empty or contains no sequences: {fasta}")


def _count_fasta_records_checked(fasta: str) -> int:
    count = 0
    for _record in _iter_fasta_records_checked(fasta):
        count += 1
    return count


@dataclass
class _PreparedDecodeInputs:
    total_fams: int
    kept_families: int
    keep_full_cpu: torch.Tensor
    sub_emits_cpu: Optional[torch.Tensor]
    none_sub: Optional[int]


@dataclass
class _DecodedDomains:
    domains: List[DomainTuple]
    domains_original: List[DomainTuple]
    gamma: Optional[torch.Tensor]
    best_path: Optional[np.ndarray]
    dec_dt: float


@dataclass
class _FinalizedScanResult:
    domains_scored: List[DomainTuple]
    domains_original_scored: List[DomainTuple]
    hit_rows: List[HitRow]
    score_pass: int
    total_domains_raw: int


class PSALM(nn.Module):
    """
    PSALM integrates an ESM model with a single MLP head for Pfam-family classification.
    Supports both training (config-based) and inference from a model bundle.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ignore_label: int = -100,
        device: Optional[str] = None,
        use_fa: Optional[bool] = None,
        warmup: Optional[bool] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.ignore_label = ignore_label
        self.model_dir: Optional[Path] = None
        self.model_source: Optional[str] = None
        self.cbm_model = None
        self.evalue_curve: Optional[EValueCurve] = None
        self.use_fa = False
        self.warmup_executed = False
        self.resolved_device: Optional[torch.device] = None
        self.status_callback = status_callback
        self._pfam_ids_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        if model_name is None:
            cfg = get_model_config()
            model_name_cfg = cfg.model_name
            esm_cfg = AutoConfig.from_pretrained(model_name_cfg)
            if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings is not None:
                esm_cfg.max_position_embeddings = cfg.max_position_embeddings
            use_fa_cfg = bool(getattr(cfg, "use_fa", False))
            device_obj = _resolve_device(device)
            self.resolved_device = device_obj
            self.use_fa = _resolve_use_fa(use_fa_cfg, device_obj)
            if self.use_fa:
                from faesm.esm import FAEsmModel
                esm_cfg.use_fa = True
                self.esm_model = FAEsmModel.from_pretrained(
                    model_name_cfg, config=esm_cfg, add_pooling_layer=False
                )
            else:
                from transformers import EsmModel
                self.esm_model = EsmModel.from_pretrained(
                    model_name_cfg, config=esm_cfg, add_pooling_layer=False
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_cfg)
            families = cfg.output_size
            try:
                self.evalue_curve = load_default_curve()
            except Exception:
                self.evalue_curve = None
        else:
            self.model_dir, self.model_source = _resolve_model_dir_with_source(model_name)
            if self.status_callback is not None:
                self.status_callback("Model source resolved.")
            device_obj = _resolve_device(device)
            self.resolved_device = device_obj
            esm_cfg = AutoConfig.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            if use_fa is None:
                use_fa = bool(getattr(esm_cfg, "use_fa", False))
            self.use_fa = _resolve_use_fa(bool(use_fa), device_obj)
            if self.use_fa:
                from faesm.esm import FAEsmModel
                esm_cfg.use_fa = True
                self.esm_model = FAEsmModel(esm_cfg, add_pooling_layer=False)
            else:
                from transformers import EsmModel
                self.esm_model = EsmModel(esm_cfg, add_pooling_layer=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            label_mapping = _load_label_mapping(self.model_dir)
            families = _infer_output_size(label_mapping)

        hidden_dim = self.esm_model.config.hidden_size
        expand_dim = 2 * hidden_dim
        self.classes = families

        self.fc1 = nn.Linear(hidden_dim, expand_dim, bias=True)
        self.ln1 = nn.LayerNorm(expand_dim, self.esm_model.config.layer_norm_eps)
        self.fc2 = nn.Linear(expand_dim, expand_dim, bias=True)
        self.ln2 = nn.LayerNorm(expand_dim, self.esm_model.config.layer_norm_eps)
        self.fc3 = nn.Linear(expand_dim, families, bias=True)
        self.relu = nn.ReLU()

        for lin in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for ln in (self.ln1, self.ln2):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

        if model_name is None:
            if cfg.freeze_esm:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
                print("ESM parameters frozen")
            else:
                print("ESM parameters unfrozen for fine-tuning")

            esm_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
            classifier_params = sum(
                p.numel() for name, p in self.named_parameters()
                if p.requires_grad and not name.startswith("esm_model.")
            )
            print(f"Trainable parameters: ESM = {esm_params:,}, Classifier = {classifier_params:,}")
        else:
            state_dict = _load_state_dict(self.model_dir)
            _load_model_state_with_fallback(self, state_dict, context="inference bundle")
            resources = _build_decoder_resources(
                label_mapping, _load_transitions(self.model_dir)
            )
            self.none_label = resources["none_label"]
            self.label_mapping = resources["label_mapping"]
            self.inverse_label_mapping = resources["inverse_label_mapping"]
            self.role_map = resources["role_map"]
            self.from_list = resources["from_list"]
            self.to_list = resources["to_list"]
            self.lp_list = resources["lp_list"]
            self.id_to_pfam = resources["id_to_pfam"]
            self.state_pfam_ids_full = resources["state_pfam_ids_full"]
            self.state_role_ids_full = resources["state_role_ids_full"]
            self.cbm_model = _load_cbm_model(self.model_dir)
            self.evalue_curve = load_default_curve()
            self.to(device_obj)
            if warmup is None:
                warmup = True
            if warmup:
                if self.status_callback is not None:
                    self.status_callback("Warmup started.")
                self._warmup()
                self.warmup_executed = True
                if self.status_callback is not None:
                    self.status_callback("Warmup finished.")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        out = self.esm_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        reps = out.last_hidden_state

        x = self.fc1(reps)
        x = self.relu(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.ln2(x)

        logits = self.fc3(x)
        output = {"logits": logits}

        if labels is not None:
            C = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
            output["loss"] = loss_fct(logits.view(-1, C), labels.view(-1))

        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, device=None, ignore_label=-100):
        model = cls(ignore_label=ignore_label)
        weights_path = cls._resolve_checkpoint_path(checkpoint_dir)
        if weights_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load .safetensors checkpoints."
                ) from exc
            state_dict = load_file(weights_path)
        else:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
        _load_model_state_with_fallback(model, state_dict, context="checkpoint load")

        device_obj = _resolve_device(device)
        model = model.to(device_obj)
        return model

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path):
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        candidates = ["model.safetensors", "pytorch_model.bin"]
        for name in candidates:
            candidate = os.path.join(checkpoint_path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"No checkpoint file found in {checkpoint_path}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    def infer(self, sequence: str) -> torch.Tensor:
        if not hasattr(self, "tokenizer"):
            raise RuntimeError("Tokenizer not initialized. Load a model bundle first.")
        device = next(self.parameters()).device
        toks = self.tokenizer(sequence, return_tensors="pt").to(device)

        self.eval()
        with torch.no_grad():
            if self.use_fa and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(**toks)
            else:
                out = self(**toks)

        logits = out["logits"]
        logits = logits[:, 1:-1, :]
        logits = logits.squeeze(0)
        return logits.float()

    def batch_infer(
        self,
        sequences: List[str],
        max_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        _ = max_batch_size
        device = next(self.parameters()).device
        self.eval()
        toks = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        with torch.no_grad():
            if self.use_fa and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(**toks)
            else:
                out = self(**toks)

            return torch.log_softmax(out["logits"], dim=-1)

    def _warmup(self) -> None:
        warmup_path = Path(__file__).resolve().parent / "inference" / "warmup.fasta"
        if not warmup_path.exists():
            return
        try:
            records = list(SeqIO.parse(str(warmup_path), "fasta"))
        except Exception:
            return
        for idx, rec in enumerate(records, start=1):
            seq_id = rec.id or f"seq{idx}"
            try:
                self._scan_one(
                    sequence=str(rec.seq),
                    seq_id=seq_id,
                    score_thresh=0.0,
                    beam_size=64,
                    dataset_size=1.0,
                    evalue_thresh=0.1,
                    prior_mid_to_start=3.4e-5,
                    prior_stop_to_mid=4.817e-3,
                    prior_stop_to_start=5.540e-3,
                    T=1.0,
                    refine_extended=True,
                    verbose=False,
                    print_output=False,
                )
            except Exception:
                continue

    def _domain_evalue(
        self,
        domain: DomainTuple,
        dataset_size: float,
    ) -> float:
        if self.evalue_curve is None:
            raise RuntimeError("E-value curve not loaded; cannot compute E-values during scan().")
        return self.evalue_curve.evalue_from_score(domain[3], dataset_size)

    def _build_hit_rows(
        self,
        seq_id: str,
        domains: List[DomainTuple],
        dataset_size: float,
    ) -> List[HitRow]:
        return build_hit_rows(
            seq_id=seq_id,
            domains=domains,
            dataset_size=dataset_size,
            evalue_for_domain=self._domain_evalue,
            label_mapping=self.label_mapping,
        )

    def _write_hits_tsv(self, path: str, hit_rows: List[HitRow]) -> None:
        write_hits_tsv(path, hit_rows)

    def _render_scan_output(
        self,
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
        domains_original_scored: List[Tuple[str, int, int, float, float, float, float, str]],
        gamma,
        best_path,
        verbose: bool,
        refine_extended: bool,
        beam_size: int,
        inf_dt: float,
        dec_dt: float,
        output: Optional[TextIO] = None,
    ) -> None:
        render_scan_output(
            sequence=sequence,
            seq_id=seq_id,
            hit_rows=hit_rows,
            score_thresh=score_thresh,
            evalue_thresh=evalue_thresh,
            score_pass=score_pass,
            total_domains_raw=total_domains_raw,
            total_fams=total_fams,
            kept_families=kept_families,
            domains_original_scored=domains_original_scored,
            gamma=gamma,
            best_path=best_path,
            verbose=verbose,
            refine_extended=refine_extended,
            beam_size=beam_size,
            inf_dt=inf_dt,
            dec_dt=dec_dt,
            model_label=self.model_name or getattr(self.esm_model.config, "name_or_path", "training"),
            model_source_label=_format_model_source_display(self.model_source),
            device_label=str(next(self.parameters()).device),
            psalm_version=_get_psalm_version(),
            label_mapping=self.label_mapping,
            output=output,
        )

    def _ensure_scan_ready(self) -> None:
        if any(
            not hasattr(self, attr)
            for attr in ("inverse_label_mapping", "role_map", "none_label")
        ):
            raise RuntimeError("Decoder resources not initialized in this model.")
        if self.cbm_model is None:
            raise RuntimeError("CBM model not loaded; score.cbm is required for scan().")

    def _prepare_decode_inputs(self, log_probs: torch.Tensor) -> _PreparedDecodeInputs:
        device = log_probs.device
        total_fams = int((log_probs.size(1) - 1) / 3)
        _length, classes = log_probs.shape
        top_idx = log_probs.argmax(dim=1)
        pos_fams = ((top_idx + 2) // 3) * 3
        cache_key = (str(device), classes)
        pfam_ids = self._pfam_ids_cache.get(cache_key)
        if pfam_ids is None or pfam_ids.device != device:
            pfam_ids = ((torch.arange(classes, device=device) + 2) // 3) * 3
            self._pfam_ids_cache[cache_key] = pfam_ids

        active_fams = pos_fams.unique(sorted=False)
        keep_full = (pfam_ids.unsqueeze(0) == active_fams.unsqueeze(1)).any(dim=0)
        keep_full[self.none_label] = True
        keep_full_cpu = keep_full.cpu().contiguous()

        total_kept = int(keep_full_cpu.sum().item())
        kept_families = max(0, (total_kept - 1) // 3)

        sub_idx = keep_full.nonzero(as_tuple=False).flatten()
        if sub_idx.numel() <= 1 and bool((sub_idx == self.none_label).all()):
            return _PreparedDecodeInputs(
                total_fams=total_fams,
                kept_families=kept_families,
                keep_full_cpu=keep_full_cpu,
                sub_emits_cpu=None,
                none_sub=None,
            )

        sub_emits = log_probs.index_select(1, sub_idx)
        none_sub = int((sub_idx == self.none_label).nonzero(as_tuple=False).item())
        sub_emits = sub_emits - torch.logsumexp(sub_emits, dim=1, keepdim=True)
        sub_emits_cpu = sub_emits.to(dtype=torch.float32).cpu().contiguous()
        return _PreparedDecodeInputs(
            total_fams=total_fams,
            kept_families=kept_families,
            keep_full_cpu=keep_full_cpu,
            sub_emits_cpu=sub_emits_cpu,
            none_sub=none_sub,
        )

    def _decode_prepared_inputs(
        self,
        *,
        prepared: _PreparedDecodeInputs,
        sequence: str,
        beam_size: int,
        prior_mid_to_start: float,
        prior_stop_to_mid: float,
        prior_stop_to_start: float,
        refine_extended: bool,
        include_verbose_artifacts: bool,
    ) -> _DecodedDomains:
        if prepared.sub_emits_cpu is None or prepared.none_sub is None:
            return _DecodedDomains(
                domains=[],
                domains_original=[],
                gamma=None,
                best_path=None,
                dec_dt=0.0,
            )

        t0 = time.time()
        domains_original, domains, best_path, _, _, _ = annotate_domains(
            prepared.sub_emits_cpu,
            prepared.keep_full_cpu,
            sequence,
            self.inverse_label_mapping,
            self.label_mapping,
            self.from_list,
            self.to_list,
            self.lp_list,
            self.role_map,
            self.state_pfam_ids_full,
            self.state_role_ids_full,
            self.id_to_pfam,
            prepared.none_sub,
            beam_size=beam_size,
            prior_mid_to_start=prior_mid_to_start,
            prior_stop_to_mid=prior_stop_to_mid,
            prior_stop_to_start=prior_stop_to_start,
            refine_extended=refine_extended,
        )
        dec_dt = time.time() - t0
        if include_verbose_artifacts:
            gamma = prepared.sub_emits_cpu.exp()
            return _DecodedDomains(
                domains=domains,
                domains_original=domains_original,
                gamma=gamma,
                best_path=best_path,
                dec_dt=dec_dt,
            )
        return _DecodedDomains(
            domains=domains,
            domains_original=[],
            gamma=None,
            best_path=None,
            dec_dt=dec_dt,
        )

    def _finalize_scan_result(
        self,
        *,
        seq_id: str,
        decoded: _DecodedDomains,
        dataset_size: float,
        score_thresh: float,
        evalue_thresh: float,
        include_verbose_artifacts: bool,
    ) -> _FinalizedScanResult:
        domains_original_scored: List[DomainTuple] = []
        if include_verbose_artifacts and decoded.domains_original:
            domains_original_scored = add_cbm_scores(decoded.domains_original, self.cbm_model)

        domains_scored = add_cbm_scores(decoded.domains, self.cbm_model) if decoded.domains else []
        total_domains_raw = len(domains_scored)
        score_filtered = [domain for domain in domains_scored if domain[3] >= score_thresh]
        score_pass = len(score_filtered)
        kept_domains: List[DomainTuple] = []
        kept_evalues: List[float] = []
        for domain in score_filtered:
            evalue = self._domain_evalue(domain, dataset_size)
            if evalue <= evalue_thresh:
                kept_domains.append(domain)
                kept_evalues.append(evalue)
        hit_rows = build_hit_rows_from_evalues(
            seq_id=seq_id,
            domains=kept_domains,
            evalues=kept_evalues,
            label_mapping=self.label_mapping,
        )
        return _FinalizedScanResult(
            domains_scored=kept_domains,
            domains_original_scored=domains_original_scored,
            hit_rows=hit_rows,
            score_pass=score_pass,
            total_domains_raw=total_domains_raw,
        )

    def _build_scan_result_from_log_probs(
        self,
        *,
        seq_id: str,
        sequence: str,
        log_probs: torch.Tensor,
        score_thresh: float,
        beam_size: int,
        dataset_size: float,
        evalue_thresh: float,
        prior_mid_to_start: float,
        prior_stop_to_mid: float,
        prior_stop_to_start: float,
        refine_extended: bool,
        include_verbose_artifacts: bool,
    ) -> Tuple[_PreparedDecodeInputs, _DecodedDomains, _FinalizedScanResult]:
        prepared = self._prepare_decode_inputs(log_probs)
        decoded = self._decode_prepared_inputs(
            prepared=prepared,
            sequence=sequence,
            beam_size=beam_size,
            prior_mid_to_start=prior_mid_to_start,
            prior_stop_to_mid=prior_stop_to_mid,
            prior_stop_to_start=prior_stop_to_start,
            refine_extended=refine_extended,
            include_verbose_artifacts=include_verbose_artifacts,
        )
        finalized = self._finalize_scan_result(
            seq_id=seq_id,
            decoded=decoded,
            dataset_size=dataset_size,
            score_thresh=score_thresh,
            evalue_thresh=evalue_thresh,
            include_verbose_artifacts=include_verbose_artifacts,
        )
        return prepared, decoded, finalized

    def _scan_one(
        self,
        sequence: str,
        seq_id: str,
        score_thresh: float,
        beam_size: int,
        dataset_size: float,
        evalue_thresh: float,
        prior_mid_to_start: float,
        prior_stop_to_mid: float,
        prior_stop_to_start: float,
        T: float,
        refine_extended: bool,
        verbose: bool,
        print_output: bool,
    ) -> List[DomainTuple]:
        self._ensure_scan_ready()

        t0 = time.time()
        logits = self.infer(sequence)
        if T > 0.0:
            logits = logits / T
        log_probs = torch.log_softmax(logits, dim=-1)
        inf_dt = time.time() - t0
        prepared, decoded, finalized = self._build_scan_result_from_log_probs(
            seq_id=seq_id,
            sequence=sequence,
            log_probs=log_probs,
            score_thresh=score_thresh,
            beam_size=beam_size,
            dataset_size=dataset_size,
            evalue_thresh=evalue_thresh,
            prior_mid_to_start=prior_mid_to_start,
            prior_stop_to_mid=prior_stop_to_mid,
            prior_stop_to_start=prior_stop_to_start,
            refine_extended=refine_extended,
            include_verbose_artifacts=verbose,
        )
        if print_output:
            self._render_scan_output(
                sequence=sequence,
                seq_id=seq_id,
                hit_rows=finalized.hit_rows,
                score_thresh=score_thresh,
                evalue_thresh=evalue_thresh,
                score_pass=finalized.score_pass,
                total_domains_raw=finalized.total_domains_raw,
                total_fams=prepared.total_fams,
                kept_families=prepared.kept_families,
                domains_original_scored=finalized.domains_original_scored,
                gamma=decoded.gamma,
                best_path=decoded.best_path,
                verbose=verbose,
                refine_extended=refine_extended,
                beam_size=beam_size,
                inf_dt=inf_dt,
                dec_dt=decoded.dec_dt,
            )

        return finalized.domains_scored

    def scan(
        self,
        sequence: str = None,
        fasta: str = None,
        score_thresh: float = 0.85,
        beam_size: int = 64,
        dataset_size: Optional[float] = None,
        evalue_thresh: float = 0.1,
        prior_mid_to_start: float = 3.4e-5,
        prior_stop_to_mid: float = 4.817e-3,
        prior_stop_to_start: float = 5.540e-3,
        T: float = 1.0,
        refine_extended: bool = True,
        to_tsv: str = None,
        to_txt: str = None,
        verbose: bool = False,
        _seq_id: Optional[str] = None,
        _print_output: bool = True,
    ) -> Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]]:
        """
        Scan a protein sequence for domain annotations and display results.

        Args:
            sequence: Protein sequence as a string (or use fasta instead)
            fasta: Path to FASTA file containing one or more protein sequences
            score_thresh: CBM score threshold for filtering domains
            beam_size: Beam size for decoding
            dataset_size: Dataset size (Z) for E-value scaling; auto-derived when omitted
            evalue_thresh: Keep only domains with E-value <= this threshold
            prior_mid_to_start: Prior for middle to start transition
            prior_stop_to_mid: Prior for stop to middle transition
            prior_stop_to_start: Prior for stop to start transition
            T: Temperature scaling factor for logits (1.0 = no scaling)
            refine_extended: Run extended refinement for long domains
            to_tsv: Path to save results as TSV with the same ordering as the HITS table
            to_txt: Path to save printed output as text (optional)
            verbose: If True, print detailed results and alignments

        Returns:
            Dict mapping seq_id -> list of domain tuples:
              (pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status)
        """
        def _normalize_path(path: Optional[str], default_ext: str) -> Optional[str]:
            if path is None:
                return None
            base, ext = os.path.splitext(path)
            return path if ext else f"{base}{default_ext}"

        def _scan_one_with_optional_capture(active_seq: str, active_seq_id: str) -> tuple[List[DomainTuple], str]:
            if not emit_printable:
                return (
                    self._scan_one(
                        sequence=active_seq,
                        seq_id=active_seq_id,
                        score_thresh=score_thresh,
                        beam_size=beam_size,
                        dataset_size=effective_dataset_size,
                        evalue_thresh=evalue_thresh,
                        prior_mid_to_start=prior_mid_to_start,
                        prior_stop_to_mid=prior_stop_to_mid,
                        prior_stop_to_start=prior_stop_to_start,
                        T=T,
                        refine_extended=refine_extended,
                        verbose=verbose,
                        print_output=False,
                    ),
                    "",
                )

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                domains = self._scan_one(
                    sequence=active_seq,
                    seq_id=active_seq_id,
                    score_thresh=score_thresh,
                    beam_size=beam_size,
                    dataset_size=effective_dataset_size,
                    evalue_thresh=evalue_thresh,
                    prior_mid_to_start=prior_mid_to_start,
                    prior_stop_to_mid=prior_stop_to_mid,
                    prior_stop_to_start=prior_stop_to_start,
                    T=T,
                    refine_extended=refine_extended,
                    verbose=verbose,
                    print_output=True,
                )
            return domains, buffer.getvalue()

        def _run_with_optional_text_capture(active_seq: str, active_seq_id: str):
            domains, report_text = _scan_one_with_optional_capture(active_seq, active_seq_id)
            if emit_console and report_text:
                sys.stdout.write(report_text)
                sys.stdout.flush()
            if emit_text_file:
                with open(txt_path, "w", encoding="utf-8") as handle:
                    handle.write(report_text)
            return domains

        def _emit_progress_safe_report(report_text: str, pbar: tqdm) -> None:
            clean = report_text.rstrip("\n")
            if clean == "":
                return
            if pbar.disable:
                print(clean + "\n", end="")
            else:
                tqdm.write(clean + "\n", end="")

        def _append_report_text(handle, report_text: str, *, wrote_any: bool) -> bool:
            clean = report_text.rstrip("\n")
            if clean == "":
                return wrote_any
            if wrote_any:
                handle.write("\n\n")
            handle.write(clean)
            return True

        def _run_scan_one_direct(active_seq: str, active_seq_id: str):
            return self._scan_one(
                sequence=active_seq,
                seq_id=active_seq_id,
                score_thresh=score_thresh,
                beam_size=beam_size,
                dataset_size=effective_dataset_size,
                evalue_thresh=evalue_thresh,
                prior_mid_to_start=prior_mid_to_start,
                prior_stop_to_mid=prior_stop_to_mid,
                prior_stop_to_start=prior_stop_to_start,
                T=T,
                refine_extended=refine_extended,
                verbose=verbose,
                print_output=False,
            )

        if dataset_size is not None and dataset_size <= 0:
            raise ValueError("dataset_size must be > 0 when provided.")
        if sequence is not None and fasta is not None:
            raise ValueError("Provide exactly one of sequence or fasta, not both.")
        if sequence is None and fasta is None:
            raise ValueError("Provide either sequence or fasta.")

        txt_path = _normalize_path(to_txt, ".txt")
        emit_console = bool(_print_output)
        emit_text_file = bool(txt_path)
        emit_printable = emit_console or emit_text_file
        effective_dataset_size = float(dataset_size) if dataset_size is not None else 1.0

        if _seq_id is None and fasta is not None:
            total_records = _count_fasta_records_checked(fasta)
            if dataset_size is None:
                effective_dataset_size = float(max(1, total_records))
            if total_records > 1:
                results: Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]] = {}
                used_ids: Dict[str, int] = {}
                unnamed = 0
                pbar = tqdm(
                    total=total_records,
                    unit="seq",
                    ncols=80,
                    bar_format=SCAN_PROGRESS_BAR_FORMAT,
                    disable=(total_records <= 1) or (not sys.stderr.isatty()),
                )
                txt_handle = open(txt_path, "w", encoding="utf-8") if emit_text_file else None
                txt_written_any = False
                try:
                    for rec in _iter_fasta_records_checked(fasta):
                        rec_id = (rec.id or "").strip()
                        if rec_id == "":
                            unnamed += 1
                            rec_id = f"seq{unnamed}"
                        if rec_id in used_ids:
                            used_ids[rec_id] += 1
                            rec_id = f"{rec_id}_{used_ids[rec_id]}"
                        else:
                            used_ids[rec_id] = 1
                        record_sequence = _normalize_sequence_text(
                            str(rec.seq),
                            f"sequence '{rec_id}' in FASTA '{fasta}'",
                        )
                        if emit_printable:
                            domains, report_text = _scan_one_with_optional_capture(record_sequence, rec_id)
                            sub = {rec_id: domains}
                        else:
                            sub = {rec_id: _run_scan_one_direct(record_sequence, rec_id)}
                            report_text = ""
                        results.update(sub)
                        pbar.update(1)
                        if emit_console:
                            _emit_progress_safe_report(report_text, pbar)
                        if txt_handle is not None:
                            txt_written_any = _append_report_text(
                                txt_handle,
                                report_text,
                                wrote_any=txt_written_any,
                            )
                finally:
                    pbar.close()
                    if txt_handle is not None:
                        txt_handle.flush()
                        txt_handle.close()
                if to_tsv:
                    rows: List[HitRow] = []
                    for seq_id, domains in results.items():
                        rows.extend(self._build_hit_rows(seq_id, domains, effective_dataset_size))
                    self._write_hits_tsv(to_tsv, rows)
                return results

        if fasta is not None:
            rec = next(_iter_fasta_records_checked(fasta))
            sequence = _normalize_sequence_text(
                str(rec.seq),
                f"sequence '{(rec.id or 'seq1').strip() or 'seq1'}' in FASTA '{fasta}'",
            )
            seq_id = (rec.id or "").strip()
        else:
            sequence = _normalize_sequence_text(sequence, "sequence input")
            seq_id = ""
        if _seq_id is not None:
            seq_id = _seq_id
        if seq_id == "":
            seq_id = "seq1"
        domains_scored = _run_with_optional_text_capture(sequence, seq_id)

        if to_tsv:
            self._write_hits_tsv(to_tsv, self._build_hit_rows(seq_id, domains_scored, effective_dataset_size))

        return {seq_id: domains_scored}

    def scan_fast(
        self,
        *,
        fasta: str,
        score_thresh: float = 0.85,
        beam_size: int = 64,
        dataset_size: Optional[float] = None,
        evalue_thresh: float = 0.1,
        prior_mid_to_start: float = 3.4e-5,
        prior_stop_to_mid: float = 4.817e-3,
        prior_stop_to_start: float = 5.540e-3,
        refine_extended: bool = True,
        to_tsv: Optional[str] = None,
        to_txt: Optional[str] = None,
        sort: bool = False,
        cpu_workers: Optional[int] = None,
        max_batch_size: int = 4096,
        max_queue_size: int = 128,
        adaptive_fast: bool = False,
        _print_output: bool = True,
        _show_progress: Optional[bool] = None,
    ) -> None:
        from psalm.fast_scan import FastWorkerPoolManager, scan_fasta_fast

        worker_pool_manager = getattr(self, "_fast_worker_pool_manager", None)
        if (
            worker_pool_manager is None
            and getattr(self, "_fast_shell_mode", False)
            and (cpu_workers is None or cpu_workers > 0)
        ):
            worker_pool_manager = FastWorkerPoolManager(self)
            self._fast_worker_pool_manager = worker_pool_manager

        return scan_fasta_fast(
            self,
            fasta=fasta,
            score_thresh=score_thresh,
            beam_size=beam_size,
            dataset_size=dataset_size,
            evalue_thresh=evalue_thresh,
            prior_mid_to_start=prior_mid_to_start,
            prior_stop_to_mid=prior_stop_to_mid,
            prior_stop_to_start=prior_stop_to_start,
            refine_extended=refine_extended,
            to_tsv=to_tsv,
            to_txt=to_txt,
            sort=sort,
            cpu_workers=cpu_workers,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
            adaptive_fast=adaptive_fast,
            _print_output=_print_output,
            _show_progress=_show_progress,
            _worker_pool_manager=worker_pool_manager,
            _status_callback=getattr(self, "_fast_worker_status_callback", None),
        )

    def decode_sequence(self, *args, **kwargs) -> Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]]:
        warnings.warn(
            "decode_sequence() is deprecated and will be removed in a future version. "
            "Use scan() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scan(*args, **kwargs)
