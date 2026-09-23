#!/usr/bin/env python3
"""Build the table from event evaluations and source-attributed reported values."""

import argparse
import hashlib
import json
import posixpath
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
CLASSICS = (
    ("mvcnn", "Multi-view CNN"),
    ("gine", "Static GINE"),
    ("bigru", "BiGRU"),
    ("mamba", "PointMamba-lite"),
)
CLASSIC_ROWS = {
    "NEXT": ("NEXT Model Results", (5, 10, 17, 23)),
    "MJD": ("MJD Model Results", (2, 3, 4, 5)),
    "EXO-200": ("EXO200 Model Results", (2, 3, 4, 5)),
    "SuperNEMO": ("SuperNEMO Model Results", (2, 3, 4, 5)),
}
TRANSFORMERS = tuple(
    (f"{family}_{encoding}", f"{label} + {pe}")
    for family, label in (("entity", "Entity"), ("region", "Region"), ("summary", "Summary"))
    for encoding, pe in (("mlp", "MLP"), ("fourier", "Fourier"), ("rope", "RoPE"))
)


def read_workbook(path):
    with ZipFile(path) as archive:
        strings = [
            "".join(t.text or "" for t in item.findall(".//s:t", NS))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        links = {
            item.attrib["Id"]: posixpath.normpath("xl/" + item.attrib["Target"])
            for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = {}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", NS):
            cells = {}
            for cell in ET.fromstring(archive.read(links[sheet.attrib[REL]])).findall(".//s:sheetData/s:row/s:c", NS):
                value = cell.findtext("s:v", default="", namespaces=NS)
                if cell.attrib.get("t") == "s":
                    value = strings[int(value)] if value else ""
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                cells[cell.attrib["r"]] = value
            sheets[sheet.attrib["name"]] = cells
    return sheets


def classic_records(sheets):
    records = []
    for dataset, (sheet, rows) in CLASSIC_ROWS.items():
        cells = sheets[sheet]
        for (key, label), row in zip(CLASSICS, rows):
            assert cells.get(f"Q{row}") == "classification"
            record = {
                "dataset": dataset, "model_key": key, "model_name": label,
                "worksheet": sheet, "row": row,
                "status": cells.get(f"A{row}"),
                "job_id": cells.get(f"C{row}"),
                "task": cells.get(f"Q{row}"),
                "source_cells": {},
            }
            for field, col in (("params", "K"), ("matched_auc", "T"), ("independence", "X")):
                ref = f"{col}{row}"
                record[field] = cells.get(ref) or None
                record["source_cells"][field] = f"'{sheet}'!{ref}"
            records.append(record)
    return records


def load_additions(path):
    """Load explicit table additions and resolve their parameter provenance."""
    if not path.exists():
        return {"records": []}
    additions = json.loads(path.read_text())
    for record in additions["records"]:
        if record["source_kind"] == "unified_result_reference":
            provenance = json.loads((path.parent / record["parameter_provenance"]).read_text())
            assert (provenance["dataset"], provenance["model_key"]) == (
                record["source_dataset"], record["source_model_key"])
            record["parameter_evidence"] = provenance
    return additions


def main_records(source, transformer_source, additions=None):
    """Prefer reviewed event evaluations over earlier reported summaries."""
    classic_keys = {key for key, _ in CLASSICS}
    records = []
    for result in source["records"]:
        if not result["main_table"] or result["model_key"] not in classic_keys:
            continue
        assert result["status"] in ("recomputed", "missing_event_inputs")
        records.append(dict(
            dataset=result["dataset"], model_key=result["model_key"],
            params=result.get("params"),
            matched_auc=None if result["matched_auc"] is None else repr(result["matched_auc"]),
            independence=None if result["I"] is None else repr(result["I"]),
            status=result["status"], protocol_sha256=result["protocol_sha256"],
            source_manifest=result["source_manifest"], input_source=result.get("input_source"),
            comparison_eligible=not (result["dataset"] == "MJD" and result["model_key"] == "gine")))
    assert len(records) == 16
    transformer_keys = {key for key, _ in TRANSFORMERS}
    assert len(transformer_source["records"]) == 30
    for result in transformer_source["records"]:
        assert result["model_key"] in transformer_keys
        records.append(dict(
            dataset=result["dataset"], model_key=result["model_key"],
            params=result["params"], matched_auc=result["matched_auc"],
            independence=result["independence"], status="historical_transformer",
            worksheet=result["worksheet"], row=result["row"],
            source_cells=result["source_cells"], comparability=result.get("comparability")))
    lookup = {(r["dataset"], r["model_key"]): r for r in records}
    result_lookup = {(r["dataset"], r["model_key"]): r for r in source["records"]}
    for addition in (additions or {}).get("records", []):
        key = addition["dataset"], addition["model_key"]
        assert isinstance(addition["comparison_eligible"], bool)
        if addition["source_kind"] == "unified_result_reference":
            assert not addition["comparison_eligible"]
            original = lookup[key]
            assert original["status"] == "missing_event_inputs"
            result = result_lookup[addition["source_dataset"], addition["source_model_key"]]
            provenance = addition["parameter_evidence"]
            assert result["status"] == "recomputed"
            assert result["input_sha256"] == provenance["prediction_input_sha256"]
            assert result["n_events"] == provenance["n_events"]
            assert result["architecture_id"] == provenance["architecture_id"]
            assert result["protocol_sha256"] == original["protocol_sha256"]
            original.update(
                params=str(provenance["parameter_count"]),
                matched_auc=repr(result["matched_auc"]), independence=repr(result["I"]),
                status="recomputed_reference", comparison_eligible=False,
                source_dataset=result["dataset"], source_model_key=result["model_key"],
                n_events=result["n_events"], input_sha256=result["input_sha256"],
                source_manifest=result["source_manifest"], input_source=result["input_source"],
                parameter_evidence=provenance)
        elif addition["source_kind"] == "reported_metric_updates":
            assert addition["model_key"] in transformer_keys
            if key not in lookup:
                record = dict(dataset=addition["dataset"], model_key=addition["model_key"],
                              params=None, matched_auc=None, independence=None,
                              status="reported_display_values")
                records.append(record)
                lookup[key] = record
            record = lookup[key]
            for field, value in addition["metrics"].items():
                assert field in ("matched_auc", "independence")
                assert record.get(field) is None, "A supplied addition must not overwrite an existing metric"
                assert 0 <= Decimal(value) <= 1
                record[field] = value
                record.setdefault("precision_by_field", {})[field] = "reported_3dp"
                record.setdefault("field_sources", {})[field] = addition["source"]
            record["comparison_eligible"] = addition["comparison_eligible"]
        else:
            assert addition["source_kind"] == "reported_display_values"
            assert not addition["comparison_eligible"]
            assert key not in lookup and addition["model_key"] in transformer_keys
            record = dict(
                dataset=addition["dataset"], model_key=addition["model_key"],
                params=None, params_display_thousands=addition["params_display_thousands"],
                matched_auc=addition["matched_auc"], independence=addition["independence"],
                status="reported_display_values", comparison_eligible=False,
                precision=addition["precision"], source=addition["source"])
            records.append(record)
            lookup[key] = record
    # Apply event-level reevaluations last, so workbook/display additions cannot
    # restore superseded values. Unselected models retain their source records.
    campaign = source.get("transformer_reevaluation", {})
    selected = {(r["dataset"], r["model_key"]) for r in campaign.get("selected_models", [])}
    updates = {
        (r["dataset"], r["model_key"]): r for r in source["records"]
        if r["model_key"] in transformer_keys and r["status"] == "recomputed"
    }
    assert set(updates) == selected, "Recomputed Transformer records must match the declared campaign"
    for key, result in updates.items():
        assert key in lookup and result["main_table"]
        assert result["protocol_sha256"] == campaign["protocol_sha256"]
        assert result["input_sha256"] and result["n_events"] > 0
        assert (result["matching_status"] == "ok") == (result["matched_auc"] is not None)
        record = lookup[key]
        prior = {field: record.get(field) for field in (
            "status", "matched_auc", "independence", "params", "source_cells",
            "precision", "precision_by_field", "field_sources", "source") if field in record}
        for field in ("precision", "precision_by_field", "field_sources", "params_display_thousands", "comparability"):
            record.pop(field, None)
        record.update(
            matched_auc=None if result["matched_auc"] is None else repr(result["matched_auc"]),
            independence=None if result["I"] is None else repr(result["I"]),
            params=result["params"], status="recomputed_transformer",
            comparison_eligible=result["comparison_eligible"],
            protocol_sha256=result["protocol_sha256"],
            source_manifest=result["source_manifest"], input_source=result["input_source"],
            input_sha256=result["input_sha256"], n_events=result["n_events"],
            evaluation_campaign=campaign["campaign_id"], previous_reported_values=prior)
    assert len(records) == len(lookup)
    return records


def parameter_range(lookup, model_key, datasets):
    """Summarize the available dataset-specific counts in rounded thousands."""
    values = []
    for dataset in datasets:
        record = lookup.get((dataset, model_key), {})
        if record.get("params_display_thousands") is not None:
            thousands = Decimal(str(record["params_display_thousands"]))
            assert thousands == thousands.to_integral_value()
        elif record.get("params") is not None:
            count = Decimal(str(record["params"]))
            assert count == count.to_integral_value()
            thousands = (count / Decimal("1000")).quantize(Decimal("1"))
        else:
            continue
        assert thousands >= 0
        values.append(int(thousands))
    if not values:
        return r"\NA"
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}--{hi}"


def metric_interval(record, field):
    """Retain uncertainty from a result supplied only to three decimals."""
    value = Decimal(str(record[field]))
    assert 0 <= value <= 1
    precision = record.get("precision_by_field", {}).get(field)
    display_only = precision == "reported_3dp" or (
        precision is None and record.get("status") == "reported_display_values")
    if display_only:
        half_unit = Decimal("0.0005")
        return max(Decimal("0"), value - half_unit), min(Decimal("1"), value + half_unit)
    return value, value


def highlighted_models(lookup, models, dataset, field):
    intervals = {
        key: metric_interval(record, field)
        for key, _ in models
        if (record := lookup.get((dataset, key), {})).get(field) is not None
        and record.get("comparison_eligible", True)
    }
    winners = set()
    for key, (lo, hi) in intervals.items():
        # Equal exact values may share first place. Overlapping rounded
        # intervals do not establish a winner, even if displayed values tie.
        if all(lo > other_hi or lo == hi == other_lo == other_hi
               for other, (other_lo, other_hi) in intervals.items() if other != key):
            winners.add(key)
    return winners


def render(records):
    lookup = {(r["dataset"], r["model_key"]): r for r in records}
    assert len(lookup) == len(records), "Duplicate semantic model keys"
    datasets = ("NEXT", "MJD", "EXO-200", "SuperNEMO")
    dataset_winners = {
        (dataset, field): highlighted_models(lookup, CLASSICS + TRANSFORMERS, dataset, field)
        for dataset in datasets for field in ("matched_auc", "independence")
    }
    output = [
        "% Generated by wing_contribution/scripts/build_classification_table.py.",
        r"\providecommand{\bestscore}[1]{\ensuremath{\mathbf{#1}}}",
        r"\definecolor{datasetbestcolor}{RGB}{0,76,153}",
        r"\providecommand{\datasetbest}[1]{\textcolor{datasetbestcolor}{#1}}",
        r"\providecommand{\NA}{\textemdash}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-detector classification results. Higher values are better for "
        r"energy-matched AUC ($\mathrm{AUC}_{\mathrm{match}}$) and energy independence ($I$). "
        r"Parameters are rounded counts or ranges in thousands across available "
        r"dataset-specific implementations; ranges are not uncertainty. Within each "
        r"model family and dataset, bold marks the highest eligible result where "
        r"recorded precision resolves the comparison; displayed ties use unrounded "
        r"values where available. \datasetbest{Blue} marks the highest AUC and $I$ across "
        r"both model families in each dataset. Highlighting denotes point estimates, not statistical "
        r"significance. Results are from single training seeds. A dash denotes an unavailable result. "
        r"$^{\ddagger}$MJD GINE uses additional PSD-label supervision and is excluded "
        r"from best-result highlighting. Evaluation details are provided in "
        r"Appendix~\ref{app:metric-details}.}",
        r"\label{tab:benchmark-main}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc*{4}{rr}@{}}",
        r"\toprule",
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in datasets) + r" \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
        r"Model & \shortstack{\#Params\\(K)} & "
        + " & ".join([r"$\mathrm{AUC}_{\mathrm{match}}$ & $I$"] * len(datasets)) + r" \\",
    ]
    for block_name, models in (("Specialized baselines", CLASSICS), ("Tokenized Transformers", TRANSFORMERS)):
        winners = {(dataset, field): highlighted_models(lookup, models, dataset, field)
                   for dataset in datasets for field in ("matched_auc", "independence")}
        output += [r"\midrule", f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{block_name}}}}}" + r" \\"]
        for key, label in models:
            if key in ("region_mlp", "summary_mlp"):
                output.append(r"\addlinespace[2pt]")
            row = [label, parameter_range(lookup, key, datasets)]
            for dataset in datasets:
                record = lookup.get((dataset, key), {})
                for field in ("matched_auc", "independence"):
                    value = record.get(field)
                    if value is None:
                        formatted = r"\NA"
                    else:
                        number = Decimal(str(value))
                        assert 0 <= number <= 1
                        formatted = f"{number:.3f}"
                        if key in winners[dataset, field]:
                            formatted = r"\bestscore{" + formatted + "}"
                        if key in dataset_winners[dataset, field]:
                            formatted = r"\datasetbest{" + formatted + "}"
                    if key == "gine" and dataset == "MJD":
                        formatted += r"$^{\ddagger}$"
                    row.append(formatted)
            output.append(" & ".join(row) + r" \\")
    output += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "figures/data/unified_results.json")
    parser.add_argument("--transformers", type=Path, help="Workbook evidence JSON; defaults beside --results")
    parser.add_argument("--additions", type=Path, help="Explicit table additions JSON; defaults beside --results")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Output root containing tables/ and figures/data/")
    args = parser.parse_args()
    source = json.loads(args.results.read_text())
    transformer_path = args.transformers or args.results.with_name("classification_transformer_workbook.json")
    transformer_source = json.loads(transformer_path.read_text())
    additions_path = args.additions or args.results.with_name("classification_table_additions.json")
    records = main_records(source, transformer_source, load_additions(additions_path))
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures/data").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables/benchmark_main.tex").write_text(render(records))
    evidence = dict(source_results=str(args.results.resolve()), sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
        source_transformers=str(transformer_path.resolve()), transformer_sha256=hashlib.sha256(transformer_path.read_bytes()).hexdigest(),
        source_additions=str(additions_path.resolve()) if additions_path.exists() else None,
        additions_sha256=hashlib.sha256(additions_path.read_bytes()).hexdigest() if additions_path.exists() else None,
        protocol=source["protocol"], protocol_registry=source.get("protocol_registry"), evaluation_campaign=source.get("transformer_reevaluation"),
        scope="Classic values and declared Transformer reevaluations use event-level unified results. NEXT PointMamba retains its explicitly referenced original population. Unselected Transformer entries retain their audited workbook, Overleaf, or user-supplied values and recorded precision.",
        missing_values="Only explicit source-backed or user-supplied additions fill missing entries; no old_* metric fields are substituted.",
        emphasis="Bold marks the highest eligible value within each model family and dataset; blue additionally marks the column maximum across both families. Both use recorded precision: full-precision values resolve displayed ties, reported three-decimal values use conservative +/-0.0005 intervals, and overlapping rounded ties are not resolved. NEXT PointMamba's referenced record, MJD GINE, and the earlier MJD display-only additions remain excluded.",
        parameter_summary="One rounded-thousands count or min--max range across available dataset-specific parameter records; missing counts are skipped and displayed-only counts do not imply exact parameters.", records=records)
    (args.output_dir / "figures/data/classification_table_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Wrote {len(records)} source-backed main-table entries with explicit comparison eligibility.")


if __name__ == "__main__":
    main()
