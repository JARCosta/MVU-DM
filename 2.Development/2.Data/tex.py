import json
from pathlib import Path


def load_results(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percent(value: float) -> str:
    try:
        return f"{value * 100:.2f}\\%"
    except Exception:
        return ""


def get_metric(data: dict, dataset: str, method_key: str, metric: str) -> str:
    try:
        value = data[dataset][method_key][metric]
    except KeyError:
        return ""
    if value in (None, "", "nan"):
        return ""
    return percent(float(value))


def build_table(data: dict, metric: str) -> str:
    # Order and labels as in current tex.tex
    artificial_datasets = [
        ("broken.s_curve", "BSC"),
        ("two.swiss", "SR1"),
        ("parallel.swiss", "SR2"),
        ("four.moons", "FM"),
    ]
    natural_datasets = [
        ("coil20", "COIL20"),
        ("orl", "ORL"),
        ("mit-cbcl", "MIT-CBCL"),
        ("olivetti", "Olivetti"),
    ]
    methods = [
        ("isomap", "Isomap"),
        ("isomap.eng", "Isomap+ENG"),
        ("lle", "LLE"),
        ("hlle", "HLLE"),
        ("le", "LE"),
        ("ltsa", "LTSA"),
        ("tsne", "T-SNE"),
        ("kpca", "K-PCA"),
        ("mvu", "MVU"),
        ("mvu.eng", "MVU+ENG"),
        ("mvu.based", "MVU-DM"),
    ]

    header_cols = [label for _, label in artificial_datasets + natural_datasets]

    # Pre-compute best values per dataset column according to the metric rule
    def get_value(dataset_key: str, method_key: str) -> float | None:
        try:
            v = data[dataset_key][method_key][metric]
        except KeyError:
            return None
        if v in (None, "", "nan"):
            return None
        try:
            return float(v)
        except Exception:
            return None

    dataset_keys = [k for k, _ in artificial_datasets + natural_datasets]
    best_per_dataset: dict[str, float] = {}
    for ds_key in dataset_keys:
        values = [get_value(ds_key, mk) for mk, _ in methods]
        # Filter Nones
        present = [v for v in values if v is not None]
        if not present:
            continue
        if metric == "1-NN":
            best_per_dataset[ds_key] = min(present)
        else:  # T or C -> larger is better
            best_per_dataset[ds_key] = max(present)

    lines = []
    lines.append("\\begin{table}[t]")
    caption = {
        "1-NN": "1-NN results (Smaller values are better)",
        "T": "Trustworthiness results (Larger values are better)",
        "C": "Continuity results (Larger values are better)",
    }[metric]
    label = {
        "1-NN": "tab:1NN",
        "T": "tab:trustworthiness",
        "C": "tab:continuity",
    }[metric]
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{center}")
    lines.append("\\begin{tabular}{|l|c|c|c|c||c|c|c|c|}")
    lines.append("\\hline")
    lines.append(
        "\\multicolumn{1}{|c|}{} & \\multicolumn{4}{c||}{Artificial Datasets} & \\multicolumn{4}{c|}{Natural Datasets} \\\\" 
    )
    # lines.append("\\cline{2-8}")
    lines.append(
        "                     & "
        + " & ".join(header_cols[:3] + header_cols[3:])
        + " \\\\" 
    )
    lines.append("\\hline")

    def format_cell(dataset_key: str, method_key: str) -> str:
        raw_val = get_value(dataset_key, method_key)
        if raw_val is None:
            return ""
        best_val = best_per_dataset.get(dataset_key)
        formatted = percent(raw_val)
        if best_val is None:
            return formatted
        # Bold if ties within tolerance
        if abs(raw_val - best_val) < 1e-12:
            return f"\\textbf{{{formatted}}}"
        return formatted

    for method_key, method_label in methods:
        # Add a separating line before MVU-related methods
        if method_key == "mvu":
            lines.append("\\hline")
        row_values = []
        # artificial datasets
        for ds_key, _ in artificial_datasets:
            row_values.append(format_cell(ds_key, method_key))
        # natural datasets
        for ds_key, _ in natural_datasets:
            row_values.append(format_cell(ds_key, method_key))
        # Join, leaving blanks when missing
        row = (
            f"{method_label}            & "
            + " & ".join(v if v else "" for v in row_values)
            + " \\\\" 
        )
        lines.append(row)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_document(data: dict) -> str:
    tables = [
        build_table(data, "1-NN"),
        build_table(data, "T"),
        build_table(data, "C"),
    ]
    # Separate tables with a blank line to match current file style
    return ("\n\n".join(tables)) + "\n"


def main():
    root = Path(__file__).resolve().parent
    json_path = root / "measures.best.json"
    tex_path = root / "tex.tex"

    data = load_results(json_path)
    tex_content = build_document(data)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Updated {tex_path} from {json_path}")


if __name__ == "__main__":
    main()


