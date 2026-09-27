"""Local dashboard: mean ± std of one or more panSim ensembles, optional Hungary data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from .aggregate import summarize
from .columns import describe
from .ground_truth import GT_KIND, align_to_dates, load_ground_truth
from .metrics import metric_options, prepare
from .parse import RunSet, collect_files, load_run_set

PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)

DEFAULT_START = "2020-09-23"
NATIONAL_POPULATION = 9_600_000
SIM_POPULATION = 179_500


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_start(text: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(text.strip(), "%Y-%m-%d").date())


def simulation_dates(n_days: int, start: np.datetime64) -> np.ndarray:
    return start + np.arange(n_days).astype("timedelta64[D]")


def load_groups_from_specs(specs: list[str]) -> list[RunSet]:
    groups: list[RunSet] = []
    for spec in specs:
        if "=" in spec:
            name, raw = spec.split("=", 1)
        else:
            raw = spec
            name = Path(raw).name or "runs"
        paths = collect_files(raw)
        group = prepare(load_run_set(name.strip() or "runs", paths))
        if group.runs:
            groups.append(group)
    return groups


def parse_group_text(text: str) -> list[RunSet]:
    """One group per line: `name path` or `name=path` or a bare path."""
    specs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            specs.append(line)
            continue
        parts = line.split()
        if len(parts) == 1:
            specs.append(parts[0])
        else:
            specs.append(parts[0] + "=" + parts[1])
    return load_groups_from_specs(specs)


def ensemble_population(groups: list[RunSet], override: float | None) -> float:
    if override and override > 0:
        return float(override)
    pops = [g.population() for g in groups if np.isfinite(g.population())]
    if pops:
        return float(np.median(pops))
    return float(SIM_POPULATION)


def scale_factors(kind: str, mode: str, sim_pop: float, national_pop: float) -> tuple[float, float]:
    """Return (simulation multiplier, ground-truth multiplier)."""
    if kind != "count":
        return 1.0, 1.0
    if mode == "national":
        return national_pop / sim_pop, 1.0
    if mode == "per100k":
        return 100_000.0 / sim_pop, 100_000.0 / national_pop
    return 1.0, sim_pop / national_pop


def build_figure(
    groups: list[RunSet],
    metric: str,
    *,
    start: np.datetime64,
    smoother: int,
    show_std: bool,
    show_runs: bool,
    show_gt: bool,
    scale_mode: str,
    sim_pop: float,
    national_pop: float,
    ground_truth: dict[str, np.ndarray] | None,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    column = describe(metric)
    sim_scale, gt_scale = scale_factors(column.kind, scale_mode, sim_pop, national_pop)
    longest = 1
    for group_index, group in enumerate(groups):
        summary = summarize(group, metric, smoother)
        if summary is None:
            continue
        longest = max(longest, len(summary.mean))
        dates = simulation_dates(len(summary.mean), start)
        color = PALETTE[group_index % len(PALETTE)]
        x = [str(d) for d in dates.tolist()]
        if show_runs:
            for run_index, run_values in enumerate(summary.runs):
                run_x = x[: len(run_values)]
                fig.add_trace(
                    go.Scatter(
                        x=run_x,
                        y=run_values * sim_scale,
                        mode="lines",
                        line={"color": color, "width": 1},
                        opacity=0.25,
                        name=f"{group.name} run {run_index + 1}",
                        legendgroup=group.name,
                        showlegend=False,
                        hovertemplate="%{x}<br>%{y:.4g}<extra>" + group.name + "</extra>",
                    )
                )
        if show_std and np.any(summary.std > 0):
            upper = (summary.mean + summary.std) * sim_scale
            lower = (summary.mean - summary.std) * sim_scale
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=group.name,
                    name=f"{group.name} + std",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.18),
                    hovertemplate="%{x}<br>mean ± std<extra>" + group.name + "</extra>",
                    name=f"{group.name} ± std",
                    legendgroup=group.name,
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=summary.mean * sim_scale,
                mode="lines",
                line={"color": color, "width": 2.4},
                name=f"{group.name} mean (n={len(group.runs)})",
                legendgroup=group.name,
                hovertemplate="%{x}<br>%{y:.4g}<extra>" + group.name + "</extra>",
            )
        )
    if show_gt and ground_truth is not None and column.gt:
        gt_kind = GT_KIND.get(column.gt, column.kind)
        _, gt_only = scale_factors(gt_kind, scale_mode, sim_pop, national_pop)
        dates = simulation_dates(longest, start)
        aligned = align_to_dates(ground_truth, dates, column.gt, gt_only)
        if np.any(np.isfinite(aligned)):
            fig.add_trace(
                go.Scatter(
                    x=[str(d) for d in dates.tolist()],
                    y=aligned,
                    mode="lines",
                    connectgaps=False,
                    line={"color": "#111111", "width": 1.6, "dash": "dash"},
                    name="Hungary (scaled)",
                    hovertemplate="%{x}<br>%{y:.4g}<extra>Hungary</extra>",
                )
            )
    y_title = column.label
    if column.kind == "count" and scale_mode == "per100k":
        y_title += " per 100k"
    elif column.kind == "count" and scale_mode == "national":
        y_title += " (national scale)"
    fig.update_layout(
        title={"text": title or column.label, "x": 0.01, "xanchor": "left"},
        margin={"l": 56, "r": 16, "t": 48, "b": 40},
        legend={"orientation": "h", "y": 1.14, "x": 0},
        hovermode="x unified",
        template="plotly_white",
        yaxis_title=y_title,
        xaxis_title="",
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, rangemode="tozero" if column.kind == "count" else "normal")
    if not fig.data:
        fig.update_layout(
            annotations=[
                {
                    "text": f"No runs contain {metric}",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                }
            ]
        )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def load_layout(path: Path | None) -> dict:
    if path and path.is_file():
        return json.loads(path.read_text())
    fallback = Path(__file__).resolve().parents[1] / "layouts" / "overview.json"
    if fallback.is_file():
        return json.loads(fallback.read_text())
    return {
        "columns": 2,
        "smoothing": 7,
        "scale": "simulation",
        "show_std": True,
        "show_runs": False,
        "show_ground_truth": True,
        "panels": [{"metric": "NI", "title": "New infections"}],
    }


def _panel(index: int, metric: str, title: str, options: list[dict[str, str]]):
    return html.Div(
        [
            html.Div(
                [
                    dcc.Input(
                        id={"type": "title", "index": index},
                        value=title,
                        type="text",
                        debounce=True,
                        style={"flex": "1", "marginRight": "8px"},
                    ),
                    dcc.Dropdown(
                        id={"type": "metric", "index": index},
                        options=options,
                        value=metric,
                        clearable=False,
                        style={"flex": "2"},
                    ),
                    html.Button("Remove", id={"type": "remove", "index": index}, n_clicks=0, style={"marginLeft": "8px"}),
                ],
                style={"display": "flex", "gap": "0", "alignItems": "center", "marginBottom": "6px"},
            ),
            dcc.Graph(id={"type": "graph", "index": index}, style={"height": "360px"}, config={"displaylogo": False}),
        ],
        id={"type": "panel", "index": index},
        style={"minWidth": "0"},
    )


def create_app(
    groups: list[RunSet],
    ground_truth: dict[str, np.ndarray] | None,
    layout_doc: dict,
    group_text: str,
    gt_path: str,
) -> Dash:
    app = Dash(__name__)
    app.title = "panSim outputs"
    state = {
        "groups": groups,
        "ground_truth": ground_truth,
        "next_index": len(layout_doc["panels"]),
    }
    options = metric_options(groups[0]) if groups else [{"label": "New infections [NI]", "value": "NI"}]
    panels = [
        _panel(i, p.get("metric", "NI"), p.get("title", ""), options) for i, p in enumerate(layout_doc["panels"])
    ]
    app.layout = html.Div(
        [
            dcc.Store(id="panel-store", data={"n": len(panels)}),
            html.H2("panSim output viewer", style={"marginBottom": "4px"}),
            html.P(
                "Each line is the mean of a stochastic ensemble. The band is ± one standard deviation. "
                "Hungary counts are scaled between the national population and the simulated population, and days with no report are left blank.",
                style={"marginTop": "0", "color": "#333", "maxWidth": "980px"},
            ),
            html.Div(
                [
                    html.Label("Ensembles, one per line: name path"),
                    dcc.Textarea(
                        id="groups",
                        value=group_text,
                        style={"width": "100%", "height": "72px", "fontFamily": "monospace"},
                    ),
                    html.Div(
                        [
                            html.Label("Ground truth workbook"),
                            dcc.Input(id="gt-path", value=gt_path, type="text", style={"width": "100%"}),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Button("Reload data", id="reload", n_clicks=0, style={"marginTop": "8px"}),
                    html.Div(id="status", children=_startup_status(groups, ground_truth), style={"marginTop": "6px", "color": "#333"}),
                ],
                style={"maxWidth": "980px", "marginBottom": "12px"},
            ),
            html.Div(
                [
                    _labeled("Start date", dcc.Input(id="start", value=layout_doc.get("start", DEFAULT_START), type="text", debounce=True)),
                    _labeled(
                        "Sim population (0 = detect)",
                        dcc.Input(id="sim-pop", value=0, type="number", debounce=True, style={"width": "120px"}),
                    ),
                    _labeled(
                        "National population",
                        dcc.Input(id="nat-pop", value=NATIONAL_POPULATION, type="number", debounce=True, style={"width": "140px"}),
                    ),
                    _labeled(
                        "Trailing average (days)",
                        dcc.Input(id="smooth", value=int(layout_doc.get("smoothing", 7)), type="number", min=1, debounce=True, style={"width": "80px"}),
                    ),
                    _labeled(
                        "Plots per row",
                        dcc.Input(id="ncols", value=int(layout_doc.get("columns", 2)), type="number", min=1, max=4, debounce=True, style={"width": "70px"}),
                    ),
                    _labeled(
                        "Scale",
                        dcc.Dropdown(
                            id="scale",
                            options=[
                                {"label": "Simulation agents", "value": "simulation"},
                                {"label": "Per 100 000", "value": "per100k"},
                                {"label": "National population", "value": "national"},
                            ],
                            value=layout_doc.get("scale", "simulation"),
                            clearable=False,
                            style={"width": "220px"},
                        ),
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "flex-end"},
            ),
            dcc.Checklist(
                id="toggles",
                options=[
                    {"label": " Standard deviation band", "value": "std"},
                    {"label": " Individual runs", "value": "runs"},
                    {"label": " Hungary ground truth", "value": "gt"},
                ],
                value=_toggle_values(layout_doc),
                style={"margin": "10px 0"},
                inline=True,
            ),
            html.Div(
                [
                    html.Button("Add plot", id="add", n_clicks=0),
                    html.Button("Save layout", id="save", n_clicks=0, style={"marginLeft": "8px"}),
                    dcc.Input(id="layout-path", value=str(_repo_root() / "viz" / "layouts" / "custom.json"), type="text", style={"width": "420px", "marginLeft": "8px"}),
                    html.Span(id="save-msg", style={"marginLeft": "8px"}),
                ]
            ),
            html.Div(id="grid", children=panels, style=_grid_style(int(layout_doc.get("columns", 2)))),
        ],
        style={"fontFamily": "system-ui, sans-serif", "padding": "16px 20px 40px"},
    )

    @app.callback(
        Output("grid", "children"),
        Output("grid", "style"),
        Output("status", "children"),
        Input("add", "n_clicks"),
        Input("reload", "n_clicks"),
        Input({"type": "remove", "index": ALL}, "n_clicks"),
        Input("ncols", "value"),
        State("grid", "children"),
        State("groups", "value"),
        State("gt-path", "value"),
        prevent_initial_call=True,
    )
    def edit_grid(add_clicks, reload_clicks, remove_clicks, ncols, children, groups_text, gt_text):
        triggered = ctx.triggered_id
        message = no_update
        if triggered == "reload":
            try:
                state["groups"] = parse_group_text(groups_text or "")
                gt_file = Path(gt_text) if gt_text else None
                state["ground_truth"] = load_ground_truth(gt_file) if gt_file and gt_file.is_file() else None
                n_runs = sum(len(g.runs) for g in state["groups"])
                gt_note = "ground truth loaded" if state["ground_truth"] is not None else "no ground truth"
                message = f"{len(state['groups'])} ensemble(s), {n_runs} run(s), {gt_note}."
            except Exception as exc:  # noqa: BLE001 - show the loader error in the page
                message = f"Could not reload: {exc}"
            options_now = metric_options(state["groups"][0]) if state["groups"] else options
            for child in children or []:
                _retarget_dropdown(child, options_now)
        elif triggered == "add":
            index = state["next_index"]
            state["next_index"] += 1
            options_now = metric_options(state["groups"][0]) if state["groups"] else options
            children = list(children or []) + [_panel(index, "NI", "New infections", options_now)]
        elif isinstance(triggered, dict) and triggered.get("type") == "remove":
            if sum(remove_clicks or []) > 0:
                children = [
                    child
                    for child in children or []
                    if child.get("props", {}).get("id", {}).get("index") != triggered["index"]
                ]
        ncols = int(ncols or 2)
        return children, _grid_style(ncols), message

    @app.callback(
        Output({"type": "graph", "index": ALL}, "figure"),
        Input({"type": "metric", "index": ALL}, "value"),
        Input({"type": "title", "index": ALL}, "value"),
        Input("start", "value"),
        Input("smooth", "value"),
        Input("scale", "value"),
        Input("toggles", "value"),
        Input("sim-pop", "value"),
        Input("nat-pop", "value"),
        Input("status", "children"),
    )
    def redraw(metrics, titles, start_text, smooth, scale, toggles, sim_pop, nat_pop, _status):
        try:
            start = _parse_start(start_text or DEFAULT_START)
        except ValueError:
            start = _parse_start(DEFAULT_START)
        toggles = toggles or []
        figures = []
        detected = ensemble_population(state["groups"], float(sim_pop or 0))
        national = float(nat_pop or NATIONAL_POPULATION)
        for metric, title in zip(metrics, titles):
            figures.append(
                build_figure(
                    state["groups"],
                    metric or "NI",
                    start=start,
                    smoother=max(1, int(smooth or 1)),
                    show_std="std" in toggles,
                    show_runs="runs" in toggles,
                    show_gt="gt" in toggles,
                    scale_mode=scale or "simulation",
                    sim_pop=detected,
                    national_pop=national,
                    ground_truth=state["ground_truth"],
                    title=title or "",
                )
            )
        return figures

    @app.callback(
        Output("save-msg", "children"),
        Input("save", "n_clicks"),
        State({"type": "metric", "index": ALL}, "value"),
        State({"type": "title", "index": ALL}, "value"),
        State("ncols", "value"),
        State("smooth", "value"),
        State("scale", "value"),
        State("toggles", "value"),
        State("start", "value"),
        State("layout-path", "value"),
        prevent_initial_call=True,
    )
    def save_layout(n_clicks, metrics, titles, ncols, smooth, scale, toggles, start, path):
        if not n_clicks:
            return no_update
        doc = {
            "columns": int(ncols or 2),
            "smoothing": int(smooth or 1),
            "scale": scale,
            "start": start,
            "show_std": "std" in (toggles or []),
            "show_runs": "runs" in (toggles or []),
            "show_ground_truth": "gt" in (toggles or []),
            "panels": [{"metric": m, "title": t or ""} for m, t in zip(metrics, titles)],
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(doc, indent=2) + "\n")
        return f"Saved {dest}"

    return app


def _retarget_dropdown(node, options):
    props = node.get("props", {})
    children = props.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _retarget_dropdown(child, options)
    elif isinstance(children, dict):
        _retarget_dropdown(children, options)
    ident = props.get("id")
    if isinstance(ident, dict) and ident.get("type") == "metric":
        props["options"] = options


def _toggle_values(layout_doc: dict) -> list[str]:
    values = []
    if layout_doc.get("show_std", True):
        values.append("std")
    if layout_doc.get("show_runs", False):
        values.append("runs")
    if layout_doc.get("show_ground_truth", True):
        values.append("gt")
    return values


def _grid_style(columns: int) -> dict:
    return {
        "display": "grid",
        "gridTemplateColumns": f"repeat({max(1, columns)}, minmax(0, 1fr))",
        "gap": "16px",
        "marginTop": "12px",
    }


def _startup_status(groups: list[RunSet], ground_truth: dict | None) -> str:
    n_runs = sum(len(g.runs) for g in groups)
    if not groups:
        gt = " Ground truth is loaded." if ground_truth is not None else ""
        return "No runs loaded. Add ensemble paths and press Reload data." + gt
    detected = ensemble_population(groups, None)
    gt = "ground truth loaded" if ground_truth is not None else "no ground truth file"
    return (
        f"{len(groups)} ensemble(s), {n_runs} run(s). "
        f"Detected simulated population {detected:.0f} (set Sim population to override, for example 179500). {gt}."
    )


def _labeled(label: str, control):
    return html.Div([html.Div(label, style={"fontSize": "12px", "color": "#444"}), control])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive plots of panSim console output.")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Ensemble directory or file. Repeat for several scenarios. Example: --group baseline=viz/runs/baseline",
    )
    parser.add_argument("--ground-truth", default=str(_repo_root() / "korona_hun.xlsx"), help="korona_hun.xlsx path")
    parser.add_argument("--layout", default="", help="Layout JSON. Defaults to viz/layouts/overview.json")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Calendar date of simulation day 0 (default 2020-09-23)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)

    groups = load_groups_from_specs(args.group)
    gt_path = Path(args.ground_truth)
    ground_truth = load_ground_truth(gt_path) if gt_path.is_file() else None
    layout_doc = load_layout(Path(args.layout) if args.layout else None)
    layout_doc["start"] = args.start_date
    lines = []
    for spec in args.group:
        if "=" in spec:
            name, raw = spec.split("=", 1)
            lines.append(f"{name} {raw}")
        else:
            lines.append(spec)
    app = create_app(groups, ground_truth, layout_doc, "\n".join(lines), str(gt_path))
    print(f"panSim viewer at http://{args.host}:{args.port}")
    if groups:
        for group in groups:
            print(f"  {group.name}: {len(group.runs)} run(s), {group.runs[0].n_days} days in the first file")
    else:
        print("  no runs loaded yet — add ensemble paths in the page and press Reload data")
    if ground_truth is None:
        print(f"  ground truth not found at {gt_path}")
    else:
        print(f"  ground truth {ground_truth['date'][0]} .. {ground_truth['date'][-1]}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
