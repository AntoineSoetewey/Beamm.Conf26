from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, pearsonr, f_oneway


APP_TITLE = "AssociationExplorer"
DEFAULT_THRESHOLD_RANGE = (0.5, 1.0)
MAX_PAIRS_TO_RENDER = 120

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "data.csv"
DESCRIPTION_FILE = BASE_DIR / "data" / "description.csv"

NETWORK_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": 12,
            "color": "#1f2933",
            "background-color": "#dbe6f2",
            "border-color": "#4f81bd",
            "border-width": 1,
            "text-wrap": "wrap",
            "text-max-width": 180,
            "width": 28,
            "height": 28,
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "line-opacity": 0.85,
        },
    },
    {
        "selector": ":selected",
        "style": {
            "border-width": 3,
            "border-color": "#1f2933",
            "line-color": "#1f2933",
            "target-arrow-color": "#1f2933",
        },
    },
]

HELP_MARKDOWN = """
### About this application

This test app explores relationships between variables in a tabular dataset.

### Association measures

- Numeric vs numeric: Pearson correlation r (threshold applied on r^2)
- Categorical vs categorical: Cramer's V (threshold applied directly on V)
- Numeric vs categorical: correlation ratio eta (threshold applied on eta^2)

### How to use

1. Adjust the threshold range to keep only associations in that strength interval.
2. Use Correlation network to inspect the global structure of associations.
3. Use Pair plots to inspect each kept pair in detail.

### Edge colors in the network

- Blue: positive Pearson correlation
- Red: negative Pearson correlation
- Orange: numeric-categorical association (eta)
- Gray: categorical-categorical association (Cramer's V)

### Interpretation tips

- Edge thickness is proportional to association strength.
- Node distance is from force-directed layout and should be interpreted qualitatively.
"""

_CALLBACKS_REGISTERED = False
_DATA_ERROR: str | None = None

_DF_FILTERED = pd.DataFrame()
_DESCRIPTIONS: dict[str, str] = {}
_NUMERIC_COLS: set[str] = set()
_CATEGORICAL_COLS: set[str] = set()
_ALL_ASSOCIATIONS: list[dict[str, Any]] = []


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramer's V for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0.0

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    min_dim = min(confusion_matrix.shape) - 1

    if n == 0 or min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Compute eta (correlation ratio) for categorical vs numeric."""
    categories_series = pd.Series(categories)
    measurements_series = pd.Series(measurements)

    mask = categories_series.notna() & measurements_series.notna()
    categories_series = categories_series[mask]
    measurements_series = measurements_series[mask]

    if len(measurements_series) == 0:
        return 0.0

    mean_global = measurements_series.mean()
    groups = measurements_series.groupby(categories_series)

    ssb = sum(len(group) * (group.mean() - mean_global) ** 2 for _, group in groups)
    sst = float(((measurements_series - mean_global) ** 2).sum())

    if sst == 0:
        return 0.0

    eta_squared = ssb / sst
    return float(np.sqrt(max(0.0, eta_squared)))


def _load_descriptions() -> dict[str, str]:
    if not DESCRIPTION_FILE.exists():
        return {}

    desc_df = pd.read_csv(DESCRIPTION_FILE)
    if not {"Variable", "Description"}.issubset(desc_df.columns):
        return {}

    descriptions = desc_df.set_index("Variable")["Description"].to_dict()
    return {str(k): str(v) for k, v in descriptions.items()}


def _compute_all_associations(df_filtered: pd.DataFrame, numeric_cols: set[str]) -> list[dict[str, Any]]:
    associations: list[dict[str, Any]] = []
    all_cols = list(df_filtered.columns)

    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            col_i = all_cols[i]
            col_j = all_cols[j]

            is_i_numeric = col_i in numeric_cols
            is_j_numeric = col_j in numeric_cols

            pair_data = df_filtered[[col_i, col_j]].dropna()
            if len(pair_data) < 2:
                continue

            if is_i_numeric and is_j_numeric:
                corr_val = pair_data[col_i].corr(pair_data[col_j])
                if pd.isna(corr_val):
                    continue

                assoc_value = float(corr_val)
                strength = float(assoc_value**2)
                threshold_metric = strength
                assoc_type = "Pearson r"
                edge_color = "#d62728" if assoc_value < 0 else "#4f81bd"

            elif (not is_i_numeric) and (not is_j_numeric):
                v_value = cramers_v(pair_data[col_i], pair_data[col_j])
                assoc_value = float(v_value)
                strength = assoc_value
                threshold_metric = strength
                assoc_type = "Cramer's V"
                edge_color = "#7f7f7f"

            else:
                if is_i_numeric:
                    eta_value = correlation_ratio(pair_data[col_j], pair_data[col_i])
                else:
                    eta_value = correlation_ratio(pair_data[col_i], pair_data[col_j])

                assoc_value = float(eta_value)
                strength = float(assoc_value**2)
                threshold_metric = strength
                assoc_type = "Eta (eta)"
                edge_color = "#ff7f0e"

            associations.append(
                {
                    "source": col_i,
                    "target": col_j,
                    "strength": strength,
                    "assoc_type": assoc_type,
                    "assoc_value": assoc_value,
                    "threshold_metric": threshold_metric,
                    "edge_color": edge_color,
                    "is_source_numeric": is_i_numeric,
                    "is_target_numeric": is_j_numeric,
                }
            )

    associations.sort(key=lambda row: row["threshold_metric"], reverse=True)
    return associations


def _load_context() -> None:
    global _DATA_ERROR, _DF_FILTERED, _DESCRIPTIONS, _NUMERIC_COLS, _CATEGORICAL_COLS, _ALL_ASSOCIATIONS

    try:
        if not DATA_FILE.exists():
            _DATA_ERROR = f"Missing data file: {DATA_FILE}"
            return

        df = pd.read_csv(DATA_FILE, low_memory=False)
        if df.empty:
            _DATA_ERROR = "The data file is empty."
            return

        selected_cols = [col for col in df.columns if df[col].nunique(dropna=True) > 1]
        if not selected_cols:
            _DATA_ERROR = "No non-constant columns available for association analysis."
            return

        df_filtered = df[selected_cols].copy()
        numeric_cols = set(df_filtered.select_dtypes(include=[np.number]).columns.tolist())
        categorical_cols = set(col for col in df_filtered.columns if col not in numeric_cols)

        _DF_FILTERED = df_filtered
        _DESCRIPTIONS = _load_descriptions()
        _NUMERIC_COLS = numeric_cols
        _CATEGORICAL_COLS = categorical_cols
        _ALL_ASSOCIATIONS = _compute_all_associations(df_filtered, numeric_cols)

        if not _ALL_ASSOCIATIONS:
            _DATA_ERROR = "No valid variable associations could be computed from this dataset."
    except Exception as exc:  # pragma: no cover - defensive for runtime data issues
        _DATA_ERROR = f"Failed to initialize AssociationExplorer: {exc}"


def _safe_description(variable: str) -> str:
    return _DESCRIPTIONS.get(variable, variable)


def _wrap_label(text: str, max_chars: int = 35) -> str:
    """Insert <br> at word boundaries so Plotly axis titles don't get clipped."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).lstrip()
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _association_tooltip(edge: dict[str, Any]) -> str:
    src = edge["source"]
    dst = edge["target"]
    assoc_type = edge["assoc_type"]
    assoc_value = edge["assoc_value"]
    threshold_metric = edge["threshold_metric"]

    if assoc_type == "Pearson r":
        return f"{src} <-> {dst} | Pearson r={assoc_value:.3f} | threshold metric r^2={threshold_metric:.3f}"
    if assoc_type == "Eta (eta)":
        return f"{src} <-> {dst} | Eta={assoc_value:.3f} | threshold metric eta^2={threshold_metric:.3f}"
    return f"{src} <-> {dst} | Cramer's V={assoc_value:.3f}"


def _normalize_threshold(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return DEFAULT_THRESHOLD_RANGE

    threshold_min = float(max(0.0, min(1.0, value[0])))
    threshold_max = float(max(0.0, min(1.0, value[1])))

    if threshold_min <= threshold_max:
        return threshold_min, threshold_max
    return threshold_max, threshold_min


def _build_network_elements(filtered_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not filtered_edges:
        return []

    nodes_in_use = sorted({edge["source"] for edge in filtered_edges}.union({edge["target"] for edge in filtered_edges}))
    elements: list[dict[str, Any]] = []

    for variable in nodes_in_use:
        variable_type = "Numeric" if variable in _NUMERIC_COLS else "Categorical"
        elements.append(
            {
                "data": {
                    "id": variable,
                    "label": variable,
                    "description": _safe_description(variable),
                    "variable_type": variable_type,
                }
            }
        )

    for edge in filtered_edges:
        edge_width = max(1.0, min(12.0, float(edge["strength"]) * 8.0))
        elements.append(
            {
                "data": {
                    "source": edge["source"],
                    "target": edge["target"],
                    "assoc_type": edge["assoc_type"],
                    "assoc_value": edge["assoc_value"],
                    "threshold_metric": edge["threshold_metric"],
                    "tooltip": _association_tooltip(edge),
                },
                "style": {
                    "line-color": edge["edge_color"],
                    "width": edge_width,
                },
            }
        )

    return elements


def _build_numeric_pair_figure(pair_data: pd.DataFrame, edge: dict[str, Any], pair_index: int) -> go.Figure:
    src = edge["source"]
    dst = edge["target"]

    x_values = pair_data[src].astype(float).to_numpy()
    y_values = pair_data[dst].astype(float).to_numpy()

    rng = np.random.default_rng(seed=(pair_index + 17) * 97)
    x_range = float(np.nanmax(x_values) - np.nanmin(x_values)) if len(x_values) > 0 else 0.0
    y_range = float(np.nanmax(y_values) - np.nanmin(y_values)) if len(y_values) > 0 else 0.0

    x_jitter = rng.uniform(-0.02 * x_range, 0.02 * x_range, len(x_values)) if x_range > 0 else np.zeros(len(x_values))
    y_jitter = rng.uniform(-0.02 * y_range, 0.02 * y_range, len(y_values)) if y_range > 0 else np.zeros(len(y_values))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values + x_jitter,
            y=y_values + y_jitter,
            mode="markers",
            customdata=np.column_stack((x_values, y_values)),
            marker={"size": 7, "opacity": 0.6, "color": "#4f81bd"},
            hovertemplate=f"<b>{src}</b>: %{{customdata[0]:.2f}}<br><b>{dst}</b>: %{{customdata[1]:.2f}}<extra></extra>",
            showlegend=False,
        )
    )

    if len(pair_data) >= 2 and x_range > 0:
        try:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            x_line = np.linspace(np.nanmin(x_values), np.nanmax(x_values), 120)
            y_line = slope * x_line + intercept
            trend_color = "#4f81bd"
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line={"color": trend_color, "width": 2},
                    hovertemplate=f"R^2: {edge['threshold_metric']:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
        except Exception:
            pass

    fig.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=_safe_description(src),
        yaxis_title=_safe_description(dst),
        hovermode="closest",
    )
    return fig


def _build_categorical_pair_children(pair_data: pd.DataFrame, edge: dict[str, Any]) -> list:
    src = edge["source"]
    dst = edge["target"]

    contingency = pd.crosstab(pair_data[src], pair_data[dst], margins=True)
    contingency.index.name = None
    contingency.columns.name = None

    table_df = contingency.reset_index()
    first_col = table_df.columns[0]
    if first_col in {None, "index"}:
        table_df = table_df.rename(columns={first_col: src})

    cont_no_margins = pd.crosstab(pair_data[src], pair_data[dst])

    heatmap = go.Figure(
        data=go.Heatmap(
            z=cont_no_margins.values,
            x=[str(col) for col in cont_no_margins.columns],
            y=[str(idx) for idx in cont_no_margins.index],
            colorscale="Blues",
            text=cont_no_margins.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverinfo="skip",
            showscale=False,
        )
    )
    heatmap.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title=_safe_description(dst),
        yaxis_title=_safe_description(src),
    )
    heatmap.update_yaxes(autorange="reversed")

    return [
        html.Div(f"{_safe_description(src)} x {_safe_description(dst)}", className="fw-semibold mb-2"),
        dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True, responsive=True, size="sm"),
        dcc.Graph(figure=heatmap, config={"displayModeBar": False}),
    ]


def _build_mixed_pair_figure(pair_data: pd.DataFrame, edge: dict[str, Any]) -> go.Figure:
    src = edge["source"]
    dst = edge["target"]

    if edge["is_source_numeric"]:
        numeric_var = src
        cat_var = dst
    else:
        numeric_var = dst
        cat_var = src

    means = pair_data.groupby(cat_var)[numeric_var].mean().sort_values(ascending=False)

    fig = px.bar(
        x=[str(idx) for idx in means.index],
        y=means.values,
        labels={"x": _safe_description(cat_var), "y": f"{_safe_description(numeric_var)} (moyenne)"},
    )
    fig.update_traces(
        text=[f"{value:.2f}" for value in means.values],
        textposition="inside",
        hovertemplate=None,
        hoverinfo="skip",
    )
    fig.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
    )
    return fig


def _build_pair_item(edge: dict[str, Any], pair_index: int) -> dbc.AccordionItem | None:
    src = edge["source"]
    dst = edge["target"]
    pair_data = _DF_FILTERED[[src, dst]].dropna()

    if len(pair_data) < 2:
        return None

    title = (
        f"{pair_index}. {src} <-> {dst} | "
        f"{edge['assoc_type']}={edge['assoc_value']:.3f} | metric={edge['threshold_metric']:.3f}"
    )

    is_src_numeric = edge["is_source_numeric"]
    is_dst_numeric = edge["is_target_numeric"]

    if is_src_numeric and is_dst_numeric:
        figure = _build_numeric_pair_figure(pair_data, edge, pair_index)
        content = [dcc.Graph(figure=figure)]
    elif (not is_src_numeric) and (not is_dst_numeric):
        content = _build_categorical_pair_children(pair_data, edge)
    else:
        figure = _build_mixed_pair_figure(pair_data, edge)
        content = [dcc.Graph(figure=figure)]

    return dbc.AccordionItem(content, title=title)


def _build_pair_plots(filtered_edges: list[dict[str, Any]], search_query: str | None) -> list:
    if not filtered_edges:
        return [dbc.Alert("No variable pairs available for the current threshold.", color="warning")]

    query = (search_query or "").strip().lower()
    if query:
        edges_to_render = [
            edge for edge in filtered_edges if query in edge["source"].lower() or query in edge["target"].lower()
        ]
    else:
        edges_to_render = filtered_edges

    if not edges_to_render:
        return [dbc.Alert("No variable pair matches your search query.", color="warning")]

    note_children: list = []
    if len(edges_to_render) > MAX_PAIRS_TO_RENDER:
        note_children.append(
            dbc.Alert(
                f"Showing the first {MAX_PAIRS_TO_RENDER} pairs out of {len(edges_to_render)} for readability.",
                color="info",
            )
        )
        edges_to_render = edges_to_render[:MAX_PAIRS_TO_RENDER]

    items: list[dbc.AccordionItem] = []
    for index, edge in enumerate(edges_to_render, start=1):
        item = _build_pair_item(edge, index)
        if item is not None:
            items.append(item)

    if not items:
        return [dbc.Alert("No plot could be generated for the selected variable pairs.", color="warning")]

    note_children.append(dbc.Accordion(items, start_collapsed=True, always_open=False, flush=False))
    return note_children


def _initial_status_text() -> str:
    if not _ALL_ASSOCIATIONS:
        return "No associations computed yet."

    threshold_min, threshold_max = DEFAULT_THRESHOLD_RANGE
    count = sum(threshold_min <= edge["threshold_metric"] <= threshold_max for edge in _ALL_ASSOCIATIONS)
    return (
        f"Ready: {count} associations in [{threshold_min:.2f}, {threshold_max:.2f}] "
        f"out of {_ALL_ASSOCIATIONS.__len__()} total pairs."
    )


def create_layout():
    if _DATA_ERROR:
        return dbc.Container(
            fluid=True,
            class_name="py-4",
            children=[
                dbc.Alert(
                    [
                        html.H4("AssociationExplorer initialization failed", className="alert-heading"),
                        html.Div(_DATA_ERROR),
                    ],
                    color="danger",
                )
            ],
        )

    dataset_rows = int(_DF_FILTERED.shape[0])
    dataset_cols = int(_DF_FILTERED.shape[1])
    numeric_count = len(_NUMERIC_COLS)
    categorical_count = len(_CATEGORICAL_COLS)

    return dbc.Container(
        fluid=True,
        class_name="py-3",
        children=[
            dcc.Store(id="assoc-filtered-edges-store", data=[]),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2(APP_TITLE, className="mb-1"),
                        html.P(
                            "Dash port of AssociationExplorer-python: thresholded association network and pairwise diagnostics.",
                            className="text-muted mb-3",
                        ),
                        dbc.Row(
                            class_name="g-2",
                            children=[
                                dbc.Col(dbc.Badge(f"Rows: {dataset_rows}", color="secondary", class_name="p-2"), md=2),
                                dbc.Col(dbc.Badge(f"Variables: {dataset_cols}", color="secondary", class_name="p-2"), md=2),
                                dbc.Col(dbc.Badge(f"Numeric: {numeric_count}", color="primary", class_name="p-2"), md=2),
                                dbc.Col(
                                    dbc.Badge(f"Categorical: {categorical_count}", color="warning", class_name="p-2"),
                                    md=2,
                                ),
                                dbc.Col(
                                    dbc.Badge(
                                        f"Computed pairs: {len(_ALL_ASSOCIATIONS)}",
                                        color="dark",
                                        class_name="p-2",
                                    ),
                                    md=3,
                                ),
                            ],
                        ),
                    ]
                ),
                class_name="mb-3 shadow-sm",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Association threshold range", className="fw-semibold mb-2"),
                        dcc.RangeSlider(
                            id="assoc-threshold-range",
                            min=0,
                            max=1,
                            step=0.01,
                            value=list(DEFAULT_THRESHOLD_RANGE),
                            marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
                class_name="mb-3 shadow-sm",
            ),
            dbc.Alert(id="assoc-status", children=_initial_status_text(), color="secondary", class_name="mb-3"),
            dcc.Tabs(
                id="assoc-tabs",
                value="network-tab",
                children=[
                    dcc.Tab(
                        label="Correlation network",
                        value="network-tab",
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        cyto.Cytoscape(
                                            id="assoc-network-graph",
                                            elements=[],
                                            layout={"name": "cose", "animate": False, "fit": True},
                                            stylesheet=NETWORK_STYLESHEET,
                                            style={"width": "100%", "height": "68vh", "border": "1px solid #d6dbe2"},
                                        ),
                                        html.Div(
                                            id="assoc-network-hover",
                                            className="mt-3",
                                            children=[
                                                dbc.Alert(
                                                    "Hover a node or an edge to inspect details.",
                                                    color="light",
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                                class_name="mt-3 shadow-sm",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Pair plots",
                        value="pairs-tab",
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            class_name="g-2 mb-2",
                                            children=[
                                                dbc.Col(
                                                    dbc.Input(
                                                        id="assoc-pair-search",
                                                        type="text",
                                                        placeholder="Filter by variable name...",
                                                        debounce=True,
                                                    ),
                                                    md=6,
                                                )
                                            ],
                                        ),
                                        html.Div(
                                            id="assoc-pair-content",
                                            children=[
                                                dbc.Alert(
                                                    "Pair plots will load from the current thresholded associations.",
                                                    color="light",
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                                class_name="mt-3 shadow-sm",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Help",
                        value="help-tab",
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Markdown(HELP_MARKDOWN, className="mt-2")
                                ),
                                class_name="mt-3 shadow-sm",
                            )
                        ],
                    ),
                ],
            ),
        ],
    )


def register_callbacks():
    global _CALLBACKS_REGISTERED
    if _CALLBACKS_REGISTERED:
        return

    if _DATA_ERROR:
        _CALLBACKS_REGISTERED = True
        return

    @callback(
        Output("assoc-network-graph", "elements"),
        Output("assoc-filtered-edges-store", "data"),
        Output("assoc-status", "children"),
        Output("assoc-status", "color"),
        Input("assoc-threshold-range", "value"),
    )
    def update_network(threshold_range):
        threshold_min, threshold_max = _normalize_threshold(threshold_range)
        filtered_edges = [
            edge
            for edge in _ALL_ASSOCIATIONS
            if threshold_min <= edge["threshold_metric"] <= threshold_max
        ]

        elements = _build_network_elements(filtered_edges)
        if not filtered_edges:
            status = (
                f"No association in [{threshold_min:.2f}, {threshold_max:.2f}]. "
                "Try lowering the minimum threshold."
            )
            return elements, filtered_edges, status, "warning"

        status = (
            f"{len(filtered_edges)} associations shown in [{threshold_min:.2f}, {threshold_max:.2f}] "
            f"out of {len(_ALL_ASSOCIATIONS)} computed pairs."
        )
        return elements, filtered_edges, status, "success"

    @callback(
        Output("assoc-pair-content", "children"),
        Input("assoc-filtered-edges-store", "data"),
        Input("assoc-pair-search", "value"),
        Input("assoc-tabs", "value"),
    )
    def update_pair_content(filtered_edges, search_query, active_tab):
        if active_tab != "pairs-tab":
            return [dbc.Alert("Open the Pair plots tab to inspect pair-level visuals.", color="light")]

        if not isinstance(filtered_edges, list):
            filtered_edges = []

        return _build_pair_plots(filtered_edges, search_query)

    @callback(
        Output("assoc-network-hover", "children"),
        Input("assoc-network-graph", "mouseoverNodeData"),
        Input("assoc-network-graph", "mouseoverEdgeData"),
    )
    def update_hover_info(node_data, edge_data):
        if edge_data:
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(edge_data.get("tooltip", ""), className="mb-1"),
                        html.Small(
                            f"Type: {edge_data.get('assoc_type', '-')}",
                            className="text-muted",
                        ),
                    ]
                ),
                class_name="shadow-sm",
            )

        if node_data:
            variable_name = node_data.get("id", "")
            variable_type = node_data.get("variable_type", "")
            description = node_data.get("description", variable_name)
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(variable_name, className="fw-semibold"),
                        html.Div(description, className="mb-1"),
                        html.Small(f"Variable type: {variable_type}", className="text-muted"),
                    ]
                ),
                class_name="shadow-sm",
            )

        return [dbc.Alert("Hover a node or an edge to inspect details.", color="light")]

    _CALLBACKS_REGISTERED = True


_load_context()


# ---------------------------------------------------------------------------
# Public API used by association_explorer_layout.py
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    source: str
    target: str
    association_type: str
    association_value: float
    strength: float
    edge_color: str
    p_value: float = 1.0


@dataclass
class PairPlotResult:
    figure: go.Figure
    contingency_table: pd.DataFrame | None = None
    data_summary: dict = field(default_factory=dict)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    selected_cols = [col for col in df.columns if df[col].nunique(dropna=True) > 1]
    return df[selected_cols].copy()


def split_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [col for col in df.columns if col not in set(numeric_cols)]
    return numeric_cols, categorical_cols


def compute_all_associations(df: pd.DataFrame) -> list[Edge]:
    df_prepared = prepare_dataframe(df)
    numeric_cols_list, _ = split_column_types(df_prepared)
    numeric_cols_set = set(numeric_cols_list)
    all_cols = list(df_prepared.columns)
    edges: list[Edge] = []

    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            col_i = all_cols[i]
            col_j = all_cols[j]
            is_i_numeric = col_i in numeric_cols_set
            is_j_numeric = col_j in numeric_cols_set
            pair_data = df_prepared[[col_i, col_j]].dropna()
            if len(pair_data) < 2:
                continue

            if is_i_numeric and is_j_numeric:
                corr_val, p_val = pearsonr(pair_data[col_i], pair_data[col_j])
                if pd.isna(corr_val):
                    continue
                assoc_value = float(corr_val)
                strength = float(assoc_value ** 2)
                assoc_type = "Pearson r"
                edge_color = "#d62728" if assoc_value < 0 else "#4f81bd"
            elif not is_i_numeric and not is_j_numeric:
                confusion_matrix = pd.crosstab(pair_data[col_i], pair_data[col_j])
                if confusion_matrix.empty:
                    continue
                chi2_stat, p_val, _, _ = chi2_contingency(confusion_matrix)
                n = confusion_matrix.to_numpy().sum()
                min_dim = min(confusion_matrix.shape) - 1
                v_value = float(np.sqrt(chi2_stat / (n * min_dim))) if (n > 0 and min_dim > 0) else 0.0
                assoc_value = v_value
                # V² puts Cramér's V on the same "proportion of explained variance" scale as
                # Pearson r² and R²_adj, enabling cross-metric strength comparisons (Cohen 1988).
                strength = v_value ** 2
                assoc_type = "Cramér's V"
                edge_color = "#7f7f7f"
            else:
                num_col = col_i if is_i_numeric else col_j
                cat_col = col_j if is_i_numeric else col_i
                eta_value = correlation_ratio(pair_data[cat_col], pair_data[num_col])
                eta_sq = float(eta_value ** 2)
                n_pairs = len(pair_data)
                k = int(pair_data[cat_col].nunique())
                # R²_adj = 1 − (1 − η²) × (n−1)/(n−k), floored at 0 (ANOVA adjusted R²).
                # More conservative than η²; comparable to r² and V² for cross-metric ranking.
                if n_pairs > k and k > 1:
                    r2_adj = max(0.0, 1.0 - (1.0 - eta_sq) * (n_pairs - 1) / (n_pairs - k))
                else:
                    r2_adj = 0.0
                groups = [grp.values for _, grp in pair_data.groupby(cat_col)[num_col] if len(grp) > 0]
                if len(groups) >= 2:
                    _, p_val = f_oneway(*groups)
                    p_val = float(p_val) if not np.isnan(p_val) else 1.0
                else:
                    p_val = 1.0
                assoc_value = float(r2_adj)
                strength = float(r2_adj)
                assoc_type = "R² adj."
                edge_color = "#ff7f0e"

            edges.append(Edge(
                source=col_i,
                target=col_j,
                association_type=assoc_type,
                association_value=assoc_value,
                strength=strength,
                edge_color=edge_color,
                p_value=float(p_val),
            ))

    edges.sort(key=lambda e: e.strength, reverse=True)
    return edges


def load_descriptions_from_dataframe(desc_df: pd.DataFrame) -> dict[str, str]:
    if not {"Variable", "Description"}.issubset(desc_df.columns):
        return {}
    descriptions = desc_df.set_index("Variable")["Description"].to_dict()
    return {str(k): str(v) for k, v in descriptions.items()}


def _data_summary_numeric_numeric(
    pair_data: pd.DataFrame, src: str, dst: str, r: float
) -> dict:
    x = pair_data[src].astype(float)
    y = pair_data[dst].astype(float)
    return {
        "type": "numeric_numeric",
        "var1": src,
        "var2": dst,
        "n": int(len(pair_data)),
        "r": round(float(r), 3),
        "direction": "positive" if r >= 0 else "négative",
        "var1_mean": round(float(x.mean()), 2),
        "var1_std": round(float(x.std()), 2),
        "var2_mean": round(float(y.mean()), 2),
        "var2_std": round(float(y.std()), 2),
    }


def _data_summary_categorical_categorical(
    pair_data: pd.DataFrame, src: str, dst: str
) -> dict:
    contingency = pd.crosstab(pair_data[src], pair_data[dst])
    n = int(contingency.to_numpy().sum())
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"type": "categorical_categorical", "var1": src, "var2": dst, "n": n, "top_cells": []}
    _, _, _, expected = chi2_contingency(contingency)
    top_cells = []
    for i, row_label in enumerate(contingency.index):
        for j, col_label in enumerate(contingency.columns):
            obs = int(contingency.iloc[i, j])
            exp = float(expected[i, j])
            std_resid = round((obs - exp) / (exp ** 0.5), 2) if exp > 0 else 0.0
            top_cells.append({
                "var1_value": str(row_label),
                "var2_value": str(col_label),
                "observed": obs,
                "expected": round(exp, 1),
                "std_residual": std_resid,
            })
    top_cells.sort(key=lambda c: abs(c["std_residual"]), reverse=True)
    return {
        "type": "categorical_categorical",
        "var1": src,
        "var2": dst,
        "n": n,
        "top_cells": top_cells[:3],
    }


def _data_summary_numeric_categorical(
    pair_data: pd.DataFrame, numeric_var: str, cat_var: str
) -> dict:
    agg = (
        pair_data.groupby(cat_var)[numeric_var]
        .agg(mean="mean", count="count")
        .sort_values("mean", ascending=False)
    )
    group_means = [
        {"group": str(grp), "mean": round(float(row["mean"]), 2), "n": int(row["count"])}
        for grp, row in agg.iterrows()
    ]
    highest = group_means[0]["group"] if group_means else ""
    lowest = group_means[-1]["group"] if group_means else ""
    diff = round(group_means[0]["mean"] - group_means[-1]["mean"], 2) if len(group_means) >= 2 else 0.0
    return {
        "type": "numeric_categorical",
        "numeric_var": numeric_var,
        "cat_var": cat_var,
        "n": int(len(pair_data)),
        "group_means": group_means,
        "highest_group": highest,
        "lowest_group": lowest,
        "mean_difference": diff,
    }


def build_pair_plot(
    edge: Edge,
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    descriptions: dict[str, str],
) -> PairPlotResult:
    src = edge.source
    dst = edge.target
    pair_data = dataframe[[src, dst]].dropna()

    if len(pair_data) < 2:
        raise ValueError(f"Not enough data for pair ({src}, {dst})")

    numeric_set = set(numeric_columns)
    is_src_numeric = src in numeric_set
    is_dst_numeric = dst in numeric_set

    def _desc(var: str) -> str:
        return _wrap_label(descriptions.get(var, var))

    if is_src_numeric and is_dst_numeric:
        x_values = pair_data[src].astype(float).to_numpy()
        y_values = pair_data[dst].astype(float).to_numpy()

        rng = np.random.default_rng(seed=42)
        x_range = float(np.nanmax(x_values) - np.nanmin(x_values)) if len(x_values) > 0 else 0.0
        y_range = float(np.nanmax(y_values) - np.nanmin(y_values)) if len(y_values) > 0 else 0.0
        x_jitter = rng.uniform(-0.02 * x_range, 0.02 * x_range, len(x_values)) if x_range > 0 else np.zeros(len(x_values))
        y_jitter = rng.uniform(-0.02 * y_range, 0.02 * y_range, len(y_values)) if y_range > 0 else np.zeros(len(y_values))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values + x_jitter,
            y=y_values + y_jitter,
            mode="markers",
            customdata=np.column_stack((x_values, y_values)),
            marker={"size": 7, "opacity": 0.6, "color": "#4f81bd"},
            hovertemplate=f"<b>{src}</b>: %{{customdata[0]:.2f}}<br><b>{dst}</b>: %{{customdata[1]:.2f}}<extra></extra>",
            showlegend=False,
        ))
        if x_range > 0:
            try:
                slope, intercept = np.polyfit(x_values, y_values, 1)
                x_line = np.linspace(np.nanmin(x_values), np.nanmax(x_values), 120)
                y_line = slope * x_line + intercept
                trend_color = "#4f81bd"
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line={"color": trend_color, "width": 2},
                    hovertemplate=f"r²: {edge.strength:.3f}<extra></extra>",
                    showlegend=False,
                ))
            except Exception:
                pass
        fig.update_layout(
            height=390,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title=_desc(src),
            yaxis_title=_desc(dst),
            hovermode="closest",
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        return PairPlotResult(
            figure=fig,
            contingency_table=None,
            data_summary=_data_summary_numeric_numeric(pair_data, src, dst, edge.association_value),
        )

    elif not is_src_numeric and not is_dst_numeric:
        contingency = pd.crosstab(pair_data[src], pair_data[dst])
        cont_with_margins = pd.crosstab(pair_data[src], pair_data[dst], margins=True)
        cont_with_margins.index.name = None
        cont_with_margins.columns.name = None

        heatmap = go.Figure(data=go.Heatmap(
            z=contingency.values,
            x=[str(c) for c in contingency.columns],
            y=[str(i) for i in contingency.index],
            colorscale="Blues",
            text=contingency.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverinfo="skip",
            showscale=False,
        ))
        heatmap.update_layout(
            height=390,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title=_desc(dst),
            yaxis_title=_desc(src),
        )
        heatmap.update_xaxes(automargin=True)
        heatmap.update_yaxes(automargin=True, autorange="reversed")
        return PairPlotResult(
            figure=heatmap,
            contingency_table=cont_with_margins,
            data_summary=_data_summary_categorical_categorical(pair_data, src, dst),
        )

    else:
        numeric_var = src if is_src_numeric else dst
        cat_var = dst if is_src_numeric else src
        means = pair_data.groupby(cat_var)[numeric_var].mean().sort_values(ascending=False)
        fig = px.bar(
            x=[str(i) for i in means.index],
            y=means.values,
            labels={"x": _desc(cat_var), "y": f"{_desc(numeric_var)} (moyenne)"},
        )
        fig.update_traces(
            text=[f"{v:.2f}" for v in means.values],
            textposition="inside",
            hovertemplate=None,
            hoverinfo="skip",
        )
        fig.update_layout(
            height=390,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            showlegend=False,
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        return PairPlotResult(
            figure=fig,
            contingency_table=None,
            data_summary=_data_summary_numeric_categorical(pair_data, numeric_var, cat_var),
        )


def compute_data_summaries(
    edges: list[Edge],
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[tuple[str, str], dict]:
    """Compute data_summary dicts for a list of edges without building Plotly figures.

    Applies the same branching logic as build_pair_plot() but only extracts the
    salient statistics (means, residuals, correlation direction) needed by the LLM.
    Returns a dict mapping (source, target) -> data_summary.
    Edges with insufficient data or unexpected errors are silently skipped.
    """
    numeric_set = set(numeric_columns)
    result: dict[tuple[str, str], dict] = {}

    for edge in edges:
        src, dst = edge.source, edge.target
        try:
            pair_data = dataframe[[src, dst]].dropna()
            if len(pair_data) < 2:
                continue
            is_src_numeric = src in numeric_set
            is_dst_numeric = dst in numeric_set
            if is_src_numeric and is_dst_numeric:
                summary = _data_summary_numeric_numeric(pair_data, src, dst, edge.association_value)
            elif not is_src_numeric and not is_dst_numeric:
                summary = _data_summary_categorical_categorical(pair_data, src, dst)
            else:
                numeric_var = src if is_src_numeric else dst
                cat_var = dst if is_src_numeric else src
                summary = _data_summary_numeric_categorical(pair_data, numeric_var, cat_var)
            result[(src, dst)] = summary
        except Exception:
            pass

    return result