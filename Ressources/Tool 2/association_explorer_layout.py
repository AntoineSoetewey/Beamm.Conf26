from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from dash import Input, Output, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

# The library file has a hyphenated name so it cannot be imported with a
# regular import statement; use importlib instead.
# The module must be registered in sys.modules before exec_module so that
# @dataclass can resolve its own module reference.
_spec = importlib.util.spec_from_file_location(
    "association_explorer",
    Path(__file__).resolve().parents[1] / "utils" / "association-explorer.py",
)
_ae = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["association_explorer"] = _ae
_spec.loader.exec_module(_ae)  # type: ignore[union-attr]

_llm_spec = importlib.util.spec_from_file_location(
    "llm_judge",
    Path(__file__).resolve().parents[1] / "utils" / "llm_judge.py",
)
_llm = importlib.util.module_from_spec(_llm_spec)  # type: ignore[arg-type]
sys.modules["llm_judge"] = _llm
_llm_spec.loader.exec_module(_llm)  # type: ignore[union-attr]

APP_TITLE = "Explorateur d'associations"
TOP_N = 10

# Minimum association strength thresholds — Cohen (1988) "small" effect size (~1 % explained variance).
# All three metrics map to approximately r² ≥ 0.01 on the common explained-variance scale,
# ensuring that the same effective cut-off applies regardless of metric type.
# Pearson |r| ≥ 0.10  →  r² ≥ 0.01  (Cohen 1988, r_small = 0.10)
MIN_PEARSON_R: float = 0.10
# Cramér V ≥ 0.10  →  V² ≥ 0.01  (Cohen 1988: w_small = 0.10 for df_min = 1, V ≡ w)
MIN_CRAMERS_V: float = 0.10
# R²_adj ≥ 0.01  — directly comparable to r² ≥ 0.01 (Cohen 1988 small η² ≈ 0.01–0.02)
MIN_R2_ADJ: float = 0.01

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "data.csv"
DESCRIPTION_FILE = BASE_DIR / "data" / "description.csv"

NETWORK_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": 16,
            "font-weight": "bold",
            "color": "#1f2933",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.75,
            "text-background-padding": "2px",
            "background-color": "#dbe6f2",
            "border-color": "#4f81bd",
            "border-width": 1.5,
            "text-wrap": "wrap",
            "text-max-width": 140,
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 6,
            "width": 36,
            "height": 36,
            "min-zoomed-font-size": 10,
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "line-color": "#4f81bd",
            "width": 2,
        },
    },
    {
        "selector": ":selected",
        "style": {
            "border-width": 3,
            "border-color": "#1f2933",
            "line-color": "#1f2933",
        },
    },
]

HELP_MARKDOWN = """
### À quoi sert cette application

Cette application vous aide à explorer quelles variables du jeu de données sont le plus fortement associées au sujet de votre requête. Elle montre les associations les plus fortes à la fois entre les variables identifiées par votre requête et entre ces variables et le reste du jeu de données.

**Une association forte ne signifie pas une causalité.** Elle indique que deux variables sont liées, pas que l'une cause l'autre. Cherchez toujours des explications alternatives avant de tirer des conclusions.

### Comment l'utiliser

1. Ouvrez l'onglet **Réseau** pour avoir une vue d'ensemble des associations les plus fortes. Utilisez-le pour repérer des clusters et identifier des paires à approfondir.
2. **Cliquez sur une ligne** dans le réseau pour ouvrir l'association correspondante dans l'onglet **Association**.

### Quelles variables apparaissent dans le réseau

Le réseau est construit en trois étapes :

1. Un ensemble de variables est sélectionné en fonction de votre requête (indiqué dans le sous-titre comme *N variables issues de la requête*).
2. Les paires sont **filtrées** selon deux critères cumulatifs, appliqués à toutes les paires impliquant au moins une variable de la requête :
   - **Significativité statistique** : seules les paires avec une p-valeur < 0,05 sont conservées.
   - **Force minimale** : les associations négligeables sont exclues — seuils fondés sur la convention « petit effet » de Cohen (1988), correspondant à environ 1 % de variance expliquée pour chaque type de métrique :
     - Variables numériques–numériques : |r de Pearson| ≥ {min_r} (r² ≥ 0,01)
     - Variables catégorielles–catégorielles : V de Cramér ≥ {min_v} (V² ≥ 0,01)
     - Numérique–catégorielle : R² ajusté ≥ {min_r2}
3. Parmi les paires restantes, les **{top_n} plus fortes** (force d'association décroissante, sur une échelle de variance expliquée commune aux trois types) sont conservées. Le réseau peut donc inclure des variables du reste du jeu de données si elles sont fortement associées à une variable de la requête.

Les variables de la requête qui ne figurent dans aucune des {top_n} premières paires sont exclues du réseau (indiqué dans le sous-titre).

### Mesures d'association utilisées

- **Numérique–numérique** : r de Pearson (affiché) ; r² comme force.
- **Catégorielle–catégorielle** : V de Cramér (affiché) ; V² comme force.
- **Numérique–catégorielle** : R² ajusté du modèle ANOVA, R² ajusté = 1 − (1 − η²) × (n−1)/(n−k), où n est le nombre d'observations et k le nombre de catégories. Plus conservateur qu'η², il pénalise les modèles avec de nombreuses catégories.

Exprimer chaque force en termes de variance expliquée (r², V², R² ajusté) permet un classement cohérent entre les trois types.

### Lire le réseau

Chaque point représente une variable. Chaque ligne relie deux variables associées.

- **La distance entre les nœuds** est inversement proportionnelle à la force statistique ; les nœuds proches ont des associations plus fortes.
- **L'opacité des lignes** est proportionnelle au score d'intérêt journalistique ; les lignes plus opaques représentent des associations plus surprenantes.

La position des points à l'écran est approximative et ne porte pas de signification propre ; concentrez-vous sur les lignes, pas sur la disposition.

### Lire les associations

Cliquez sur une ligne dans l'onglet Réseau pour ouvrir l'association correspondante. L'association montre la distribution des deux variables et la nature de leur association.

### Score d'intérêt journalistique

En plus d'un visuel de l'association entre les deux variables, un **score d'intérêt journalistique** (0–1) est affiché. Il est le produit de deux composantes indépendantes :

- **Surprise sémantique** (0–1) : moyenne de deux sous-dimensions évaluées par un modèle de langage, qui ignorent délibérément la force statistique :
  - *Documentation dans la littérature* : dans quelle mesure cette association est-elle établie scientifiquement ? (0 = consensus documenté, 1 = rarement ou jamais publiée)
  - *Plausibilité théorique* : les cadres théoriques et empiriques existants permettaient-ils d'anticiper cette association ? (0 = pleinement attendue, 1 = aucun cadre connu ne permettait de l'anticiper)
- **Fiabilité statistique** (0–1) : calculée à partir de la p-valeur uniquement — plus la p-valeur est faible, plus la fiabilité est élevée. Le score croît sans saturation, ce qui permet de distinguer des p-valeurs même très petites.

`intérêt journalistique = surprise_sémantique × fiabilité_statistique`

Un score élevé signifie que l'association est à la fois surprenante et statistiquement solide ; le candidat potentiel idéal pour un article de fond.

Une **piste de réflexion** générée automatiquement est affichée sous chaque graphique. Elle contextualise la force de l'association et les sous-scores de surprise pour proposer des hypothèses que l'association pourrait soulever, sans tirer de conclusion : c'est au journaliste d'interpréter.

Le raisonnement préalable du modèle, qui a guidé l'attribution des scores, est également consultable en cliquant sur « Voir le raisonnement du modèle » sous les sous-scores.
""".format(top_n=TOP_N, min_r=MIN_PEARSON_R, min_v=MIN_CRAMERS_V, min_r2=MIN_R2_ADJ)



_TYPE_PLAIN = {
    "Pearson r": "deux variables numériques",
    "Cramér's V": "deux variables catégorielles",
    "R² adj.": "numérique vs catégorielle",
}

_CALLBACKS_REGISTERED = False
_DATA_ERROR: str | None = None
_DF_PREPARED: pd.DataFrame = pd.DataFrame()
_NUMERIC_COLS: list[str] = []
_DESCRIPTIONS: dict[str, str] = {}
_TOP_EDGES: list = []
_SELECTED_VARIABLES_COUNT: int | None = None
_SELECTED_VARIABLES_SET: set[str] = set()
_NEWSWORTHINESS: dict[tuple[str, str], dict] = {}
_NEWSWORTHINESS_AVAILABLE: bool = False
_NEWSWORTHINESS_ERROR: str | None = None


def _passes_strength_filter(edge) -> bool:
    """Return True if the edge meets its metric-specific minimum strength threshold."""
    if edge.association_type == "Pearson r":
        return abs(edge.association_value) >= MIN_PEARSON_R
    if edge.association_type == "Cramér's V":
        return edge.association_value >= MIN_CRAMERS_V
    # R² adj. (numeric-categorical)
    return edge.association_value >= MIN_R2_ADJ


def _load_context(selected_variables: list[str] | None = None) -> None:
    global _DATA_ERROR, _DF_PREPARED, _NUMERIC_COLS, _DESCRIPTIONS, _TOP_EDGES
    global _SELECTED_VARIABLES_COUNT, _SELECTED_VARIABLES_SET
    global _NEWSWORTHINESS, _NEWSWORTHINESS_AVAILABLE, _NEWSWORTHINESS_ERROR

    try:
        if not DATA_FILE.exists():
            _DATA_ERROR = f"Fichier de données manquant : {DATA_FILE}"
            return

        df = pd.read_csv(DATA_FILE, low_memory=False)

        if selected_variables is not None:
            missing = [v for v in selected_variables if v not in df.columns]
            if missing:
                _DATA_ERROR = f"Variables sélectionnées introuvables dans le jeu de données : {missing}"
                return
            _SELECTED_VARIABLES_COUNT = len(selected_variables)
            _SELECTED_VARIABLES_SET = set(selected_variables)

        _DF_PREPARED = _ae.prepare_dataframe(df)
        _NUMERIC_COLS, _ = _ae.split_column_types(_DF_PREPARED)

        all_edges = _ae.compute_all_associations(df)

        if selected_variables is not None:
            selected_set = set(selected_variables)
            all_edges = [
                e for e in all_edges
                if e.source in selected_set or e.target in selected_set
            ]

        # Filter 1: statistical significance
        all_edges = [e for e in all_edges if e.p_value < 0.05]

        # Filter 2: negligible associations (below Cohen's small effect for each metric)
        all_edges = [e for e in all_edges if _passes_strength_filter(e)]

        if not all_edges:
            _DATA_ERROR = (
                "Aucune association ne satisfait les critères de filtrage "
                "(p < 0,05 et force minimale selon Cohen 1988). "
                "Essayez d'élargir la liste des variables ou de vérifier le jeu de données."
            )
            return

        # Sort by association strength (r²-equivalent scale) descending, then cap at TOP_N.
        # r², V², and R²_adj are all on the "proportion of variance explained" scale,
        # making cross-metric comparisons coherent (Cohen 1988).
        all_edges.sort(key=lambda e: e.strength, reverse=True)
        _TOP_EDGES = all_edges[:TOP_N]

        if DESCRIPTION_FILE.exists():
            desc_df = pd.read_csv(DESCRIPTION_FILE)
            _DESCRIPTIONS = _ae.load_descriptions_from_dataframe(desc_df)

        try:
            data_summaries = _ae.compute_data_summaries(
                _TOP_EDGES, _DF_PREPARED, _NUMERIC_COLS
            )
            _NEWSWORTHINESS = _llm.score_newsworthiness(
                _TOP_EDGES, _DESCRIPTIONS, data_summaries
            )
            _NEWSWORTHINESS_AVAILABLE = True
        except Exception as llm_exc:
            _NEWSWORTHINESS_AVAILABLE = False
            _NEWSWORTHINESS_ERROR = str(llm_exc)

    except Exception as exc:
        _DATA_ERROR = f"Échec de l'initialisation de l'Association Explorer : {exc}"


def _css_selector(raw_id: str) -> str:
    return "#" + re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", raw_id)


def _compute_layout_positions(edges: list) -> dict[str, dict[str, float]]:
    """Single Kamada-Kawai on the full graph with a globally consistent distance matrix.

    - Direct edges: dist in [1, 3] using global min/max normalization (strong=1, weak=3)
    - Non-adjacent pairs within same component: shortest-path sum of edge distances
    - Pairs in different components: large separation (8) so clusters don't overlap
    """
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge.source, edge.target, weight=edge.strength)
    if len(G) == 0:
        return {}

    nodes = list(G.nodes())

    # 1. Globally normalize strengths → distances [1, 3]
    all_strengths = [d["weight"] for _, _, d in G.edges(data=True)]
    g_min, g_max = min(all_strengths), max(all_strengths)
    g_range = g_max - g_min if g_max > g_min else 1.0

    # Build a helper graph with transformed distances as edge weights
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for u, v, data in G.edges(data=True):
        norm = (data["weight"] - g_min) / g_range   # 0 = weakest, 1 = strongest
        H.add_edge(u, v, dist=1.0 + 2.0 * (1.0 - norm))  # strong→1.0, weak→3.0

    # 2. All-pairs shortest paths within each component (uses transformed distances)
    inter = 8.0
    dist_matrix: dict = {u: {v: inter for v in nodes} for u in nodes}
    for u in nodes:
        dist_matrix[u][u] = 0.0
    for comp in nx.connected_components(G):
        lengths = dict(nx.all_pairs_dijkstra_path_length(H.subgraph(comp), weight="dist"))
        for u, row in lengths.items():
            for v, d in row.items():
                dist_matrix[u][v] = d

    # 3. Single global Kamada-Kawai — no post-processing needed
    pos = nx.kamada_kawai_layout(G, dist=dist_matrix)
    scale = 300.0
    return {n: {"x": float(xy[0] * scale), "y": float(xy[1] * scale)} for n, xy in pos.items()}




def _statistical_reliability(p_value: float) -> float:
    """Score de fiabilité statistique basé uniquement sur la p-valeur.

    Formule : -log10(p) / (-log10(p) + 3), croissance asymptotique vers 1.
    Pas de saturation — distingue des p-valeurs même très petites.
    """
    clamped_p = max(p_value, 1e-300)
    log_p = -math.log10(clamped_p)
    return log_p / (log_p + 3.0)


def _build_cytoscape_elements(edges: list) -> list[dict]:
    if not edges:
        return []

    nodes_in_use = sorted({e.source for e in edges} | {e.target for e in edges})

    # Collect semantic surprise scores from LLM and compute combined newsworthiness
    # combined = semantic_surprise × statistical_reliability
    raw_llm: list[float | None] = []
    for edge in edges:
        nw = _NEWSWORTHINESS.get((edge.source, edge.target)) or _NEWSWORTHINESS.get((edge.target, edge.source))
        raw_llm.append(nw["score"] if nw else None)

    combined_scores: list[float | None] = []
    for i, edge in enumerate(edges):
        sem = raw_llm[i]
        if _NEWSWORTHINESS_AVAILABLE and sem is not None:
            reliability = _statistical_reliability(edge.p_value)
            combined_scores.append(sem * reliability)
        else:
            combined_scores.append(None)

    if _NEWSWORTHINESS_AVAILABLE:
        valid = [s for s in combined_scores if s is not None]
        score_min = min(valid) if valid else 0.0
        score_max = max(valid) if valid else 1.0
        score_range = score_max - score_min if score_max > score_min else 1.0
    else:
        score_min = score_max = score_range = 1.0

    # First pass: compute per-edge alphas from combined newsworthiness
    edge_records = []
    for i, edge in enumerate(edges):
        combined = combined_scores[i]

        tooltip = f"{edge.source} — {edge.target}"
        if combined is not None:
            tooltip += f" · intérêt journalistique : {combined:.2f}"

        if _NEWSWORTHINESS_AVAILABLE and combined is not None:
            normalized = (combined - score_min) / score_range
            alpha = round(0.1 + 0.9 * normalized, 3)
        else:
            alpha = 0.85

        edge_records.append({
            "idx": i, "source": edge.source, "target": edge.target,
            "association_type": edge.association_type,
            "association_value": edge.association_value,
            "tooltip": tooltip, "alpha": alpha,
        })

    # Node alpha = max alpha of all connected edges
    node_alpha: dict[str, float] = {v: 0.0 for v in nodes_in_use}
    for er in edge_records:
        node_alpha[er["source"]] = max(node_alpha[er["source"]], er["alpha"])
        node_alpha[er["target"]] = max(node_alpha[er["target"]], er["alpha"])

    positions = _compute_layout_positions(edges)
    elements: list[dict] = []

    for variable in nodes_in_use:
        variable_type = "Numeric" if variable in _NUMERIC_COLS else "Categorical"
        elem: dict = {
            "data": {
                "id": variable,
                "label": variable,
                "description": _DESCRIPTIONS.get(variable, variable),
                "variable_type": variable_type,
                "alpha": node_alpha[variable],
            }
        }
        if variable in positions:
            elem["position"] = positions[variable]
        elements.append(elem)

    for er in edge_records:
        elements.append({
            "data": {
                "id": f"e{er['idx']}",
                "source": er["source"],
                "target": er["target"],
                "association_type": er["association_type"],
                "association_value": er["association_value"],
                "tooltip": er["tooltip"],
                "alpha": er["alpha"],
            },
        })

    return elements


def _newsworthiness_badge(edge) -> list:
    """Score global + deux sous-scores — affiché avant le graphique."""
    if not _NEWSWORTHINESS_AVAILABLE:
        return []
    nw = _NEWSWORTHINESS.get((edge.source, edge.target)) or _NEWSWORTHINESS.get((edge.target, edge.source))
    if not nw:
        return []
    semantic_surprise = nw["score"]
    reliability = _statistical_reliability(edge.p_value)
    combined = semantic_surprise * reliability
    badge_color = "success" if combined >= 0.4 else "warning" if combined >= 0.2 else "secondary"

    doc_score = nw.get("documentation_litterature")
    plaus_score = nw.get("plausibilite_theorique")
    justif_doc = nw.get("justification_documentation", "")
    justif_plaus = nw.get("justification_plausibilite", "")

    small_label = {"fontSize": "0.78rem", "fontWeight": "600", "color": "#374151"}
    small_text  = {"fontSize": "0.78rem", "color": "#6b7280"}
    small_value = {"fontSize": "0.78rem", "color": "#374151", "fontWeight": "500"}

    children = [
        html.Div(
            [
                dbc.Badge(
                    f"Intérêt journalistique : {combined:.2f}",
                    color=badge_color,
                    className="me-2",
                    style={"fontSize": "0.75rem"},
                ),
                html.Span(
                    f"surprise sémantique : {semantic_surprise:.2f} · fiabilité statistique : {reliability:.2f}",
                    style={"fontSize": "0.75rem", "color": "#6b7280"},
                ),
            ]
        ),
    ]

    raisonnement = nw.get("raisonnement", "")

    if doc_score is not None and plaus_score is not None:
        subscore_children = [
            html.Span("Surprise sémantique :", style=small_label),
            html.Div(
                [
                    html.Span(f"Documentation littérature : {doc_score:.2f}", style=small_value),
                    html.Span(f" — {justif_doc}", style=small_text),
                ],
                style={"marginTop": "3px"},
            ),
            html.Div(
                [
                    html.Span(f"Plausibilité théorique : {plaus_score:.2f}", style=small_value),
                    html.Span(f" — {justif_plaus}", style=small_text),
                ],
                style={"marginTop": "3px"},
            ),
        ]
        if raisonnement:
            subscore_children.append(
                html.Details(
                    [
                        html.Summary(
                            "Voir le raisonnement du modèle",
                            style={"fontSize": "0.75rem", "color": "#9ca3af", "cursor": "pointer", "userSelect": "none", "marginTop": "6px"},
                        ),
                        html.P(
                            raisonnement,
                            style={"fontSize": "0.78rem", "color": "#6b7280", "marginTop": "6px", "marginBottom": "0"},
                        ),
                    ],
                )
            )
        children.append(html.Div(subscore_children, style={"marginTop": "8px"}))

    return [html.Div(children, className="mb-2")]


def _newsworthiness_interpretation(edge) -> list:
    """Interprétation journalistique — affichée après le graphique."""
    if not _NEWSWORTHINESS_AVAILABLE:
        return []
    nw = _NEWSWORTHINESS.get((edge.source, edge.target)) or _NEWSWORTHINESS.get((edge.target, edge.source))
    if not nw:
        return []
    interpretation = nw.get("interpretation", "")
    if not interpretation:
        return []
    return [
        html.Div(
            [
                html.Span("Piste de réflexion : ", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#374151"}),
                html.Span(interpretation, style={"fontSize": "0.78rem", "color": "#6b7280"}),
            ],
            style={"marginTop": "10px"},
        )
    ]


def _build_pair_accordion(exact_pair: dict) -> list:
    if not _TOP_EDGES:
        return [dbc.Alert("Aucune association à afficher.", color="warning")]

    src = exact_pair["source"].lower()
    tgt = exact_pair["target"].lower()
    matching = [
        e for e in _TOP_EDGES
        if (e.source.lower() == src and e.target.lower() == tgt)
        or (e.source.lower() == tgt and e.target.lower() == src)
    ]

    if not matching:
        return [dbc.Alert("Cette paire ne figure pas dans les résultats principaux.", color="warning")]

    edge = matching[0]
    try:
        result = _ae.build_pair_plot(
            edge=edge,
            dataframe=_DF_PREPARED,
            numeric_columns=_NUMERIC_COLS,
            descriptions=_DESCRIPTIONS,
        )
    except ValueError:
        return [dbc.Alert("Impossible de générer un graphique pour cette paire.", color="warning")]

    return [
        html.H5(
            f"{edge.source} — {edge.target}",
            className="mb-1",
            style={"fontWeight": "600", "color": "#111827"},
        ),
        *_newsworthiness_badge(edge),
        dcc.Graph(figure=result.figure, config={"displayModeBar": False}),
        *_newsworthiness_interpretation(edge),
    ]


def _build_query_coverage_note(
    selected_count: int | None,
    selected_in_network: int,
) -> list:
    if selected_count is None:
        return []

    sep = html.Span(" · ")
    excluded = selected_count - selected_in_network

    query_part = html.Span(f"{selected_count} variables issues de la requête")

    in_network_part = html.Span(
        f"{selected_in_network} sur {selected_count} apparaissent dans le réseau"
        + (f" ({excluded} exclues — aucune association suffisamment forte)" if excluded > 0 else "")
    )

    return [query_part, sep, in_network_part]


def create_layout():
    if _DATA_ERROR:
        return dbc.Container(
            fluid=True,
            class_name="py-4",
            children=[
                dbc.Alert(
                    [
                        html.H4("Échec de l'initialisation de l'Association Explorer", className="alert-heading"),
                        html.Div(_DATA_ERROR),
                    ],
                    color="danger",
                )
            ],
        )

    nodes_in_network_set = {e.source for e in _TOP_EDGES} | {e.target for e in _TOP_EDGES}
    selected_in_network = len(nodes_in_network_set & _SELECTED_VARIABLES_SET)

    network_elements = _build_cytoscape_elements(_TOP_EDGES)
    opacity_rules = [
        {
            "selector": _css_selector(elem["data"]["id"]),
            "style": {"opacity": elem["data"]["alpha"]},
        }
        for elem in network_elements
        if "alpha" in elem.get("data", {})
    ]
    network_stylesheet = NETWORK_STYLESHEET + opacity_rules

    if _NEWSWORTHINESS_AVAILABLE:
        encoding_hint = (
            "Distance entre nœuds = force statistique · opacité des lignes = intérêt journalistique"
            " · cliquez sur une ligne pour afficher l'association"
        )
    else:
        encoding_hint = (
            "Distance entre nœuds = force statistique"
            " · cliquez sur une ligne pour afficher l'association"
        )

    tab_style = {
        "borderRadius": "6px 6px 0 0",
        "padding": "8px 20px",
        "fontWeight": "500",
        "fontSize": "0.875rem",
        "border": "1px solid #e5e7eb",
        "borderBottom": "none",
        "background": "#f9fafb",
        "color": "#6b7280",
    }
    tab_selected_style = {
        **tab_style,
        "background": "#ffffff",
        "color": "#111827",
        "borderTop": "2px solid #4f81bd",
    }

    return dbc.Container(
        fluid=True,
        style={"maxWidth": "1200px"},
        class_name="py-4 px-4",
        children=[
            dcc.Store(id="assoc-exact-pair", data=None),
            # Header
            html.Div(
                className="mb-4",
                children=[
                    html.H4(APP_TITLE, style={"fontWeight": "600", "color": "#111827", "marginBottom": "4px"}),
                    html.Div(
                        style={"fontSize": "0.8rem", "color": "#9ca3af", "lineHeight": "1.6"},
                        children=_build_query_coverage_note(
                            selected_count=_SELECTED_VARIABLES_COUNT,
                            selected_in_network=selected_in_network,
                        ),
                    ),
                ],
            ),
            # Tabs
            dcc.Tabs(
                id="assoc-tabs",
                value="network-tab",
                style={"borderBottom": "1px solid #e5e7eb"},
                children=[
                    dcc.Tab(
                        label="Réseau",
                        value="network-tab",
                        style=tab_style,
                        selected_style=tab_selected_style,
                        children=[
                            html.Div(
                                style={"border": "1px solid #e5e7eb", "borderTop": "none", "borderRadius": "0 6px 6px 6px", "background": "#fff", "padding": "16px"},
                                children=[
                                    html.Div(
                                        encoding_hint,
                                        style={"fontSize": "0.78rem", "color": "#9ca3af", "marginBottom": "8px"},
                                    ),
                                    cyto.Cytoscape(
                                        id="assoc-network-graph",
                                        elements=network_elements,
                                        layout={"name": "preset", "fit": True, "padding": 40},
                                        stylesheet=network_stylesheet,
                                        style={"width": "100%", "height": "72vh"},
                                    ),
                                    html.Div(
                                        id="assoc-network-hover",
                                        style={"marginTop": "10px", "minHeight": "48px"},
                                        children=[
                                            html.Span(
                                                "Survolez une variable ou une ligne pour afficher les détails.",
                                                style={"fontSize": "0.8rem", "color": "#9ca3af"},
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Association",
                        value="pairs-tab",
                        style=tab_style,
                        selected_style=tab_selected_style,
                        children=[
                            html.Div(
                                style={"border": "1px solid #e5e7eb", "borderTop": "none", "borderRadius": "0 6px 6px 6px", "background": "#fff", "padding": "16px"},
                                children=[
                                    html.Div(id="assoc-pair-content"),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Aide",
                        value="help-tab",
                        style=tab_style,
                        selected_style=tab_selected_style,
                        children=[
                            html.Div(
                                style={"border": "1px solid #e5e7eb", "borderTop": "none", "borderRadius": "0 6px 6px 6px", "background": "#fff", "padding": "24px"},
                                children=[
                                    dcc.Markdown(HELP_MARKDOWN, style={"fontSize": "0.875rem", "color": "#374151"}),
                                ],
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
    _CALLBACKS_REGISTERED = True

    if _DATA_ERROR:
        return

    @callback(
        Output("assoc-pair-content", "children"),
        Input("assoc-tabs", "value"),
        Input("assoc-exact-pair", "data"),
    )
    def update_pair_content(active_tab, exact_pair):
        if active_tab != "pairs-tab":
            return no_update
        if not exact_pair:
            return [html.Span(
                "Cliquez sur une ligne dans l'onglet Réseau pour afficher l'association entre deux variables.",
                style={"fontSize": "0.875rem", "color": "#9ca3af"},
            )]
        return _build_pair_accordion(exact_pair)

    @callback(
        Output("assoc-network-hover", "children"),
        Input("assoc-network-graph", "mouseoverNodeData"),
        Input("assoc-network-graph", "mouseoverEdgeData"),
    )
    def update_hover_info(node_data, edge_data):
        triggered_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        small = {"fontSize": "0.8rem", "color": "#6b7280"}
        if "mouseoverEdgeData" in triggered_prop and edge_data:
            return html.Span(edge_data.get("tooltip", ""), style={"fontSize": "0.875rem", "color": "#111827"})
        if "mouseoverNodeData" in triggered_prop and node_data:
            return html.Div([
                html.Span(node_data.get("id", ""), style={"fontWeight": "600", "fontSize": "0.875rem", "color": "#111827", "marginRight": "8px"}),
                html.Span(node_data.get("description", ""), style={"fontSize": "0.875rem", "color": "#374151"}),
            ])
        return html.Span("Survolez une variable ou une ligne pour afficher les détails.", style=small)

    @callback(
        Output("assoc-tabs", "value"),
        Output("assoc-exact-pair", "data"),
        Input("assoc-network-graph", "tapEdgeData"),
        prevent_initial_call=True,
    )
    def on_graph_click(edge_data):
        if edge_data:
            return "pairs-tab", {"source": edge_data["source"], "target": edge_data["target"]}
        raise PreventUpdate


def init(selected_variables: list[str] | None = None) -> None:
    """Load data and register callbacks. Call this before create_layout().

    Args:
        selected_variables: subset of column names to analyse. Pass None to
            use all columns in the dataset (default behaviour).
            In the global app, pass the variables determined by the user query.
    """
    _load_context(selected_variables=selected_variables)
    register_callbacks()
