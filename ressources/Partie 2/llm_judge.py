from __future__ import annotations

import json
import os

from dotenv import find_dotenv, load_dotenv
from mistralai import Mistral

_CACHE: dict[tuple[str, str], dict] = {}


def _strength_label(edge) -> str:
    """French qualification of association strength for the interpretation step only.

    Thresholds mirror Cohen (1988): small |r|/V = 0.10, medium = 0.30, large = 0.50;
    R²_adj equivalents: small = 0.01, medium = 0.09, large = 0.25.
    """
    t = edge.association_type
    v = edge.association_value
    if t == "Pearson r":
        val = abs(v)
        if val >= 0.50:
            return "association forte"
        elif val >= 0.30:
            return "association modérée"
        return "association faible"
    if t == "Cramér's V":
        if v >= 0.50:
            return "association forte"
        elif v >= 0.30:
            return "association modérée"
        return "association faible"
    # R² adj. (numeric-categorical)
    if v >= 0.25:
        return "association forte"
    elif v >= 0.09:
        return "association modérée"
    return "association faible"


def _format_data_summary(summary: dict, descriptions: dict[str, str] | None = None) -> str:
    """Format a data_summary dict into a concise French natural-language description.

    The output is injected into the LLM prompt exclusively for the interpretation
    step (Phase 2). It must never influence the semantic surprise scores.
    Returns an empty string when the summary is empty or its type is unknown.
    descriptions: optional mapping from raw column name to human-readable label.
    """
    desc = descriptions or {}

    def label(name: str) -> str:
        return desc.get(name, name)

    t = summary.get("type")
    if t == "numeric_numeric":
        r = summary.get("r", 0)
        direction = summary.get("direction", "")
        return f"r = {r:.2f} (corrélation {direction})."
    if t == "numeric_categorical":
        num_var = label(summary.get("numeric_var", ""))
        cat_var = label(summary.get("cat_var", ""))
        highest = summary.get("highest_group", "")
        lowest = summary.get("lowest_group", "")
        diff = summary.get("mean_difference", 0)
        group_means = summary.get("group_means", [])
        groups_str = " ; ".join(
            f"{g['group']} : {g['mean']}" for g in group_means
        )
        return (
            f"Moyennes de {num_var} par groupe ({cat_var}) : {groups_str}. "
            f"Groupe avec la moyenne la plus haute : {highest} ; plus basse : {lowest}. "
            f"Écart : {diff}."
        )
    if t == "categorical_categorical":
        v1 = label(summary.get("var1", ""))
        v2 = label(summary.get("var2", ""))
        top_cells = summary.get("top_cells", [])
        cells_parts = []
        for c in top_cells:
            direction = "sur-représenté" if c["std_residual"] > 0 else "sous-représenté"
            cells_parts.append(
                f"({c['var1_value']}, {c['var2_value']}) : {direction} "
                f"({c['observed']} cas observés, {c['expected']:.1f} attendus si indépendants)"
            )
        cells_str = " ; ".join(cells_parts)
        return f"Combinaisons les plus marquantes ({v1} × {v2}) : {cells_str}."
    return ""


def _get_api_key() -> str:
    load_dotenv(find_dotenv(usecwd=False), override=True)
    key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY_FREE")
    if not key:
        raise EnvironmentError(
            "No Mistral API key found. Set MISTRAL_API_KEY or MISTRAL_API_KEY_FREE in your .env file."
        )
    return key


def score_newsworthiness(
    edges: list,
    descriptions: dict[str, str],
    data_summaries: dict[tuple[str, str], dict] | None = None,
) -> dict[tuple[str, str], dict]:
    """Batch-score edge newsworthiness via Mistral in a single API call.

    Returns a dict mapping (source, target) -> {
        "score": float,                      # moyenne des deux sous-scores
        "documentation_litterature": float,
        "plausibilite_theorique": float,
        "justification_documentation": str,
        "justification_plausibilite": str,
        "interpretation": str,
        "raisonnement": str,
    }.
    data_summaries: optional per-edge data summaries from compute_data_summaries().
    When provided, the interpretation is grounded in the observed data.
    Results are cached in memory to avoid redundant API calls across page reloads.
    """
    uncached = [
        e for e in edges
        if (e.source, e.target) not in _CACHE and (e.target, e.source) not in _CACHE
    ]
    if uncached:
        _call_llm(uncached, descriptions, data_summaries or {})

    result: dict[tuple[str, str], dict] = {}
    for e in edges:
        key = (e.source, e.target) if (e.source, e.target) in _CACHE else (e.target, e.source)
        result[(e.source, e.target)] = _CACHE.get(key, {
            "score": 0.5,
            "documentation_litterature": 0.5,
            "plausibilite_theorique": 0.5,
            "justification_documentation": "",
            "justification_plausibilite": "",
            "interpretation": "",
            "raisonnement": "",
        })
    return result


def _call_llm(
    edges: list,
    descriptions: dict[str, str],
    data_summaries: dict[tuple[str, str], dict] | None = None,
) -> None:
    ds = data_summaries or {}
    pairs = []
    for i, edge in enumerate(edges):
        raw_summary = ds.get((edge.source, edge.target)) or ds.get((edge.target, edge.source))
        formatted = _format_data_summary(raw_summary, descriptions) if raw_summary else None
        pairs.append({
            "id": i,
            "description_1": descriptions.get(edge.source, edge.source),
            "description_2": descriptions.get(edge.target, edge.target),
            "association_type": edge.association_type,
            # The two fields below are for Phase 2 ONLY — must NOT influence semantic scores.
            "force_pour_interpretation_uniquement": _strength_label(edge),
            "donnees_observees_pour_interpretation_uniquement": formatted,
        })

    prompt = f"""Tu évalues des associations statistiques selon deux dimensions de surprise sémantique, et tu proposes des pistes de réflexion journalistiques pour chaque paire.

══════════════════════════════════════════════
PHASE 1 — ÉVALUATION SÉMANTIQUE (étapes 1 à 5)
══════════════════════════════════════════════
RÈGLE ABSOLUE : les étapes 1 à 5 doivent être réalisées en ignorant totalement les champs "force_pour_interpretation_uniquement" et "donnees_observees_pour_interpretation_uniquement". Ces champs ne doivent PAS influencer les scores de documentation ni de plausibilité. Évalue uniquement la SURPRISE SÉMANTIQUE à partir des descriptions des variables.

Pour chaque paire, procède impérativement dans l'ordre suivant :

1. raisonnement
   Avant d'attribuer le moindre score, raisonne en 4 à 6 phrases :
   - 2 à 3 phrases sur la documentation scientifique : que dit (ou ne dit pas) la littérature sur cette association ? Existe-t-il des études établies, des méta-analyses, ou au contraire une quasi-absence de travaux ?
   - 2 à 3 phrases sur la plausibilité théorique : les cadres théoriques, modèles ou régularités empiriques existants permettaient-ils d'anticiper cette association, ou est-elle inattendue au regard de la théorie ?
   Ce raisonnement ancre les scores qui suivent.

2. documentation_litterature (0.0–1.0) — à attribuer après le raisonnement ci-dessus, SANS tenir compte de la force statistique ni des données observées
   Dans quelle mesure cette association est-elle documentée dans la littérature scientifique ?
   0.0 = consensus établi, très bien documentée
   1.0 = rarement ou jamais documentée

3. plausibilite_theorique (0.0–1.0) — à attribuer après le raisonnement ci-dessus, SANS tenir compte de la force statistique ni des données observées
   Les cadres théoriques et empiriques existants permettaient-ils d'anticiper cette association ?
   0.0 = association pleinement attendue au regard des théories et régularités connues
   1.0 = aucun cadre théorique connu ne permettait de l'anticiper, association surprenante

4. justification_documentation
   1 à 2 phrases résumant le score de documentation, en cohérence avec le raisonnement.

5. justification_plausibilite
   1 à 2 phrases résumant le score de plausibilité théorique, sans affirmer de lien causal, en cohérence avec le raisonnement.

══════════════════════════════════════════════
PHASE 2 — PISTE DE RÉFLEXION (étape 6)
══════════════════════════════════════════════
Maintenant seulement, tu peux utiliser les champs "force_pour_interpretation_uniquement" et "donnees_observees_pour_interpretation_uniquement" fournis avec chaque paire.

6. interpretation
   3 à 5 phrases proposant des pistes de réflexion pour un journaliste :
   - Mentionne explicitement la force de l'association indiquée dans "force_pour_interpretation_uniquement" (ex. « cette association modérée… »).
   - Si "donnees_observees_pour_interpretation_uniquement" est non nul, ancre la piste dans les faits observés : cite chaque valeur clé une seule fois, en l'intégrant naturellement à la phrase où tu mentionnes la force (ex. pour un mean plot : « les hommes ont une moyenne de X contre Y chez les femmes, soit un écart de Z » ; pour une corrélation : intègre r à la mention de la force, ex. « cette association modérée (r = X) est positive — ne répète pas r ailleurs »). N'invente pas de chiffres — utilise uniquement ceux fournis.
   - Vocabulaire : le mot « corrélation » est réservé aux paires de type Pearson r (deux variables continues). Pour une paire continue–catégorielle (R² adj.) ou catégorielle–catégorielle (V de Cramér), utilise exclusivement « association » ou « relation » — jamais « corrélation ».
   - Appuie-toi sur les sous-scores de surprise que tu viens d'attribuer pour contextualiser l'intérêt journalistique (ex. « peu documentée dans la littérature, cette association... »).
   - Suggère des hypothèses que l'association pourrait soulever, quelles théories existantes pourraient l'éclairer, quelles questions cela ouvre.
   - Sans tirer de conclusion définitive et sans affirmer de lien causal.

Exemples de calibration (scores sémantiques uniquement) :
- salaire vs. niveau d'éducation : documentation = 0.05, plausibilité = 0.05 (relation classique, pleinement attendue)
- âge vs. statut de retraite : documentation = 0.02, plausibilité = 0.02 (quasi-définitionnelle)
- tabagisme vs. santé pulmonaire : documentation = 0.05, plausibilité = 0.05 (universellement connu)
- statut vaccinal vs. bien-être subjectif : documentation = 0.70, plausibilité = 0.75
- fréquence internet vs. confiance politique : documentation = 0.75, plausibilité = 0.80
- taille vs. satisfaction de vie : documentation = 0.80, plausibilité = 0.90

Paires à évaluer :
{json.dumps(pairs, ensure_ascii=False, indent=2)}

Retourne un objet JSON avec une seule clé "results" contenant un tableau, une entrée par paire.
Le champ "raisonnement" doit apparaître en premier afin que le score soit attribué après le raisonnement :
{{
  "results": [
    {{
      "id": <même id que l'entrée>,
      "raisonnement": "<4-6 phrases de raisonnement préalable>",
      "documentation_litterature": <0.0-1.0>,
      "plausibilite_theorique": <0.0-1.0>,
      "justification_documentation": "<1-2 phrases>",
      "justification_plausibilite": "<1-2 phrases>",
      "interpretation": "<3-5 phrases intégrant la force et les sous-scores>"
    }}
  ]
}}"""

    api_key = _get_api_key()
    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    data = json.loads(raw)
    scored: list[dict] = data.get("results", [])

    for item in scored:
        idx = int(item["id"])
        if idx >= len(edges):
            continue
        edge = edges[idx]
        doc = max(0.0, min(1.0, float(item.get("documentation_litterature", 0.5))))
        plaus = max(0.0, min(1.0, float(item.get("plausibilite_theorique", 0.5))))
        _CACHE[(edge.source, edge.target)] = {
            "score": (doc + plaus) / 2,
            "documentation_litterature": doc,
            "plausibilite_theorique": plaus,
            "justification_documentation": str(item.get("justification_documentation", "")),
            "justification_plausibilite": str(item.get("justification_plausibilite", "")),
            "interpretation": str(item.get("interpretation", "")),
            "raisonnement": str(item.get("raisonnement", "")),
        }
