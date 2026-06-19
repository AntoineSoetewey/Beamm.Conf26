#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERFACE GRADIO - VERSION SIMPLIFIÉE EXPLICATIONS LLM
=======================================================

✅ Recherche sémantique simple (sans pondération)
✅ Descriptions enrichies depuis les fichiers Excel du step 3
✅ Explications longues et détaillées via LLM + heuristique
✅ Recherche par code INSENSIBLE à la casse
✅ Détection améliorée des questions sur variables spécifiques
✅ Enquêtes supportées : EU-SILC, HFCS, EU-LFS, HBS, IPCAL, DEMOBEL

Auteur: Carnot
Date: Mars 2026
"""

import gradio as gr
import json
import re
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from step3_rag_engine import MultiSurveyRAG
    from step4_llm_wrapper import LLMValidator
    from sentence_transformers import SentenceTransformer
    import chromadb
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    exit(1)

OLLAMA_AVAILABLE = False
try:
    import requests
    response = requests.get('http://localhost:11434/api/tags', timeout=1)
    if response.status_code == 200:
        OLLAMA_AVAILABLE = True
except:
    pass
 
 
# ==============================================================================
# PATTERNS DE DÉTECTION DES CODES DE VARIABLES
# ==============================================================================
# Un code de variable est typiquement :
#   - EU-SILC  : HY040N, PL073, HS021, HH010  → lettre(s) + chiffres + lettre optionnelle
#   - HFCS     : DA1000, DA1120, DI1410i        → 2 lettres + 4 chiffres + lettre optionnelle
#   - EU-LFS   : HWUSUAL, ILOSTAT               → tout en lettres ou lettres+chiffres
#   - HBS      : variés
#
# Pattern générique : 2-8 lettres MAJ + 0-5 chiffres + 0-3 caractères alphanumériques
# On exclut les mots courants français/anglais (que, est, une, les, des, etc.)

# Mots à exclure de la détection (stopwords courants en MAJ)
_STOPWORDS = {
    # Français courants
    'QUE', 'EST', 'UNE', 'LES', 'DES', 'QUI', 'PAR', 'SUR', 'DANS',
    'POUR', 'AVEC', 'SON', 'SES', 'CES', 'MON', 'TON', 'NOS', 'VOS',
    'QUOI', 'QUELS', 'QUELLES', 'QUEL', 'QUELLE',
    # Mots thématiques souvent en début de question (≤ 6 mots → passe 3)
    'NIVEAU', 'HEURES', 'REVENUS', 'VARIABLES', 'CODES', 'IMPACT',
    'DEPENSES', 'DÉPENSES', 'PRECARITE', 'PRÉCARITÉ', 'PATRIMOINE',
    'NOUVEAU', 'NOUVEAUX', 'SEMAINE', 'TRAVAIL', 'EMPLOI', 'EDUCATION',
    'LOGEMENT', 'SANTE', 'SANTÉ', 'REGION', 'WALLONIE', 'BELGIQUE',
    # Anglais courants
    'THE', 'AND', 'FOR', 'ARE', 'NOT', 'HAS', 'HAVE', 'HAD',
    # Termes techniques génériques
    'CODE', 'VAR', 'NOM', 'NAME', 'TYPE', 'ID',
}


def _is_likely_variable_code(token: str) -> bool:
    """
    Heuristique : un token est probablement un code de variable si :
      1. Il n'est pas un mot courant français/anglais
      2. Il contient au moins un chiffre, OU est tout en MAJ et >= 4 chars
         (ex: HWUSUAL, ILOSTAT)
      3. Sa longueur est entre 3 et 12 caractères
    """
    t = token.upper()
    if t in _STOPWORDS:
        return False
    if len(t) < 3 or len(t) > 12:
        return False
    # Contient au moins un chiffre → fort signal
    if re.search(r'\d', t):
        return True
    # Tout en lettres MAJ, >= 5 chars → peut être code EU-LFS (HWUSUAL, etc.)
    if re.match(r'^[A-Z]{5,}$', t):
        return True
    return False


def detect_variable_code(question: str) -> str | None:
    """
    Détecte si la question porte sur une variable spécifique.

    Stratégie en 2 passes :

    PASSE 1 — Patterns contextuels (forte confiance)
        Cherche un code dans des formulations explicites comme :
        "Que signifie HY040N ?", "c'est quoi DA1000", "variable HWUSUAL", etc.
        → Supporte apostrophes droites ET curly quotes (' ' ‛ ‚)
        → Supporte majuscules ET minuscules dans la question

    PASSE 2 — Token isolé (fallback)
        Si la question ressemble à un code seul (ex: "HY040N" ou "hwusual")
        → Retourne le token nettoyé
    """
    # Normaliser les apostrophes curly en apostrophe droite
    q_norm = question.strip()
    q_norm = q_norm.replace('\u2019', "'").replace('\u2018', "'") \
                   .replace('\u201a', "'").replace('\u201b', "'")

    # ── PASSE 1 : patterns contextuels ──────────────────────────────────────

    # Groupe capturant : token alphanumérique 3-12 chars
    CODE = r'([A-Za-z]{2,8}[0-9]{0,5}[A-Za-z0-9]{0,3})'

    contextual_patterns = [
        # "que signifie HY040N"  /  "qu'est-ce que HY040N"
        rf"(?:que\s+signifie|qu['\s]+est.{0,10}que|c['\s]+est\s+quoi|veut\s+dire)\s+{CODE}",
        # "HY040N signifie quoi"  /  "HY040N c'est quoi"
        rf"{CODE}\s+(?:signifie|veut\s+dire|c['\s]+est\s+quoi)",
        # "expliquer / définir / détailler HY040N"
        rf"(?:expliqu|défini|détaill|décrire|décris|describe|explain)\w*\s+(?:la\s+variable\s+|le\s+code\s+)?{CODE}",
        # "variable HY040N"  /  "code HY040N"
        rf"(?:variable|code|var)\s+{CODE}",
        # "HY040N ?"  (code suivi directement d'un point d'interrogation)
        rf"{CODE}\s*\?",
        # "what is HY040N"  /  "what does HY040N mean"
        rf"(?:what\s+is|what\s+does|meaning\s+of|definition\s+of)\s+{CODE}",
    ]

    for pattern in contextual_patterns:
        match = re.search(pattern, q_norm, re.IGNORECASE)
        if match:
            detected = match.group(1)
            if _is_likely_variable_code(detected):
                print(f"✅ [detect_variable_code] Pattern contextuel → '{detected}'")
                return detected.upper()

    # ── PASSE 2 : question = code seul ──────────────────────────────────────
    # Si la question (nettoyée de la ponctuation) est un unique token
    token_only = re.sub(r'[^A-Za-z0-9]', '', q_norm)
    if token_only and re.match(r'^[A-Za-z]{2,8}[0-9]{0,5}[A-Za-z0-9]{0,3}$', token_only):
        if _is_likely_variable_code(token_only):
            print(f"✅ [detect_variable_code] Token seul → '{token_only}'")
            return token_only.upper()

    # ── PASSE 3 : scan de tous les tokens (fallback large) ───────────────────
    # Cherche n'importe quel token ressemblant à un code parmi les tokens de la question.
    # On ne l'active que si la question est COURTE (≤ 6 mots) pour éviter les faux positifs.
    words = q_norm.split()
    if len(words) <= 6:
        for word in words:
            clean = re.sub(r'[^A-Za-z0-9]', '', word)
            if _is_likely_variable_code(clean):
                print(f"✅ [detect_variable_code] Scan tokens → '{clean}'")
                return clean.upper()

    print(f"⚠️  [detect_variable_code] Aucun code détecté dans: '{question}'")
    return None


# ==============================================================================
# APPLICATION
# ==============================================================================

class AssociationExplorer:
    
    def __init__(self):
        print("="*80)
        print("🚀 ASSOCIATIONEXPLORER - VERSION EXPLICATIONS LLM")
        print("="*80)
        
        self.chroma_path = Path('../data/embeddings/chroma_db')
        self.variables_path = Path('../data/unified/unified_variables.json')
        
        if not self.chroma_path.exists():
            self.chroma_path = Path('data/embeddings/chroma_db')
            self.variables_path = Path('data/unified/unified_variables.json')
        
        print("\n🔧 Initialisation du système RAG...")
        self.rag = MultiSurveyRAG(self.chroma_path, self.variables_path)
    def validate_with_llm(self, question: str, variable_code: str, variable_name: str, variable_desc: str) -> tuple:
        """Valide avec Llama 3.2 — utilise la description enrichie du step 3 si disponible."""
        if not OLLAMA_AVAILABLE:
            return "unknown", ""

        # Préférer la description longue enrichie du step 3
        enriched_desc = self.llm_validator.explain_from_excel(variable_code, prefer='llm')
        if enriched_desc and 'non trouvée' not in enriched_desc:
            desc_to_use = enriched_desc[:300]
        else:
            desc_to_use = variable_desc[:200]

        try:
            prompt = f"""Tu es un expert en variables statistiques.

Question de l'utilisateur : "{question}"

Variable à analyser :
- Code : {variable_code}
- Nom : {variable_name}
- Description : {desc_to_use}

Ta tâche :
1. Détermine si cette variable répond à la question (exact/partial/irrelevant)
2. Explique EN UNE PHRASE COURTE pourquoi

Format de réponse OBLIGATOIRE :
STATUS: [exact/partial/irrelevant]
EXPLICATION: [ta phrase courte]

Sois précis et concis."""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model'  : 'llama3.2',
                    'prompt' : prompt,
                    'stream' : False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 60,
                        'num_ctx'    : 512,
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()['response'].strip()
                status      = "unknown"
                explanation = ""

                for line in result.split('\n'):
                    if 'STATUS:' in line.upper():
                        if 'exact'      in line.lower(): status = 'exact'
                        elif 'partial'  in line.lower(): status = 'partial'
                        elif 'irrelevant' in line.lower(): status = 'irrelevant'
                    elif 'EXPLICATION:' in line.upper():
                        explanation = line.split(':', 1)[1].strip()

                if not explanation:
                    lines = [l.strip() for l in result.split('\n') if l.strip()]
                    if len(lines) > 1:
                        explanation = lines[1][:150]

                return status, explanation

        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout {variable_code}")
            return "unknown", ""
        except Exception as e:
            print(f"   ❌ Erreur {variable_code}: {e}")
            return "unknown", ""

        return "unknown", ""
    
    def format_result_card(
        self, 
        rank: int,
        code: str, 
        survey: str, 
        name: str, 
        description: str,
        score: float,
        category: str,
        llm_status: str = None,
        llm_explanation: str = None
    ) -> str:
        """Format carte SIMPLE avec explication textuelle LLM"""
        
        colors = {
            'EU-SILC' : '#4A90E2',
            'HFCS'    : '#9B59B6',
            'EU-LFS'  : '#27AE60',
            'HBS'     : '#E67E22',
            'IPCAL'   : '#C0392B',
            'DEMOBEL' : '#16A085',
        }
        badge_color = colors.get(survey, '#95A5A6')

        # Préférer la description enrichie du step 3 si disponible
        enriched = self.llm_validator.explain_from_excel(code, prefer='llm') \
                   if hasattr(self, 'llm_validator') else ''
        if enriched and 'non trouvée' not in enriched and len(enriched) > 50:
            desc_to_show = enriched
        else:
            desc_to_show = description
        desc_short = desc_to_show[:200] + "..." if len(desc_to_show) > 200 else desc_to_show
        
        llm_text = ""
        if llm_status and llm_status != 'unknown':
            badge_config = {
                'exact': {
                    'bg': '#10B981',
                    'label': '✓ Pertinent',
                    'icon': '✅'
                },
                'partial': {
                    'bg': '#F39C12',
                    'label': '⚠ Partiel',
                    'icon': '⚠️'
                },
                'irrelevant': {
                    'bg': '#E74C3C',
                    'label': '✗ Non pertinent',
                    'icon': '❌'
                }
            }.get(llm_status, {'bg': '#95A5A6', 'label': '? Inconnu', 'icon': '❓'})
            
            badge = f"""
            <div style="
                display: inline-block;
                background: {badge_config['bg']};
                color: white;
                padding: 5px 13px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 600;
                margin-top: 10px;
            ">{badge_config['label']}</div>
            """
            
            explanation = ""
            if llm_explanation:
                explanation = f"""
                <p style="
                    color: #5A5A5A;
                    font-size: 13px;
                    line-height: 1.5;
                    margin: 8px 0 0 0;
                    padding: 10px;
                    background: #F8F9FA;
                    border-radius: 4px;
                ">
                    <strong>{badge_config['icon']} Explication:</strong> {llm_explanation}
                </p>
                """
            
            llm_text = badge + explanation
        
        return f"""
        <div style="
            border-left: 4px solid {badge_color};
            padding: 16px 18px;
            margin: 12px 0;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="
                        background: {badge_color};
                        color: white;
                        padding: 3px 10px;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: 600;
                        margin-right: 8px;
                    ">{survey}</span>
                    <span style="color: #7F8C8D; font-size: 12px;">{category}</span>
                </div>
                <span style="
                    background: #F0F0F0;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                    color: #2C3E50;
                ">Score: {score:.3f}</span>
            </div>
            
            <h3 style="
                color: #2C3E50;
                font-size: 16px;
                font-weight: 700;
                margin: 8px 0;
            ">{rank}. {code}</h3>
            
            <p style="
                color: #34495E;
                font-size: 14px;
                font-weight: 600;
                margin: 6px 0;
            ">{name}</p>
            
            <p style="
                color: #7F8C8D;
                font-size: 13px;
                line-height: 1.5;
                margin: 6px 0 0 0;
            ">{desc_short}</p>
            
            {llm_text}
        </div>
        """
    
    def search_by_question(
        self, 
        question: str, 
        top_k: int, 
        use_llm: bool,
        filters: List[str]
    ) -> str:
        """Recherche par question avec explications LLM"""
        
        if not question.strip():
            return '<p style="color: #E74C3C; padding: 20px; text-align: center;">⚠️ Veuillez saisir une question</p>'
        
        # Détecter si la question porte sur une variable spécifique
        detected_code = detect_variable_code(question)
        if detected_code:
            print(f"   🔍 Détection: Question sur variable '{detected_code}'")
            # Vérifier que le code existe vraiment dans l'index
            if detected_code in self.code_index_case_insensitive:
                return self.search_by_code_with_explanation(detected_code, use_llm)
            else:
                print(f"   ⚠️  Code '{detected_code}' non trouvé dans l'index → recherche sémantique")
        
        results = self.rag.search_by_question(question, top_k=top_k)
        
        if filters:
            results = [r for r in results if r['metadata']['survey'] in filters]
        
        if not results:
            return '<p style="color: #E74C3C; padding: 20px; text-align: center;">❌ Aucun résultat trouvé</p>'
        
        html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            <h2 style="margin: 0 0 6px 0; font-size: 20px; font-weight: 700;">
                🔍 Résultats pour: "{question}"
            </h2>
            <p style="margin: 0; font-size: 14px; opacity: 0.9;">
                {len(results)} variable{'s' if len(results) > 1 else ''} trouvée{'s' if len(results) > 1 else ''}
            </p>
        </div>
        """
        
        for i, res in enumerate(results, 1):
            meta = res['metadata']
            code = meta['code']
            
            if code in self.rag.variables_by_code:
                var  = self.rag.variables_by_code[code]
                name = var.get('name_en', 'N/A')
                # Préférer la description enrichie du step 3
                enriched = self.llm_validator.explain_from_excel(code, prefer='llm') \
                           if hasattr(self, 'llm_validator') else ''
                desc = (enriched if enriched and 'non trouvée' not in enriched and len(enriched) > 50
                        else var.get('description_long', var.get('description_short', 'N/A')))
                cat  = var.get('category', 'N/A')
            else:
                name = meta.get('name_en', 'N/A')
                desc = 'N/A'
                cat  = meta.get('category', 'N/A')
            
            llm_status = None
            llm_explanation = None
            
            if use_llm and OLLAMA_AVAILABLE:
                print(f"   Validation LLM {i}/{len(results)}: {code}...")
                llm_status, llm_explanation = self.validate_with_llm(
                    question, 
                    code, 
                    name,
                    desc
                )
            
            html += self.format_result_card(
                i, code, meta['survey'], name, desc, res['score'], cat, 
                llm_status, llm_explanation
            )
        
        avg = sum(r['score'] for r in results) / len(results)
        surveys = {}
        for r in results:
            s = r['metadata']['survey']
            surveys[s] = surveys.get(s, 0) + 1
        
        html += f"""
        <div style="
            background: #F8F9FA;
            padding: 16px;
            border-radius: 6px;
            margin-top: 20px;
            border: 1px solid #E0E0E0;
        ">
            <h4 style="margin: 0 0 8px 0; color: #2C3E50; font-size: 14px; font-weight: 700;">📊 Statistiques</h4>
            <div style="color: #5A5A5A; font-size: 13px; line-height: 1.6;">
                <strong>Score moyen:</strong> {avg:.3f}<br>
                <strong>Répartition:</strong> {' • '.join([f'{k}: {v}' for k, v in surveys.items()])}
            </div>
        </div>
        """
        
        return html
    
    def search_by_code_with_explanation(self, code: str, use_llm: bool = True) -> str:
        """Recherche par code avec explication vulgarisée LLM"""
        
        code_input = code.strip()
        code_upper = code_input.upper()
        
        real_code = self.code_index_case_insensitive.get(code_upper)
        
        if not real_code:
            return f'''<p style="color: #E74C3C; padding: 20px; text-align: center;">
                ❌ Variable "{code_input}" non trouvée<br>
                <span style="font-size: 12px; color: #95A5A6;">
                Essayez: HY040N, PL073, DA1000, DI1410i, etc.
                </span>
            </p>'''
        
        result = self.rag.search_by_code(real_code)
        
        if not result['found']:
            return f'<p style="color: #E74C3C; padding: 20px; text-align: center;">❌ Variable "{code_input}" non trouvée</p>'
        
        colors = {
            'EU-SILC' : '#4A90E2',
            'HFCS'    : '#9B59B6',
            'EU-LFS'  : '#27AE60',
            'HBS'     : '#E67E22',
            'IPCAL'   : '#C0392B',
            'DEMOBEL' : '#16A085',
        }

        # Récupérer l'explication enrichie depuis le cache step 3
        llm_explanation = ""
        if use_llm:
            # Priorité 1 : cache Excel step 3
            cached = self.llm_validator.explain_from_excel(real_code, prefer='llm')
            if cached and 'non trouvée' not in cached and len(cached) > 50:
                llm_explanation = cached
                print(f"   📂 Description enrichie depuis cache Excel")
            elif OLLAMA_AVAILABLE:
                # Priorité 2 : génération à la volée
                print(f"   🤖 Génération explication vulgarisée pour {real_code}...")
                try:
                    prompt = (
                        f"Explique de manière longue et détaillée (minimum 6 phrases) "
                        f"pour un non-expert la variable statistique suivante. "
                        f"Donne un exemple concret chiffré si possible.\n\n"
                        f"{result['code']} ({result['survey']}): {result['name']}\n"
                        f"{result['description'][:300]}"
                    )
                    resp = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model'  : 'llama3.2',
                            'prompt' : prompt,
                            'stream' : False,
                            'options': {'temperature': 0.2, 'num_predict': 400, 'num_ctx': 1024}
                        },
                        timeout=60
                    )
                    if resp.status_code == 200:
                        llm_explanation = resp.json()['response'].strip()
                except Exception as e:
                    print(f"   ❌ Erreur LLM: {e}")
        
        html = f"""
        <div style="
            border-left: 4px solid {colors.get(result['survey'], '#95A5A6')};
            padding: 20px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        ">
            <div style="margin-bottom: 16px;">
                <span style="
                    background: {colors.get(result['survey'], '#95A5A6')};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                ">{result['survey']}</span>
                <span style="
                    color: #7F8C8D;
                    font-size: 12px;
                    margin-left: 8px;
                ">{result['category']}</span>
            </div>
            
            <h2 style="color: #2C3E50; font-size: 22px; margin: 12px 0; font-weight: 700;">
                {result['code']}
            </h2>
            
            <h3 style="color: #34495E; font-size: 16px; font-weight: 600; margin: 12px 0; line-height: 1.4;">
                {result['name']}
            </h3>
        """
        
        if llm_explanation:
            html += f"""
            <div style="
                background: #E8F5E9;
                padding: 16px;
                border-radius: 6px;
                margin: 16px 0;
                border-left: 4px solid #4CAF50;
            ">
                <h4 style="
                    color: #2E7D32;
                    font-size: 14px;
                    font-weight: 700;
                    margin: 0 0 8px 0;
                ">💡 Explication simple</h4>
                <p style="
                    color: #1B5E20;
                    font-size: 14px;
                    line-height: 1.6;
                    margin: 0;
                ">{llm_explanation}</p>
            </div>
            """
        
        html += f"""
            <div style="margin-top: 16px;">
                <h4 style="
                    color: #7F8C8D;
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    margin: 0 0 8px 0;
                ">Description technique</h4>
                <p style="color: #7F8C8D; font-size: 14px; line-height: 1.6; margin: 0;">
                    {result['description']}
                </p>
            </div>
        </div>
        """
        
        return html
    
    def search_by_code(self, code: str, use_llm: bool = True) -> str:
        """
        Mode 3 — Lookup direct par code avec explication vulgarisée LLM.
        Hiérarchie d'affichage :
          1. Code + Nom (header)
          2. 💡 Explication simple (fond vert, LLM si disponible)
          3. Description technique (gris)
        """
        return self.search_by_code_with_explanation(code, use_llm=use_llm)


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_interface():
    """Crée l'interface finale"""
    
    app = AssociationExplorer()
    
    css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        max-width: 1200px;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 36px;
        font-weight: 800;
    }
    
    .tab-nav button {
        font-size: 14px;
        font-weight: 600;
    }
    """
    
    with gr.Blocks(title="AssociationExplorer", css=css) as interface:
        
        gr.Markdown("""
        <div style="text-align: center; padding: 30px 20px 20px 20px;">
            <h1>🔍 AssociationExplorer</h1>
            <p style="font-size: 15px; color: #7F8C8D; margin: 10px 0;">
                Système intelligent de recommandation de variables
            </p>
            <p style="font-size: 13px; color: #95A5A6;">
                <strong>EU-SILC · HFCS · EU-LFS · HBS · IPCAL · DEMOBEL</strong>
            </p>
            <p style="
                display: inline-block;
                background: #F0F0F0;
                padding: 6px 14px;
                border-radius: 16px;
                font-size: 11px;
                color: #5A5A5A;
                margin-top: 8px;
                font-weight: 500;
            ">
                Méthode 1: SentenceTransformers (E5-Large) + ChromaDB + Llama 3.2
            </p>
        </div>
        """)
        
        with gr.Tabs():
            
            with gr.Tab("📝 Recherche par Question"):
                
                with gr.Row():
                    with gr.Column(scale=2):
                        q_input = gr.Textbox(
                            label="💬 Posez votre question",
                            placeholder="Ex: Quelles variables concernent les revenus locatifs ? ou : Que signifie HY040N ?",
                            lines=2
                        )
                        
                        with gr.Row():
                            q_topk = gr.Slider(
                                3, 20, value=10, step=1,
                                label="📊 Nombre de résultats"
                            )
                            q_llm = gr.Checkbox(
                                label="🤖 Validation LLM (Llama 3.2)",
                                value=OLLAMA_AVAILABLE,
                                interactive=OLLAMA_AVAILABLE,
                                info="Explique la pertinence de chaque variable" if OLLAMA_AVAILABLE else "Ollama non disponible"
                            )
                        
                        q_filters = gr.CheckboxGroup(
                            choices=["EU-SILC", "HFCS", "EU-LFS", "HBS", "IPCAL", "DEMOBEL"],
                            label="🗂️ Filtrer par enquête (optionnel)",
                            value=[]
                        )
                        
                        q_btn = gr.Button("🔍 Rechercher", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💡 Exemples\n*Cliquez pour tester*")
                        
                        examples = [
                            "Quelles variables concernent les revenus locatifs?",
                            "Variables sur le patrimoine immobilier",
                            "Heures travaillées par semaine",
                            "Précarité énergétique en Wallonie",
                            "Impact du COVID-19 sur les revenus",
                            "Niveau d'éducation",
                            "Dépenses alimentaires",
                            "Que signifie HY040N ?",
                            "C'est quoi HWUSUAL ?",
                            "Expliquer DA1000",
                            # ── Nouvelles enquêtes ──────────────────────────
                            "Nouveaux codes IPCAL 2022",
                            "Que signifie A0270 ?",
                            "Variables revenus cadastraux Demobel",
                            "C'est quoi abo_h_fm ?",
                        ]
                        
                        for ex in examples:
                            gr.Button(ex, size="sm").click(
                                fn=lambda x=ex: x,
                                outputs=q_input
                            )
                
                q_output = gr.HTML()
                
                q_btn.click(
                    fn=app.search_by_question,
                    inputs=[q_input, q_topk, q_llm, q_filters],
                    outputs=q_output
                )
            
            with gr.Tab("🔍 Recherche par Code"):

                gr.Markdown("""
                ### 🔎 Mode 3 — Lookup direct par code
                Entrez un code de variable pour obtenir sa **fiche complète** :
                explication vulgarisée + description technique.

                *(Majuscules/minuscules acceptées : `DI1410i` = `di1410i`)*
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        c_input = gr.Textbox(
                            label="Code de la variable",
                            placeholder="Ex: HY040N, DI1410i, da1000, HWUSUAL",
                            lines=1
                        )

                        c_llm = gr.Checkbox(
                            label="🤖 Explication vulgarisée (Llama 3.2)",
                            value=OLLAMA_AVAILABLE,
                            interactive=OLLAMA_AVAILABLE,
                            info="Génère une explication simple + exemple concret" if OLLAMA_AVAILABLE else "Ollama non disponible"
                        )

                        c_btn = gr.Button("🔍 Afficher la fiche", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 💡 Exemples\n*Cliquez pour tester*")

                        code_examples = [
                            "HY040N",
                            "PL073",
                            "DA1000",
                            "HWUSUAL",
                            "DI1410i",
                            "HS021",
                            "DA1120",
                            # ── Nouvelles enquêtes ──────────────────────────
                            "A0270",
                            "B1062",
                            "abo_h_fm",
                            "yhoooir_hyg_sm",
                        ]
                        for ex in code_examples:
                            gr.Button(ex, size="sm").click(
                                fn=lambda x=ex: x,
                                outputs=c_input
                            )

                c_output = gr.HTML()

                c_btn.click(
                    fn=app.search_by_code,
                    inputs=[c_input, c_llm],
                    outputs=c_output
                )
        
        gr.Markdown("""
        <div style="
            text-align: center;
            padding: 24px 20px;
            color: #95A5A6;
            font-size: 11px;
            border-top: 1px solid #E0E0E0;
            margin-top: 30px;
        ">
            <p style="margin: 3px 0;">AssociationExplorer | Méthode 1: SentenceTransformers</p>
            <p style="margin: 3px 0;">Modèle: intfloat/multilingual-e5-large (1024D) | Performance: ★★★★★</p>
        </div>
        """)
    
    return interface
 
 
if __name__ == "__main__":
    print("🚀 Lancement interface avec explications LLM...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7895,
        share=False
    )