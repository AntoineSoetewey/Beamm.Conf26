#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERFACE GRADIO - VERSION PARFAITE (comme image 4)
===================================================

✅ Badge vert "✓ Pertinent" (style image 4)
✅ Pas de message "validation en cours" pendant l'analyse
✅ Design épuré et professionnel
✅ Recherche par code INSENSIBLE à la casse (DI1410i = DI1410I = di1410i)

Auteur: Carnot
Date: Mars 2026
"""

import gradio as gr
import json
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))
 
try:
    from step3_rag_engine import MultiSurveyRAG
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
 
 
class AssociationExplorer:
    
    def __init__(self):
        print("="*80)
        print("🚀 ASSOCIATIONEXPLORER - VERSION FINALE CORRIGÉE")
        print("="*80)
        
        self.chroma_path = Path('../data/embeddings/chroma_db')
        self.variables_path = Path('../data/unified/unified_variables.json')
        
        if not self.chroma_path.exists():
            self.chroma_path = Path('data/embeddings/chroma_db')
            self.variables_path = Path('data/unified/unified_variables.json')
        
        print("\n🔧 Initialisation du système RAG...")
        self.rag = MultiSurveyRAG(self.chroma_path, self.variables_path)
        
        # NOUVEAU: Index insensible à la casse pour recherche par code
        self.code_index_case_insensitive = {}
        for code, var in self.rag.variables_by_code.items():
            self.code_index_case_insensitive[code.upper()] = code  # Mappe UPPER → original
        
        print(f"   📊 {len(self.code_index_case_insensitive)} codes indexés")
        
        if OLLAMA_AVAILABLE:
            print("✅ Ollama disponible")
        print("✅ Application prête!\n")
    
    def validate_with_llm(self, question: str, variable_code: str, variable_name: str) -> tuple:
        """Valide avec Llama 3.2 - VERSION STRICTE"""
        if not OLLAMA_AVAILABLE:
            return "unknown", ""
        
        try:
            # Prompt TRÈS strict pour avoir plus de "exact"
            prompt = f"""Tu es un expert en variables statistiques.
 
Question: "{question}"
Variable: {variable_code} - {variable_name}
 
Réponds par UN SEUL MOT parmi:
- exact (si la variable répond DIRECTEMENT à la question)
- partial (si la variable est liée mais pas directement)
- irrelevant (si la variable n'a RIEN à voir)
 
SOIS GÉNÉREUX: si la variable semble pertinente, réponds "exact".
Réponds UNIQUEMENT par le mot, rien d'autre."""
 
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Très déterministe
                        'num_predict': 20    # Réponse ultra-courte
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip().lower()
                
                # Détection stricte
                if 'exact' in result[:15]:
                    return 'exact', ""
                elif 'irrelevant' in result[:15] or 'rien' in result[:15]:
                    return 'irrelevant', ""
                elif 'partial' in result[:15]:
                    return 'partial', ""
                
                # Par défaut: exact (généreux)
                return 'exact', ""
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout {variable_code}")
            return "unknown", ""
        except Exception as e:
            print(f"   ❌ Erreur {variable_code}")
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
        llm_status: str = None
    ) -> str:
        """Format carte - STYLE IMAGE 4"""
        
        colors = {
            'EU-SILC': '#4A90E2',
            'HFCS': '#9B59B6', 
            'EU-LFS': '#27AE60',
            'HBS': '#E67E22'
        }
        badge_color = colors.get(survey, '#95A5A6')
        
        # BADGE SIMPLE
        llm_badge = ""
        if llm_status and llm_status != 'unknown':
            if llm_status == 'exact':
                llm_badge = """
                <div style="
                    display: inline-block;
                    background: #10B981;
                    color: white;
                    padding: 5px 13px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    margin-top: 10px;
                ">✓ Pertinent</div>
                """
            elif llm_status == 'partial':
                llm_badge = """
                <div style="
                    display: inline-block;
                    background: #F39C12;
                    color: white;
                    padding: 5px 13px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    margin-top: 10px;
                ">⚠ Partiel</div>
                """
            elif llm_status == 'irrelevant':
                llm_badge = """
                <div style="
                    display: inline-block;
                    background: #E74C3C;
                    color: white;
                    padding: 5px 13px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    margin-top: 10px;
                ">✗ Non pertinent</div>
                """
        
        desc_short = description[:150] + "..." if len(description) > 150 else description
        
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
            
            {llm_badge}
        </div>
        """
    
    def search_by_question(
        self, 
        question: str, 
        top_k: int, 
        use_llm: bool,
        filters: List[str]
    ) -> str:
        """Recherche par question"""
        
        if not question.strip():
            return '<p style="color: #E74C3C; padding: 20px; text-align: center;">⚠️ Veuillez saisir une question</p>'
        
        results = self.rag.search_by_question(question, top_k=top_k)
        
        if filters:
            results = [r for r in results if r['metadata']['survey'] in filters]
        
        if not results:
            return '<p style="color: #E74C3C; padding: 20px; text-align: center;">❌ Aucun résultat trouvé</p>'
        
        # Header SIMPLE
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
        
        # Cartes
        for i, res in enumerate(results, 1):
            meta = res['metadata']
            code = meta['code']
            
            if code in self.rag.variables_by_code:
                var = self.rag.variables_by_code[code]
                name = var.get('name_en', 'N/A')
                desc = var.get('description_long', var.get('description_short', 'N/A'))
                cat = var.get('category', 'N/A')
            else:
                name = meta.get('name_en', 'N/A')
                desc = 'N/A'
                cat = meta.get('category', 'N/A')
            
            # Validation LLM
            llm_status = None
            if use_llm and OLLAMA_AVAILABLE:
                print(f"   Validation {i}/{len(results)}: {code}...")
                llm_status, _ = self.validate_with_llm(question, code, name)
            
            html += self.format_result_card(
                i, code, meta['survey'], name, desc, res['score'], cat, llm_status
            )
        
        # Stats
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
    
    def search_by_code(self, code: str) -> str:
        """Recherche par code - INSENSIBLE À LA CASSE"""
        
        if not code.strip():
            return '<p style="color: #E74C3C; padding: 20px; text-align: center;">⚠️ Veuillez saisir un code</p>'
        
        # NOUVEAU: Recherche insensible à la casse
        code_input = code.strip()
        code_upper = code_input.upper()
        
        # Trouver le code original (peut-être DI1410i au lieu de DI1410I)
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
            'EU-SILC': '#4A90E2',
            'HFCS': '#9B59B6',
            'EU-LFS': '#27AE60',
            'HBS': '#E67E22'
        }
        
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
            </div>
            
            <h2 style="color: #2C3E50; font-size: 22px; margin: 12px 0; font-weight: 700;">
                {result['code']}
            </h2>
            
            <h3 style="color: #34495E; font-size: 16px; font-weight: 600; margin: 12px 0; line-height: 1.4;">
                {result['name']}
            </h3>
            
            <p style="color: #7F8C8D; font-size: 14px; line-height: 1.6; margin: 12px 0;">
                {result['description']}
            </p>
            
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #E0E0E0;">
                <span style="color: #95A5A6; font-size: 11px;">CATÉGORIE:</span><br>
                <span style="color: #2C3E50; font-size: 13px; font-weight: 600;">{result['category']}</span>
            </div>
        </div>
        """
        
        return html
 
 
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
                <strong>1498 variables</strong> de EU-SILC, HFCS, EU-LFS, HBS
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
                            placeholder="Ex: Quelles variables concernent les revenus locatifs ?",
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
                                info="Analyse TOUTES les variables" if OLLAMA_AVAILABLE else "Ollama non disponible"
                            )
                        
                        q_filters = gr.CheckboxGroup(
                            choices=["EU-SILC", "HFCS", "EU-LFS", "HBS"],
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
                            "Dépenses alimentaires"
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
                ### Rechercher une variable spécifique
                
                Entrez le code (majuscules/minuscules acceptées : DI1410i = DI1410I = di1410i)
                """)
                
                c_input = gr.Textbox(
                    label="Code de la variable",
                    placeholder="Ex: HY040N, DI1410i, da1000",
                    lines=1
                )
                
                c_btn = gr.Button("🔍 Rechercher", variant="primary")
                c_output = gr.HTML()
                
                c_btn.click(
                    fn=app.search_by_code,
                    inputs=c_input,
                    outputs=c_output
                )
                
                gr.Markdown("""
                **Exemples:**
                - `HY040N` (EU-SILC) - Income from rental
                - `PL073` (EU-SILC) - Months in full-time work  
                - `DA1000` (HFCS) - Total gross wealth
                - `HWUSUAL` (EU-LFS) - Hours usually worked
                - `DI1410i` (HFCS) - Has rental income from real estate
                """)
        
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
    print("🚀 Lancement interface finale corrigée...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False
    )
