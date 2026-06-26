#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MÉTHODE 1 - STEP 2 : Génération des embeddings avec E5-Large 
                     VERSION OPTIMALE (description_short + description_long)
                     (Modèle performant multilingue)
====================================================================

SOLUTION : Rendre tous les IDs uniques sans perte de données
En ajoutant un suffixe _1, _2, etc. aux doublons

OPTIMISATION: Utilise les DEUX descriptions pour de meilleurs embeddings

AMÉLIORATION MAJEURE :
- Modèle: intfloat/multilingual-e5-large (1024D)
- Performance: ★★★★★ (au lieu de ★★☆☆☆)
- Matching FR↔EN: Excellent
- Scores attendus: 0.75+ pour bons matchs
 
Note: Plus lent mais BEAUCOUP plus précis !

Auteur: Carnot
Date: Mars 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import Counter
 
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
except ImportError as e:
    print(f"❌ Erreur: {e}")
    print("pip install sentence-transformers chromadb")
    exit(1)
 
 
class EmbeddingBuilderE5:
    """
    Générateur d'embeddings avec E5-Large
    
    Améliorations vs version précédente:
    - Modèle E5-Large (1024D) au lieu de MiniLM (384D)
    - Préfixe 'query:' pour recherches, 'passage:' pour documents
    - Meilleur matching cross-lingue (FR↔EN)
    """
    
    def __init__(self, input_file: Path, output_dir: Path):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("🔧 INITIALISATION (VERSION E5-LARGE)")
        print("="*80)
        
        print("\n📦 Chargement du modèle E5-Large...")
        print("   Modèle: intfloat/multilingual-e5-large")
        print("   ⚠️  Premier chargement: ~2.5 GB à télécharger")
        print("   ⏳ Cela peut prendre 5-10 minutes...")
        
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        print(f"\n   ✅ Modèle chargé")
        print(f"   📊 Dimension: {self.model.get_sentence_embedding_dimension()}D")
        print(f"   🌍 Langues: 100+ (excellent FR↔EN)")
        print(f"   ⚡ Performance: ★★★★★")
        
        self.variables = []
        
    def load_variables(self) -> List[Dict]:
        """Charge + rend les IDs uniques"""
        print(f"\n📊 Chargement de {self.input_file}...")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Fichier introuvable: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_variables = data['variables']
        print(f"   Variables chargées: {len(raw_variables)}")
        
        # RENDRE LES IDs UNIQUES
        print("\n🔧 Traitement des IDs...")
        
        id_counts = Counter(var['variable_id'] for var in raw_variables)
        duplicates = {id: count for id, count in id_counts.items() if count > 1}
        
        if duplicates:
            print(f"   ⚠️  {len(duplicates)} IDs en double détectés")
            total_dups = sum(duplicates.values()) - len(duplicates)
            print(f"   📊 Doublons à renommer: {total_dups}")
        
        id_occurrence = {}
        
        for var in raw_variables:
            original_id = var['variable_id']
            
            if original_id in duplicates:
                if original_id not in id_occurrence:
                    id_occurrence[original_id] = 0
                else:
                    id_occurrence[original_id] += 1
                    var['variable_id'] = f"{original_id}_{id_occurrence[original_id]}"
                    var['original_id'] = original_id
        
        self.variables = raw_variables
        print(f"   ✅ {len(self.variables)} variables avec IDs uniques")
        
        return self.variables
    
    def create_text_for_embedding(self, var: Dict, mode: str = 'passage') -> str:
        """
        Crée le texte à embedder
        
        IMPORTANT E5: Préfixer avec 'passage:' pour documents
        
        Args:
            var: Variable dictionary
            mode: 'passage' pour documents, 'query' pour recherches
        """
        parts = []
        
        # 1. Code et nom
        parts.append(f"{var['code']}: {var['name_en']}")
        
        # 2. Description courte
        if var.get('description_short'):
            parts.append(var['description_short'])
        
        # 3. Description longue (ENRICHIE)
        if var.get('description_long'):
            parts.append(var['description_long'])
        
        # 4. Topic/catégorie
        if var.get('topic'):
            parts.append(var['topic'])
        if var.get('category'):
            parts.append(var['category'])
        
        # 5. Tags
        if var.get('tags'):
            parts.extend(var['tags'])
        
        text = ' | '.join(parts)
        
        # PRÉFIXE E5 (crucial pour performance !)
        if mode == 'passage':
            text = f"passage: {text}"
        elif mode == 'query':
            text = f"query: {text}"
        
        return text
    
    def generate_embeddings(self) -> np.ndarray:
        """Génère les embeddings avec E5-Large"""
        print(f"\n{'='*80}")
        print("🔧 GÉNÉRATION DES EMBEDDINGS (E5-LARGE)")
        print("="*80)
        
        print("\n📝 Préparation des textes (avec préfixe 'passage:')...")
        texts = []
        for var in tqdm(self.variables, desc="Textes"):
            text = self.create_text_for_embedding(var, mode='passage')
            texts.append(text)
        
        print(f"\n   ✅ {len(texts)} textes préparés")
        
        # Statistiques
        lengths = [len(t) for t in texts]
        print(f"   📊 Longueur moyenne: {sum(lengths) / len(lengths):.0f} chars")
        print(f"   📊 Longueur min: {min(lengths)} chars")
        print(f"   📊 Longueur max: {max(lengths)} chars")
        
        print(f"\n💫 Génération des vecteurs E5-Large...")
        print(f"   ⚠️  Plus lent que MiniLM mais BEAUCOUP plus précis")
        print(f"   ⏳ Temps estimé: 3-5 minutes pour 1500 variables")
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=16,  # Plus petit batch pour E5-Large (modèle plus gros)
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"\n   ✅ Shape: {embeddings.shape}")
        print(f"   📊 Dimension: {embeddings.shape[1]}D (vs 384D pour MiniLM)")
        
        # Sauvegarder
        embeddings_file = self.output_dir / 'embeddings.npy'
        np.save(embeddings_file, embeddings)
        print(f"\n💾 {embeddings_file}")
        print(f"   Taille: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   (Plus gros car 1024D au lieu de 384D)")
        
        return embeddings
    
    def create_chroma_collection(
        self, 
        client: chromadb.PersistentClient,
        collection_name: str,
        variables: List[Dict],
        embeddings: np.ndarray,
        description: str
    ):
        """Crée une collection ChromaDB"""
        try:
            client.delete_collection(name=collection_name)
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": description}
        )
        
        ids = [var['variable_id'] for var in variables]
        documents = [self.create_text_for_embedding(var, mode='passage') for var in variables]
        metadatas = [
            {
                'survey': var['survey'],
                'code': var['code'],
                'category': var.get('category', ''),
                'name_en': var['name_en'][:500],
                'original_id': var.get('original_id', var['variable_id'])
            }
            for var in variables
        ]
        
        if len(ids) != len(set(ids)):
            raise ValueError(f"IDs en double dans {collection_name}")
        
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc=f"  {collection_name}"):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist()
            )
        
        print(f"   ✅ {collection.count()} documents indexés")
        
        return collection
    
    def create_all_indexes(self, embeddings: np.ndarray):
        """Crée tous les index ChromaDB"""
        print(f"\n{'='*80}")
        print("🗄️  CRÉATION DES INDEX CHROMADB")
        print("="*80)
        
        chroma_path = str(self.output_dir / 'chroma_db')
        print(f"\nChemin: {chroma_path}")
        
        client = chromadb.PersistentClient(path=chroma_path)
        
        print("\n📊 Index unifié...")
        self.create_chroma_collection(
            client,
            'unified_variables',
            self.variables,
            embeddings,
            f'All {len(self.variables)} variables (E5-Large embeddings)'
        )
        
        surveys = [
            ('EU-SILC', 'eu_silc_variables'),
            ('HFCS', 'hfcs_variables'),
            ('EU-LFS', 'eu_lfs_variables'),
            ('HBS', 'hbs_variables')
        ]
        
        for survey_name, collection_name in surveys:
            print(f"\n📊 Index {survey_name}...")
            
            survey_vars = [v for v in self.variables if v['survey'] == survey_name]
            survey_indices = [i for i, v in enumerate(self.variables) if v['survey'] == survey_name]
            survey_embeddings = embeddings[survey_indices]
            
            if len(survey_vars) == 0:
                continue
            
            self.create_chroma_collection(
                client,
                collection_name,
                survey_vars,
                survey_embeddings,
                f'{survey_name} only (E5-Large)'
            )
        
        print(f"\n{'='*80}")
        print("✅ TOUS LES INDEX CRÉÉS")
        print("="*80)
    
    def save_metadata(self):
        """Sauvegarde métadonnées"""
        metadata = {
            'model': 'intfloat/multilingual-e5-large',
            'model_version': 'v1',
            'dimension': 1024,
            'total_variables': len(self.variables),
            'optimization': 'E5-Large with passage/query prefixes',
            'performance': '★★★★★ (vs ★★☆☆☆ for MiniLM)',
            'note': 'IDs uniques. Embeddings E5-Large pour excellent matching FR↔EN',
            'surveys': {
                survey: len([v for v in self.variables if v['survey'] == survey])
                for survey in ['EU-SILC', 'HFCS', 'EU-LFS', 'HBS']
            }
        }
        
        metadata_file = self.output_dir / 'index_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Métadonnées: {metadata_file}")
 
 
def main():
    input_file = Path('../data/unified/unified_variables.json')
    output_dir = Path('../data/embeddings')
    
    if not input_file.exists():
        input_file = Path('data/unified/unified_variables.json')
        output_dir = Path('data/embeddings')
    
    try:
        builder = EmbeddingBuilderE5(input_file, output_dir)
        builder.load_variables()
        embeddings = builder.generate_embeddings()
        builder.create_all_indexes(embeddings)
        builder.save_metadata()
        
        print(f"\n{'='*80}")
        print("✅ TERMINÉ - VERSION E5-LARGE")
        print("="*80)
        print(f"\n📊 {len(builder.variables)} variables indexées")
        print("🎯 Modèle E5-Large (1024D)")
        print("⚡ Performance attendue: Scores 0.75+ pour bons matchs")
        print("🌍 Excellent matching FR↔EN")
        
        print("\n📈 AMÉLIORATION ATTENDUE:")
        print("   AVANT (MiniLM): 'revenus locatifs' → HY040N absent (Score 0.38)")
        print("   APRÈS (E5-Large): 'revenus locatifs' → HY040N top-3 (Score 0.75+)")
        
        print("\n🚀 Prochaine étape: python step3_rag_engine.py")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
 
 
if __name__ == '__main__':
    main()