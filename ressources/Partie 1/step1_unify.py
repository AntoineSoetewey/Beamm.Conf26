#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 - UNIFICATION FINALE des 4 enquêtes (EU-SILC, HFCS, EU-LFS, HBS)
=================================================

STRATÉGIE DES DESCRIPTIONS :
1. HFCS      : short=variable_name, long=survey_definition+technical_definition
2. HBS       : short=short_description, long=long_description
3. EU-SILC   : short=variable_name, long=description
4. EU-LFS    : short=short_description, long=long_description


STRATÉGIE D'ENRICHISSEMENT :
- EU-SILC : Déjà riche (description = 2328 chars)
- HFCS    : Enrichir avec TOUS les champs
- EU-LFS  : Enrichir avec section + category + type + remarks + codes
- HBS     : Enrichir avec file_type + category + format + values + notes

Auteur: Carnot
Date: Mars 2026
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
 
 
class SurveyUnifier:
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unified_variables = []
        
    def load_survey(self, filename: str, survey_name: str) -> pd.DataFrame:
        print(f"\n📊 Chargement de {survey_name}...")
        filepath = self.input_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier introuvable: {filepath}")
        
        df = pd.read_excel(filepath, sheet_name=0)
        print(f"   ✅ {len(df)} variables trouvées")
        
        return df
    
    def _safe_str(self, value) -> str:
        """Convertit en string en gérant les NaN"""
        if pd.isna(value):
            return ''
        return str(value).strip()
    
    def harmonize_silc(self, df: pd.DataFrame) -> List[Dict]:
        """
        EU-SILC : Déjà riche, pas besoin d'enrichissement
        - short: variable_name
        - long: description
        """
        print("   🔧 Harmonisation EU-SILC...")
        
        unified = []
        for _, row in df.iterrows():
            desc_short = self._safe_str(row.get('variable_name', ''))
            desc_long = self._safe_str(row.get('description', ''))
            
            var = {
                'variable_id': f"SILC_{self._safe_str(row['variable_code'])}",
                'survey': 'EU-SILC',
                'code': self._safe_str(row['variable_code']),
                'name_en': desc_short,
                'name_fr': desc_short,
                'category': self._safe_str(row.get('variable_category', '')),
                'subcategory': self._safe_str(row.get('file_type', '')),
                'topic': self._safe_str(row.get('topic', '')),
                'unit': self._safe_str(row.get('unit', '')),
                'reference_period': self._safe_str(row.get('reference_period', '')),
                'format': self._safe_str(row.get('variable_type', '')),
                'description_short': desc_short,
                'description_long': desc_long,
                'values_format': self._safe_str(row.get('values_format', '')),
                'tags': [],
                'keywords_positive': [],
                'keywords_negative': [],
                'related_vars': {},
                'metadata': {}
            }
            unified.append(var)
        
        avg_short = sum(len(v['description_short']) for v in unified) / len(unified)
        avg_long = sum(len(v['description_long']) for v in unified) / len(unified)
        print(f"      Short moyen: {avg_short:.0f} chars")
        print(f"      Long moyen: {avg_long:.0f} chars")
        
        return unified
    
    def harmonize_hfcs(self, df: pd.DataFrame) -> List[Dict]:
        """
        HFCS ENRICHI : Combine TOUS les champs
        """
        print("   🔧 Harmonisation HFCS (ENRICHIE)...")
        
        unified = []
        for _, row in df.iterrows():
            desc_short = self._safe_str(row.get('variable_name', ''))
            
            # ENRICHISSEMENT : Tous les champs
            parts = []
            
            if desc_short:
                parts.append(f"Variable: {desc_short}")
            
            question = self._safe_str(row.get('question_text', ''))
            if question:
                parts.append(f"Question: {question}")
            
            survey_def = self._safe_str(row.get('survey_definition', ''))
            if survey_def:
                parts.append(f"Survey definition: {survey_def}")
            
            tech_def = self._safe_str(row.get('technical_definition', ''))
            if tech_def:
                parts.append(f"Technical definition: {tech_def}")
            
            notes = self._safe_str(row.get('notes', ''))
            if notes:
                parts.append(f"Notes: {notes}")
            
            coding = self._safe_str(row.get('coding', ''))
            if coding:
                parts.append(f"Coding: {coding}")
            
            filtering = self._safe_str(row.get('filtering', ''))
            if filtering:
                parts.append(f"Filtering: {filtering}")
            
            ref_unit = self._safe_str(row.get('reference_unit', ''))
            if ref_unit:
                parts.append(f"Unit: {ref_unit}")
            
            ref_period = self._safe_str(row.get('reference_period', ''))
            if ref_period:
                parts.append(f"Period: {ref_period}")
            
            desc_long = ' | '.join(parts) if parts else desc_short
            
            var = {
                'variable_id': f"HFCS_{self._safe_str(row['variable_code'])}",
                'survey': 'HFCS',
                'code': self._safe_str(row['variable_code']),
                'name_en': desc_short,
                'name_fr': desc_short,
                'category': self._infer_category_hfcs(row['variable_code']),
                'subcategory': self._safe_str(row.get('document_type', '')),
                'topic': '',
                'unit': ref_unit,
                'reference_period': ref_period,
                'format': 'text',
                'description_short': desc_short,
                'description_long': desc_long,
                'values_format': coding,
                'tags': [],
                'keywords_positive': [],
                'keywords_negative': [],
                'related_vars': {},
                'metadata': {}
            }
            unified.append(var)
        
        avg_short = sum(len(v['description_short']) for v in unified) / len(unified)
        avg_long = sum(len(v['description_long']) for v in unified) / len(unified)
        print(f"      Short moyen: {avg_short:.0f} chars")
        print(f"      Long moyen: {avg_long:.0f} chars")
        
        return unified
    
    def harmonize_lfs(self, df: pd.DataFrame) -> List[Dict]:
        """
        EU-LFS ENRICHI : Ajoute section + category + type + codes + remarks
        """
        print("   🔧 Harmonisation EU-LFS (ENRICHIE)...")
        
        unified = []
        for _, row in df.iterrows():
            desc_short = self._safe_str(row.get('short_description', ''))
            if not desc_short:
                desc_short = self._safe_str(row.get('description', ''))
            
            # ENRICHISSEMENT EU-LFS
            parts = []
            
            # Base : long_description
            long_desc = self._safe_str(row.get('long_description', ''))
            if long_desc:
                parts.append(long_desc)
            
            # Section
            section = self._safe_str(row.get('section', ''))
            if section:
                parts.append(f"Section: {section}")
            
            # Category
            category = self._safe_str(row.get('variable_category', ''))
            if category:
                parts.append(f"Category: {category}")
            
            # Type
            var_type = self._safe_str(row.get('variable_type', ''))
            if var_type:
                parts.append(f"Type: {var_type}")
            
            # Periodicity
            periodicity = self._safe_str(row.get('periodicity', ''))
            if periodicity:
                parts.append(f"Periodicity: {periodicity}")
            
            # Codes
            codes = self._safe_str(row.get('codes', ''))
            if codes:
                parts.append(f"Codes: {codes}")
            
            # Remarks
            remarks = self._safe_str(row.get('remarks', ''))
            if remarks:
                parts.append(f"Remarks: {remarks}")
            
            desc_long = ' | '.join(parts) if parts else desc_short
            
            var = {
                'variable_id': f"LFS_{self._safe_str(row['variable_code'])}",
                'survey': 'EU-LFS',
                'code': self._safe_str(row['variable_code']),
                'name_en': desc_short,
                'name_fr': desc_short,
                'category': category,
                'subcategory': section,
                'topic': '',
                'unit': '',
                'reference_period': periodicity,
                'format': var_type,
                'description_short': desc_short,
                'description_long': desc_long,
                'values_format': codes,
                'tags': [],
                'keywords_positive': [],
                'keywords_negative': [],
                'related_vars': {},
                'metadata': {}
            }
            unified.append(var)
        
        avg_short = sum(len(v['description_short']) for v in unified) / len(unified)
        avg_long = sum(len(v['description_long']) for v in unified) / len(unified)
        print(f"      Short moyen: {avg_short:.0f} chars")
        print(f"      Long moyen: {avg_long:.0f} chars")
        
        return unified
    
    def harmonize_hbs(self, df: pd.DataFrame) -> List[Dict]:
        """
        HBS ENRICHI : Ajoute file_type + category + format + values + notes
        """
        print("   🔧 Harmonisation HBS (ENRICHIE)...")
        
        unified = []
        for _, row in df.iterrows():
            desc_short = self._safe_str(row.get('short_description', ''))
            if not desc_short:
                desc_short = self._safe_str(row.get('description', ''))
            
            # ENRICHISSEMENT HBS
            parts = []
            
            # Base : long_description
            long_desc = self._safe_str(row.get('long_description', ''))
            if long_desc:
                parts.append(long_desc)
            
            # File type
            file_type = self._safe_str(row.get('file_type', ''))
            if file_type:
                parts.append(f"File type: {file_type}")
            
            # Category
            category = self._safe_str(row.get('variable_category', ''))
            if category:
                parts.append(f"Category: {category}")
            
            # Format
            fmt = self._safe_str(row.get('format', ''))
            if fmt:
                parts.append(f"Format: {fmt}")
            
            # Possible values
            values = self._safe_str(row.get('possible_values', ''))
            if values:
                parts.append(f"Possible values: {values}")
            
            # Notes
            notes = self._safe_str(row.get('notes', ''))
            if notes:
                parts.append(f"Notes: {notes}")
            
            desc_long = ' | '.join(parts) if parts else desc_short
            
            var = {
                'variable_id': f"HBS_{self._safe_str(row['variable_code'])}",
                'survey': 'HBS',
                'code': self._safe_str(row['variable_code']),
                'name_en': desc_short,
                'name_fr': desc_short,
                'category': category,
                'subcategory': file_type,
                'topic': '',
                'unit': '',
                'reference_period': '',
                'format': fmt,
                'description_short': desc_short,
                'description_long': desc_long,
                'values_format': values,
                'tags': [],
                'keywords_positive': [],
                'keywords_negative': [],
                'related_vars': {},
                'metadata': {}
            }
            unified.append(var)
        
        avg_short = sum(len(v['description_short']) for v in unified) / len(unified)
        avg_long = sum(len(v['description_long']) for v in unified) / len(unified)
        print(f"      Short moyen: {avg_short:.0f} chars")
        print(f"      Long moyen: {avg_long:.0f} chars")
        
        return unified
    
    def _infer_category_hfcs(self, code) -> str:
        if pd.isna(code):
            return ''
        code_str = str(code)
        if len(code_str) < 2:
            return 'other'
        prefix = code_str[:2].upper()
        mapping = {
            'HY': 'income', 'HG': 'income', 'DA': 'wealth',
            'HB': 'housing', 'HC': 'credit', 'HD': 'assets',
            'SA': 'technical', 'RA': 'personal', 'PA': 'personal',
            'PE': 'employment'
        }
        return mapping.get(prefix, 'other')
    
    def unify_all(self) -> List[Dict]:
        print("="*80)
        print("🔄 UNIFICATION ENRICHIE DES 4 ENQUÊTES")
        print("="*80)
        
        surveys = [
            ('eusilc_optimized.xlsx', 'EU-SILC', self.harmonize_silc),
            ('hfcs_variables.xlsx', 'HFCS', self.harmonize_hfcs),
            ('eulfs_variables.xlsx', 'EU-LFS', self.harmonize_lfs),
            ('hbs_variables.xlsx', 'HBS', self.harmonize_hbs)
        ]
        
        for filename, survey_name, harmonize_func in surveys:
            df = self.load_survey(filename, survey_name)
            unified = harmonize_func(df)
            self.unified_variables.extend(unified)
            print(f"   ✅ {len(unified)} variables unifiées")
        
        print(f"\n{'='*80}")
        print(f"✅ TOTAL: {len(self.unified_variables)} variables unifiées")
        print(f"{'='*80}")
        
        return self.unified_variables
    
    def save_unified(self) -> Path:
        output_file = self.output_dir / 'unified_variables.json'
        
        print(f"\n💾 Sauvegarde vers {output_file}...")
        
        data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_variables': len(self.unified_variables),
                'enrichment_strategy': 'All 4 surveys use ALL available metadata fields',
                'surveys': {
                    'EU-SILC': len([v for v in self.unified_variables if v['survey'] == 'EU-SILC']),
                    'HFCS': len([v for v in self.unified_variables if v['survey'] == 'HFCS']),
                    'EU-LFS': len([v for v in self.unified_variables if v['survey'] == 'EU-LFS']),
                    'HBS': len([v for v in self.unified_variables if v['survey'] == 'HBS'])
                }
            },
            'variables': self.unified_variables
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON sauvegardé ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        output_jsonl = self.output_dir / 'unified_variables.jsonl'
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for var in self.unified_variables:
                f.write(json.dumps(var, ensure_ascii=False) + '\n')
        
        print(f"✅ JSONL sauvegardé ({output_jsonl.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return output_file
 
 
def main():
    input_dir = Path('../data/input')
    output_dir = Path('../data/unified')
    
    if not input_dir.exists():
        input_dir = Path('data/input')
        output_dir = Path('data/unified')
    
    unifier = SurveyUnifier(input_dir, output_dir)
    
    try:
        unified_vars = unifier.unify_all()
        output_file = unifier.save_unified()
        
        print(f"\n{'='*80}")
        print("📊 STATISTIQUES FINALES")
        print(f"{'='*80}")
        
        for survey in ['EU-SILC', 'HFCS', 'EU-LFS', 'HBS']:
            count = len([v for v in unified_vars if v['survey'] == survey])
            print(f"  {survey}: {count} variables")
        
        print(f"\n  TOTAL: {len(unified_vars)} variables")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: {e}")
 
 
if __name__ == '__main__':
    main()