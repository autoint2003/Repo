'''
Flesch–Kincaid Grade Level (*_grade_level)

What it is: an estimated US school grade level needed to understand the text.
Direction: lower is easier (e.g., 8 ≈ 8th grade; 12 ≈ high school senior; 16+ ≈ college).
Why it changes: driven mostly by sentence length and syllables per word.

Flesch Reading Ease is driven by two things:

Average sentence length (words per sentence)

Longer sentences → score goes down (harder to read)
Average word syllable count (syllables per word)

More syllables per word → score goes down (harder to read)
'''

import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
from datetime import datetime
import nltk
import spacy
from scripts.text_simplification import LexicalSimplifier
from scripts.syntax_simplifier import SyntaxSimplifier
# from scripts.t5_simplification_summarization import T5TextProcessor
from scripts.syntactic_evaluation import analyze_readability
from scripts.complexity_evaluation import judge_complexity

load_dotenv()
DOCUMENTS = os.getenv('SOURCE_DOC')

try:
	nltk.data.find('corpora/wordnet')
except LookupError:
	nltk.download('wordnet')

# Check spaCy (if you didn't add it via URL in uv)
if not spacy.util.is_package("en_core_web_sm"):
	spacy.cli.download("en_core_web_sm")

df = pd.read_csv(DOCUMENTS)
text_simplifier = LexicalSimplifier()
syntactic_simplifier = SyntaxSimplifier()
# llm_text_simplifier = T5TextProcessor()


def _safe_analyze_readability(value):
	if not isinstance(value, str) or not value.strip():
		return None, None
	try:
		grade_level, reading_ease = analyze_readability(value)
		return float(grade_level), float(reading_ease)
	except Exception:
		return None, None


# Ensure output columns exist
if 'syntactic_simplified' not in df.columns:
	df['syntactic_simplified'] = ''

# Store lexical-only simplification output (before syntactic simplification)
if 'text_simplified' not in df.columns:
	df['text_simplified'] = ''

# Add columns to store test results
if 'text_vs_text_simplified_complexity_result' not in df.columns:
	df['text_vs_text_simplified_complexity_result'] = None

# Numeric readability columns (preferred for CSV analysis)
for col in [
	'text_grade_level',
	'text_reading_ease',
	'text_simplified_grade_level',
	'text_simplified_reading_ease',
	'syntactic_simplified_grade_level',
	'syntactic_simplified_reading_ease',
	'syntactic_simplified_minus_text_simplified_grade_level',
	'syntactic_simplified_minus_text_simplified_reading_ease',
]:
	if col not in df.columns:
		df[col] = None


# Iterate through rows; placeholder for future processing function
for idx, row in df.iterrows():
	syntactic_simplified = row.get('syntactic_simplified', '')
	existing_text_simplified = row.get('text_simplified', '')

	_syntactic_simplified_done = (isinstance(syntactic_simplified, str) and syntactic_simplified.strip()) or (
		not isinstance(syntactic_simplified, str) and pd.notna(syntactic_simplified)
	)
	_text_simplified_done = (
		isinstance(existing_text_simplified, str) and existing_text_simplified.strip()
	) or (not isinstance(existing_text_simplified, str) and pd.notna(existing_text_simplified))

	# Only skip rows that already have both outputs populated
	if _syntactic_simplified_done and _text_simplified_done:
		continue
	text = row['body_content']
	text_simplified = text_simplifier.simplify_text(text)
	syntactic_simplified_val = syntactic_simplifier.simplify_text(text_simplified)
	df.at[idx, 'syntactic_simplified'] = syntactic_simplified_val
	df.at[idx, 'text_simplified'] = text_simplified

	# Basic tests between columns (placeholders for more advanced checks)
	# Test 1: compare original text vs text_simplified
	df.at[idx, 'text_vs_text_simplified_complexity_result'] = asyncio.run(judge_complexity(syntactic_simplified_val, text))
	# Test 2 (LLM) removed: llm columns are not used/populated
	# Syntactic/readability evaluation (store both lexical-only and syntax-simplified)
	text_grade, text_ease = _safe_analyze_readability(text)
	text_simplified_grade, text_simplified_ease = _safe_analyze_readability(text_simplified)
	syntactic_simplified_grade, syntactic_simplified_ease = _safe_analyze_readability(syntactic_simplified_val)

	# Preferred numeric columns
	df.at[idx, 'text_grade_level'] = text_grade
	df.at[idx, 'text_reading_ease'] = text_ease
	df.at[idx, 'text_simplified_grade_level'] = text_simplified_grade
	df.at[idx, 'text_simplified_reading_ease'] = text_simplified_ease
	df.at[idx, 'syntactic_simplified_grade_level'] = syntactic_simplified_grade
	df.at[idx, 'syntactic_simplified_reading_ease'] = syntactic_simplified_ease

	if text_simplified_grade is not None and syntactic_simplified_grade is not None:
		df.at[idx, 'syntactic_simplified_minus_text_simplified_grade_level'] = (
			syntactic_simplified_grade - text_simplified_grade
		)
	if text_simplified_ease is not None and syntactic_simplified_ease is not None:
		df.at[idx, 'syntactic_simplified_minus_text_simplified_reading_ease'] = (
			syntactic_simplified_ease - text_simplified_ease
		)



# Reorder readability/result columns to the requested sequence while preserving others
_ordered_readability_cols = [
 'text_vs_text_simplified_complexity_result',
 'text_simplified',
 'syntactic_simplified',
 'text_grade_level',
 'text_simplified_grade_level',
 'syntactic_simplified_grade_level',
 'text_reading_ease',
 'text_simplified_reading_ease',
 'syntactic_simplified_reading_ease',
 'syntactic_simplified_minus_text_simplified_grade_level',
 'syntactic_simplified_minus_text_simplified_reading_ease',
]
_ordered_readability_cols = [c for c in _ordered_readability_cols if c in df.columns]
_remaining_cols = [c for c in df.columns if c not in _ordered_readability_cols]
df = df[_remaining_cols + _ordered_readability_cols]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'results/test_{timestamp}.csv'
df.to_csv(output_file, index=False)