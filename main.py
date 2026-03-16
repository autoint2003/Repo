import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
import nltk
import spacy
# from test_simpl import LexicalSimplifier
# from t5_simplification_summarization import T5TextProcessor
# from scripts.syntactic_evaluation import analyze_readability
# from scripts.complexity_evaluation import judge_complexity
from syntax_simplifier import SyntaxSimplifier


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
# text_simplifier = LexicalSimplifier()
# llm_text_simplifier = T5TextProcessor()
syntax_simplifier = SyntaxSimplifier()

# Ensure 'simplified_body' and 'llm_simplified_body' columns exist
if 'simplified_body' not in df.columns:
	df['simplified_body'] = ''
if 'llm_simplified_body' not in df.columns:
    df['llm_simplified_body'] = ''

# Add columns to store test results
if 'text_vs_text_simplified_complexity_result' not in df.columns:
	df['text_vs_text_simplified_complexity_result'] = None

if 'simplified_body_vs_llm_complexity_result' not in df.columns:
	df['simplified_body_vs_llm_complexity_result'] = None

if 'text_syntactic_evaluation' not in df.columns:
	df['text_syntactic_evaluation'] = None

if 'text_simplified_syntactic_evaluation' not in df.columns:
	df['text_simplified_syntactic_evaluation'] = None
	
if 'llm_syntactic_evaluation' not in df.columns:
	df['llm_syntactic_evaluation'] = None

# Iterate through rows; placeholder for future processing function
for idx, row in df.iterrows():
	simplified_body = row.get('simplified_body', '')
	if (isinstance(simplified_body, str) and simplified_body.strip()) or (
		not isinstance(simplified_body, str) and pd.notna(simplified_body)
	):
		continue
	text = row['body_content']
	# text_simplified = text_simplifier.simplify_text(text)
	# llm_simplified = llm_text_simplifier.simplify_text(text)
	# df.at[idx, 'simplified_body'] = text_simplified    
	# df.at[idx, 'llm_simplified_body'] = llm_simplified  

	syntax_simplified = syntax_simplifier.simplify_text(text)
	df.at[idx, 'simplified_syntax_body'] = syntax_simplified

	# Basic tests between columns (placeholders for more advanced checks)
	
	# # Test 1: compare original text vs text_simplified
	# df.at[idx, 'text_vs_text_simplified_complexity_result'] = asyncio.run(judge_complexity(text_simplified, text))
	# # Test 2: compare simplified_body vs llm_simplified_body
	# df.at[idx, 'simplified_body_vs_llm_complexity_result'] = asyncio.run(judge_complexity(text_simplified, llm_simplified))
	# # Test 3: evaluate syntactic quality of text_simplified
	# df.at[idx, 'text_simplified_syntactic_evaluation'] = analyze_readability(text_simplified)
	# # Test 4: evaluate syntactic quality of llm_simplified
	# df.at[idx, 'llm_syntactic_evaluation'] = analyze_readability(llm_simplified)
	# # Test 5: evaluate syntactic quality of original text
	# df.at[idx, 'text_syntactic_evaluation'] = analyze_readability(text)
	
df.to_csv('cna_articles_results.csv', index=False)