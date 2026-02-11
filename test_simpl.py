import spacy
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from sentence_transformers import SentenceTransformer, util
import lemminflect
from lemminflect import getInflection
from tqdm import tqdm
import pandas as pd

class LexicalSimplifier:
    def __init__(self, threshold=4.5, similarity_cutoff=0.35):
        # set threshold for complexity
        # set similarity cutoff for semantic similarity
        # Load pre-trained models (Zero-Dataset approach)
        self.nlp = spacy.load("en_core_web_sm")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff
        
    def _get_wordnet_pos(self, spacy_pos):
        """Map spaCy POS tags to WordNet POS tags."""
        pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
        return pos_map.get(spacy_pos, None)

    def simplify_text(self, text):
        print("Processing text...")
        doc = self.nlp(text)
        simplified_tokens = []

        # Using tqdm for visibility on longer news articles
        for token in tqdm(doc, desc="Simplifying"):
            # 1. IDENTIFY: Is the word complex? (Exclude stop words/punctuation)
            if not token.is_stop and not token.is_punct and zipf_frequency(token.text, 'en') < self.threshold:
                word_freq = zipf_frequency(token.text, 'en')
                print(f"\n🔍 COMPLEX WORD IDENTIFIED: '{token.text}' (POS: {token.pos_}, Frequency: {word_freq:.2f})")
                
                # 2. GENERATE: Get candidates from WordNet
                wn_pos = self._get_wordnet_pos(token.pos_)
                candidates = set()
                if wn_pos:
                    for synset in wn.synsets(token.text, pos=wn_pos):
                        for lemma in synset.lemmas():
                            cand = lemma.name().replace('_', ' ')
                            # Only keep candidates that are objectively simpler
                            if zipf_frequency(cand, 'en') > zipf_frequency(token.text, 'en'):
                                candidates.add(cand)

                print(f"   📋 Generated {len(candidates)} candidate(s): {candidates if candidates else 'None'}")

                # 3. SELECT & INFLECT: Find the best match
                best_cand = None
                max_sim = -1
                candidate_scores = []
                
                for cand in candidates:
                    # Inflect candidate to match original token's tag (e.g., VBD, NNS)
                    inflected_forms = getInflection(cand, tag=token.tag_)
                    if not inflected_forms: continue
                    inflected = inflected_forms[0]  # Get the first inflection form
                    
                    # Verify Semantic Similarity
                    orig_sim = util.cos_sim(self.sim_model.encode(token.text), 
                                           self.sim_model.encode(inflected)).item()
                    
                    candidate_scores.append((inflected, orig_sim, zipf_frequency(inflected, 'en')))
                    
                    if orig_sim > self.similarity_cutoff and orig_sim > max_sim:
                        max_sim = orig_sim
                        best_cand = inflected
                
                # Print ranking of alternatives
                if candidate_scores:
                    print(f"   📊 RANKING OF ALTERNATIVES (sorted by similarity):")
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    for i, (word, sim, freq) in enumerate(candidate_scores, 1):
                        status = "✅ SELECTED" if word == best_cand else "❌ Below threshold" if sim <= self.similarity_cutoff else "⚠️  Not best"
                        print(f"      {i}. '{word}' - Similarity: {sim:.4f}, Frequency: {freq:.2f} {status}")
                
                if best_cand:
                    print(f"   ✨ REPLACEMENT: '{token.text}' → '{best_cand}'")
                else:
                    print(f"   ⚠️  NO SUITABLE REPLACEMENT (keeping original)")
                
                simplified_tokens.append(best_cand if best_cand else token.text)
            else:
                simplified_tokens.append(token.text)

        # Reconstruct sentence (simple join for demo)
        return " ".join(simplified_tokens).replace(" .", ".").replace(" ,", ",")

# --- Example Usage ---
print("="*80)
print("LEXICAL SIMPLIFICATION DEMONSTRATION")
print("="*80)

simplifier = LexicalSimplifier(threshold=4.5)
news_article = "The government implemented a strategy to diminish the economic volatility."

print(f"\n📝 Original: {news_article}")
print("\n" + "="*80)
print("PROCESSING...")
print("="*80)

result = simplifier.simplify_text(news_article)

print("\n" + "="*80)
print(f"✅ Simplified: {result}")
print("="*80)

# Load the CNA articles CSV file (first 10 rows only)
print("\n\n" + "="*80)
print("TESTING ON REAL NEWS ARTICLE")
print("="*80)

df = pd.read_csv('cna_articles.csv', nrows=10)

# Display basic information about the dataframe
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Find the first row with valid body_content
article_text = None
article_idx = None
for idx in range(1,len(df)):
    text = df['body_content'].iloc[idx]
    if pd.notna(text) and isinstance(text, str) and len(text.strip()) > 0:
        article_text = text
        article_idx = idx
        break

if article_text is None:
    print("\n⚠️  No valid articles found in the dataset!")
else:
    print("\n" + "="*80)
    print("PROCESSING ARTICLE...")
    print("="*80)

    simplified_article = simplifier.simplify_text(article_text)

    print(f"\n📰 Using article at index {article_idx}")
    print(f"\n--- Original Article ---\n{article_text[:1000]}..." if len(article_text) > 1000 else f"\n--- Original Article ---\n{article_text}")

    print("\n" + "="*80)
    print(f"--- Simplified Article ---\n{simplified_article[:1000]}..." if len(simplified_article) > 1000 else f"--- Simplified Article ---\n{simplified_article}")
    print("="*80)