import spacy
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from sentence_transformers import SentenceTransformer, util
import lemminflect
from lemminflect import getInflection
from tqdm import tqdm
import pandas as pd
from analyze_complex_word_identification import ComplexWordAnalyzer

class LexicalSimplifier:
    def __init__(self, threshold=4.5, similarity_cutoff=0.35):
        self.nlp = spacy.load("en_core_web_sm")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff
        self.analyzer = ComplexWordAnalyzer(threshold=threshold, min_word_length=4)  # FIX

    def _get_wordnet_pos(self, spacy_pos):
        pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
        return pos_map.get(spacy_pos, None)

    def _reconstruct_text(self, doc, override_index=None, override_text=None):
        """
        Rebuild the original text using spaCy whitespace, optionally replacing one token.
        This avoids the common 'space before punctuation' issues.
        """
        parts = []
        for i, tok in enumerate(doc):
            if override_index is not None and i == override_index:
                parts.append((override_text if override_text is not None else tok.text) + tok.whitespace_)
            else:
                parts.append(tok.text_with_ws)
        return "".join(parts)

    def simplify_text(self, text):
        print("Processing text...")
        doc = self.nlp(text)
        simplified_tokens = [tok.text for tok in doc]

        # Sentence embedding for the ORIGINAL sentence (computed once)
        original_sentence = self._reconstruct_text(doc)
        orig_emb = self.sim_model.encode(
            original_sentence,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        for i, token in enumerate(tqdm(doc, desc="Simplifying")):
            # 1. IDENTIFY: Is the word complex? (Exclude stop words/punctuation)
            if self.analyzer._is_complex_word(token):
                word_freq = zipf_frequency(token.text, "en")
                print(f"\n🔍 COMPLEX WORD IDENTIFIED: '{token.text}' (POS: {token.pos_}, Frequency: {word_freq:.2f})")

                # 2. GENERATE: WordNet candidates (objectively simpler by freq)
                wn_pos = self._get_wordnet_pos(token.pos_)
                candidates = set()
                if wn_pos:
                    for synset in wn.synsets(token.text, pos=wn_pos):
                        for lemma in synset.lemmas():
                            cand = lemma.name().replace("_", " ")
                            if zipf_frequency(cand, "en") > zipf_frequency(token.text, "en"):
                                candidates.add(cand)

                print(f"   📋 Generated {len(candidates)} candidate(s): {candidates if candidates else 'None'}")

                # 3. SELECT & INFLECT: sentence-level similarity
                best_cand = None
                max_sim = -1
                candidate_scores = []

                for cand in candidates:
                    inflected_forms = getInflection(cand, tag=token.tag_)
                    if not inflected_forms:
                        continue
                    inflected = inflected_forms[0]

                    # Build the sentence with THIS token replaced
                    modified_sentence = self._reconstruct_text(doc, override_index=i, override_text=inflected)

                    # Compare embeddings of whole sentences
                    mod_emb = self.sim_model.encode(
                        modified_sentence,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
                    sent_sim = util.cos_sim(orig_emb, mod_emb).item()

                    candidate_scores.append((inflected, sent_sim, zipf_frequency(inflected, "en")))

                    if sent_sim > self.similarity_cutoff and sent_sim > max_sim:
                        max_sim = sent_sim
                        best_cand = inflected

                # Print ranking of alternatives
                if candidate_scores:
                    print("   📊 RANKING OF ALTERNATIVES (sorted by sentence similarity):")
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    for rank, (word, sim, freq) in enumerate(candidate_scores, 1):
                        status = (
                            "✅ SELECTED" if word == best_cand
                            else "❌ Below threshold" if sim <= self.similarity_cutoff
                            else "⚠️  Not best"
                        )
                        print(f"      {rank}. '{word}' - SentSim: {sim:.4f}, Frequency: {freq:.2f} {status}")

                if best_cand:
                    print(f"   ✨ REPLACEMENT: '{token.text}' → '{best_cand}'")
                    simplified_tokens[i] = best_cand
                else:
                    print("   ⚠️  NO SUITABLE REPLACEMENT (keeping original)")

        # Reconstruct output preserving spacing/punctuation
        # (replace tokens but keep original whitespace)
        out_parts = []
        for i, tok in enumerate(doc):
            out_parts.append(simplified_tokens[i] + tok.whitespace_)
        return "".join(out_parts)

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