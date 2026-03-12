import spacy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from lemminflect import getInflection
from tqdm import tqdm
import pandas as pd
from analyze_complex_word_identification import ComplexWordAnalyzer

class LexicalSimplifier:
    def __init__(self, threshold=4.5, similarity_cutoff=0.35, batch_size=32, min_word_length=4):
        self.nlp = spacy.load("en_core_web_sm")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Plain BERT
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert.eval()

        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff
        self.batch_size = batch_size
        self.analyzer = ComplexWordAnalyzer(threshold=threshold, min_word_length=min_word_length)

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
    
    def _encode_sentences(self, sentences):
        """
        Batch-encode a list of sentences with offset mappings.
        """
        encoded = self.tokenizer(
            sentences,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            max_length=512
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.bert(**encoded)

        # outputs.last_hidden_state: [B, seq_len, hidden]
        return outputs.last_hidden_state, offset_mapping

    def _span_embedding_from_offsets(self, token_embeddings, offset_mapping, char_start, char_end):
        """
        Extract mean pooled embedding of BERT subtokens whose character spans
        overlap [char_start, char_end).
        token_embeddings: [seq_len, hidden]
        offset_mapping: [seq_len, 2]
        """
        target_indices = []
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start < char_end and end > char_start:
                target_indices.append(i)

        if not target_indices:
            return None

        span_emb = token_embeddings[target_indices, :]
        word_emb = span_emb.mean(dim=0)
        word_emb = F.normalize(word_emb, p=2, dim=0)
        return word_emb

    def _batched_candidate_similarities(
        self,
        original_sentence,
        original_char_start,
        original_char_end,
        candidate_sentences,
        candidate_spans,
    ):
        """
        Compute contextual similarities between the original target span and
        multiple candidate replacement spans in batch.
        """
        # Encode original once
        orig_hidden, orig_offsets = self._encode_sentences([original_sentence])
        orig_emb = self._span_embedding_from_offsets(
            orig_hidden[0], orig_offsets[0], original_char_start, original_char_end
        )
        if orig_emb is None:
            return [None] * len(candidate_sentences)

        sims = []

        for start_idx in range(0, len(candidate_sentences), self.batch_size):
            batch_sentences = candidate_sentences[start_idx:start_idx + self.batch_size]
            batch_spans = candidate_spans[start_idx:start_idx + self.batch_size]

            batch_hidden, batch_offsets = self._encode_sentences(batch_sentences)

            for j, (char_start, char_end) in enumerate(batch_spans):
                cand_emb = self._span_embedding_from_offsets(
                    batch_hidden[j], batch_offsets[j], char_start, char_end
                )

                if cand_emb is None:
                    sims.append(None)
                else:
                    sim = F.cosine_similarity(
                        orig_emb.unsqueeze(0), cand_emb.unsqueeze(0)
                    ).item()
                    sims.append(sim)

        return sims

    def simplify_text(self, text):
        print("Processing text...")
        doc = self.nlp(text)
        simplified_tokens = [tok.text for tok in doc]

        original_sentence = self._reconstruct_text(doc)

        inside_quotes = False

        for i, token in enumerate(tqdm(doc, desc="Simplifying")):
            # Toggle quote state
            if token.text in {'"', "``", "''", "“", "”"}:
                inside_quotes = not inside_quotes
                continue

            # Skip tokens inside quotes
            if inside_quotes:
                continue

            # 1. IDENTIFY: Is the word complex? (Exclude stop words/punctuation)
            if self.analyzer._is_complex_word(token):
                word_freq = zipf_frequency(token.text, "en")
                print(f"\n🔍 COMPLEX WORD IDENTIFIED: '{token.text}' (POS: {token.pos_}, Frequency: {word_freq:.2f})")

                # Original token character span in original_sentence
                original_char_start = token.idx
                original_char_end = token.idx + len(token.text)

                # 2. GENERATE: WordNet candidates (objectively simpler by freq)
                wn_pos = self._get_wordnet_pos(token.pos_)
                candidates = set()

                if wn_pos:
                    for synset in wn.synsets(token.text, pos=wn_pos):
                        for lemma in synset.lemmas():
                            cand = lemma.name().replace("_", " ")

                            # Skip multi-word candidates
                            if " " in cand:
                                continue

                            # Only keep objectively simpler candidates
                            if zipf_frequency(cand, "en") > zipf_frequency(token.text, "en"):
                                candidates.add(cand)

                print(f"   📋 Generated {len(candidates)} candidate(s): {candidates if candidates else 'None'}")

                # 3. SELECT & INFLECT: sentence-level similarity
                # Prepare all candidates first
                prepared_candidates = []
                candidate_sentences = []
                candidate_spans = []

                for cand in candidates:
                    inflected_forms = getInflection(cand, tag=token.tag_)
                    if not inflected_forms:
                        continue

                    inflected = inflected_forms[0]
                    modified_sentence = self._reconstruct_text(
                        doc,
                        override_index=i,
                        override_text=inflected
                    )

                    modified_char_start = token.idx
                    modified_char_end = modified_char_start + len(inflected)

                    prepared_candidates.append(inflected)
                    candidate_sentences.append(modified_sentence)
                    candidate_spans.append((modified_char_start, modified_char_end))

                best_cand = None
                max_sim = -1
                candidate_scores = []

                if candidate_sentences:
                    similarities = self._batched_candidate_similarities(
                        original_sentence=original_sentence,
                        original_char_start=original_char_start,
                        original_char_end=original_char_end,
                        candidate_sentences=candidate_sentences,
                        candidate_spans=candidate_spans,
                    )

                    for inflected, sim in zip(prepared_candidates, similarities):
                        if sim is None:
                            continue

                        freq = zipf_frequency(inflected, "en")
                        candidate_scores.append((inflected, sim, freq))

                        if sim > self.similarity_cutoff and sim > max_sim:
                            max_sim = sim
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