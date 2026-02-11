# Text Simplification Methodology Documentation

## Overview

The text simplification system in [test_simpl.py](test_simpl.py) implements a **Lexical Simplification** approach that replaces complex words with simpler synonyms while preserving semantic meaning and grammatical correctness. This is a zero-dataset, non-generative approach that does not require fine-tuning or labeled training data.

## Current Methodology: Pipeline-Based Lexical Simplification

### Architecture

The system follows a classic three-stage pipeline architecture:

```
Input Text → IDENTIFY → GENERATE → SELECT → INFLECT → Output Text
```

### Implementation Details

#### 1. Initialization (`__init__`)

**Components Loaded:**
- **spaCy (`en_core_web_sm`)**: POS tagging, tokenization, and linguistic analysis
- **SentenceTransformer (`all-MiniLM-L6-v2`)**: Semantic similarity computation
- **WordNet (via NLTK)**: Synonym dictionary for candidate generation
- **WordFreq**: Corpus-based word frequency database
- **LemmInflect**: Morphological inflection engine

**Hyperparameters:**
- `threshold` (default: 4.5): Zipf frequency cutoff for identifying complex words
- `similarity_cutoff` (default: 0.35): Minimum cosine similarity for replacement acceptance

#### 2. Stage 1: IDENTIFY Complex Words

**Process:**
```python
if not token.is_stop and not token.is_punct and zipf_frequency(token.text, 'en') < threshold:
```

**Criteria for Complexity:**
1. Not a stop word (e.g., "the", "and", "is")
2. Not punctuation
3. Zipf frequency < threshold (4.5)

**Zipf Frequency Scale:**
- 7-8: Very common (e.g., "the", "is", "good")
- 5-6: Common (e.g., "happy", "work", "family")
- 4-5: Moderate (e.g., "strategy", "implement")
- 2-3: Rare (e.g., "diminish", "volatility")
- <2: Very rare (e.g., specialized jargon)

**Rationale:** Words with lower frequency are statistically less familiar to readers.

#### 3. Stage 2: GENERATE Candidates

**Method: WordNet Synonym Lookup**

```python
for synset in wn.synsets(token.text, pos=wn_pos):
    for lemma in synset.lemmas():
        cand = lemma.name().replace('_', ' ')
        if zipf_frequency(cand, 'en') > zipf_frequency(token.text, 'en'):
            candidates.add(cand)
```

**Process:**
1. Map spaCy POS tag to WordNet POS (NOUN, VERB, ADJ, ADV)
2. Retrieve all synsets (semantic groups) for the word
3. Extract all lemmas (word forms) from each synset
4. Filter candidates: Keep only words with **higher frequency** than original

**Key Feature:** Objective simplification criterion (frequency-based filtering)

#### 4. Stage 3: SELECT Best Candidate

**Semantic Similarity Verification:**

```python
orig_sim = util.cos_sim(
    self.sim_model.encode(token.text), 
    self.sim_model.encode(inflected)
).item()
```

**Selection Criteria:**
1. Cosine similarity > `similarity_cutoff` (0.35)
2. Highest similarity among all valid candidates

**Purpose:** Ensures semantic preservation - prevents meaning drift

#### 5. Stage 4: INFLECT to Match Grammar

**Morphological Matching:**

```python
inflected_forms = getInflection(cand, tag=token.tag_)
```

**Examples:**
- Original: "implemented" (VBD - past tense verb)
- Candidate: "use"
- Inflected: "used" ✓

- Original: "strategies" (NNS - plural noun)
- Candidate: "plan"
- Inflected: "plans" ✓

**Purpose:** Maintains grammatical correctness and sentence fluency

#### 6. Text Reconstruction

**Post-processing:**
- Joins tokens with spaces
- Fixes spacing around punctuation (`. ,` → `.,`)
- Preserves original sentence structure

### Output Format

The system provides verbose diagnostic output:

```
🔍 COMPLEX WORD IDENTIFIED: 'implemented' (POS: VERB, Frequency: 4.23)
   📋 Generated 3 candidate(s): {'use', 'apply', 'utilize'}
   📊 RANKING OF ALTERNATIVES (sorted by similarity):
      1. 'used' - Similarity: 0.6234, Frequency: 6.45 ✅ SELECTED
      2. 'applied' - Similarity: 0.5891, Frequency: 5.12 ⚠️  Not best
      3. 'utilized' - Similarity: 0.2145, Frequency: 3.89 ❌ Below threshold
   ✨ REPLACEMENT: 'implemented' → 'used'
```

## Evaluation on Real Data

The script tests the methodology on two datasets:

1. **Synthetic Example:**
   ```
   "The government implemented a strategy to diminish the economic volatility."
   ```

2. **Real News Articles:** Loads CNA articles from [cna_articles.csv](cna_articles.csv) for realistic testing

## Strengths of Current Approach

### 1. No LLM API Dependency
- Runs entirely offline after model download
- No API costs or rate limits
- Deterministic results (reproducible)

### 2. Explainability
- Every replacement is justified with metrics
- Transparent decision-making process
- Auditable candidate rankings

### 3. Semantic Safety
- Embedding-based similarity check prevents wrong substitutions
- WordNet provides semantically related synonyms
- POS filtering ensures grammatical category preservation

### 4. Linguistic Correctness
- Morphological inflection maintains grammar
- Token-level processing preserves sentence structure
- Punctuation handling maintains readability

### 5. Zero-Shot Capability
- No training data required
- Works on any domain without fine-tuning
- Language-agnostic framework (can extend to other languages with appropriate resources)

## Limitations and Challenges

### 1. WordNet Coverage Gaps

**Problem:** WordNet doesn't contain:
- Proper nouns (e.g., "Brexit", "COVID-19")
- Recent slang/neologisms (e.g., "cryptocurrency", "influencer")
- Domain-specific jargon (e.g., "phenotype", "blockchain")

**Impact:** Complex words without WordNet entries cannot be simplified

### 2. Context Blindness

**Problem:** Token-level processing ignores context

**Example:**
- "The bank raised interest rates" → "bank" (financial institution)
- "The river bank eroded" → "bank" (riverside)

WordNet may suggest "shore" for both cases, which is wrong in financial context.

### 3. Multi-Word Expression Handling

**Problem:** Cannot simplify phrasal verbs or idioms

**Examples:**
- "carried out" (phrasal verb) → processed as separate words
- "piece of cake" (idiom) → treating "piece" and "cake" independently loses meaning

### 4. Overly Conservative Replacement

**Problem:** High similarity threshold (0.35) may reject valid simplifications

**Example:**
- "automobile" → "car" might be rejected if embeddings differ significantly
- Technical terms without close semantic neighbors get no replacement

### 5. Frequency Database Limitations

**Problem:** Zipf frequencies are corpus-dependent

**Issues:**
- Trained on web text, may not reflect all demographics
- Temporal bias (older words may have inflated frequencies)
- Domain bias (general corpus vs. specialized text)

### 6. Sentence-Level Coherence

**Problem:** Word-by-word replacement can create awkward phrasing

**Example:**
- Original: "The implementation was remarkably efficient"
- Simplified: "The use was very efficient"
- Issue: "implementation" → "use" loses the nominal form's clarity

### 7. Processing Speed

**Problem:** Sequential token processing with embedding computation is slow

**Impact:**
- Long articles take significant time
- Real-time applications are infeasible
- Not suitable for batch processing large corpora

## Improvements to Current Approach

### 1. Expand Candidate Generation

**A. Multiple Dictionary Sources**
```python
# Combine WordNet + Wiktionary + ConceptNet
candidates = set()
candidates.update(get_wordnet_synonyms(word))
candidates.update(get_wiktionary_synonyms(word))
candidates.update(get_conceptnet_related(word))
```

**Benefits:**
- Broader coverage of modern vocabulary
- Cross-linguistic resources (ConceptNet)
- Slang and colloquial alternatives

**B. Paraphrase Database (PPDB)**
```python
from ppdb import PPDB
ppdb = PPDB('ppdb-2.0-s-lexical')
paraphrases = ppdb.get_paraphrases(word, pos=pos_tag)
```

**Benefits:**
- Statistically derived from parallel corpora
- Includes phrasal equivalents
- Probabilistic scores for ranking

### 2. Context-Aware Selection

**A. Contextualized Embeddings (BERT-based)**
```python
from transformers import AutoTokenizer, AutoModel

# Instead of: encode(token.text)
# Use: encode entire sentence and extract token embedding
context_embedding = model(sentence)[token_position]
```

**Benefits:**
- Disambiguation based on context
- "bank" (finance) vs "bank" (river) handled correctly
- Better semantic similarity measurement

**B. Sentence-Level Coherence Check**
```python
original_sentence_emb = encode(original_sentence)
simplified_sentence_emb = encode(simplified_sentence)
coherence_score = cosine_sim(original_sentence_emb, simplified_sentence_emb)

if coherence_score < coherence_threshold:
    reject_replacement()
```

**Benefits:**
- Prevents replacements that break sentence meaning
- Global optimization vs. local token-level changes

### 3. Multi-Word Expression Support

**A. Phrase Detection**
```python
# Use spaCy's phrase matcher
from spacy.matcher import PhraseMatcher

phrasal_verbs = ["carry out", "look into", "set up"]
matcher = PhraseMatcher(nlp.vocab)
matcher.add("PHRASAL_VERBS", [nlp(text) for text in phrasal_verbs])
```

**B. Compound Simplification**
```python
if is_phrasal_verb(token_span):
    simplified = simplify_phrase(token_span)  # "carry out" → "do"
else:
    simplified = simplify_token(token)
```

**Benefits:**
- Idiom preservation or proper simplification
- Better handling of collocations

### 4. Adaptive Thresholding

**A. Reader-Level Customization**
```python
# Different thresholds for different audiences
thresholds = {
    'children': 5.5,      # Only very common words
    'ESL': 4.5,           # Moderate simplification
    'general': 3.5,       # Light simplification
    'technical': 2.5      # Minimal simplification
}
```

**B. Dynamic Threshold Based on Document**
```python
# Adjust based on document complexity distribution
avg_complexity = calculate_average_zipf(document)
threshold = adaptive_threshold(avg_complexity, target_readability)
```

**Benefits:**
- Personalized simplification
- Maintains technical precision where needed
- Avoids over-simplification

### 5. Grammaticality Verification

**A. Language Model Scoring**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def perplexity(sentence):
    # Lower perplexity = more grammatical
    return model.compute_perplexity(sentence)

if perplexity(simplified_sent) > perplexity(original_sent) * 1.2:
    reject_replacement()  # Simplified version is too unnatural
```

**Benefits:**
- Catches awkward phrasings
- Ensures fluency
- Ranks alternatives by naturalness

### 6. Readability Metrics Integration

**A. Pre/Post Comparison**
```python
from textstat import flesch_reading_ease, flesch_kincaid_grade

original_fre = flesch_reading_ease(original_text)
simplified_fre = flesch_reading_ease(simplified_text)

improvement = simplified_fre - original_fre
required_improvement = 10  # Flesch score points

if improvement < required_improvement:
    apply_more_aggressive_simplification()
```

**Benefits:**
- Quantifiable simplification goals
- Ensures meaningful improvement
- Avoids unnecessary changes

### 7. Optimization: Batch Processing

**A. Vectorized Similarity Computation**
```python
# Instead of computing similarity one-by-one
# Encode all candidates at once
candidate_embeddings = model.encode(list(candidates))
original_embedding = model.encode(token.text)

# Batch cosine similarity
similarities = util.cos_sim(original_embedding, candidate_embeddings)
```

**Benefits:**
- 10-50x speedup
- GPU acceleration
- Enables real-time processing

### 8. User Feedback Loop

**A. Interactive Mode**
```python
for replacement in proposed_replacements:
    user_input = input(f"Replace '{original}' with '{replacement}'? (y/n/suggest)")
    if user_input == 'y':
        apply_replacement()
    elif user_input == 'suggest':
        alternative = input("Your suggestion: ")
        learn_preference(original, alternative)
```

**B. Learning from Corrections**
```python
# Store user preferences
preference_db[original_word].add(user_preferred_replacement)

# Use in future replacements
if original_word in preference_db:
    candidates = preference_db[original_word] | generate_candidates()
```

**Benefits:**
- Personalized over time
- Domain adaptation
- Continuous improvement

## Alternative Approaches (Non-LLM, Non-API)

### A. Rule-Based Methods

#### 1. Syntactic Simplification Rules

**Technique:** Transform complex sentence structures into simpler forms

**Examples:**
```python
# Passive → Active voice
"The bill was passed by Congress" → "Congress passed the bill"

# Subordinate → Coordinate clauses
"Although it rained, we went out" → "It rained. We went out."

# Relative clause → Separate sentence
"The man who lives next door is a teacher" → "The man lives next door. He is a teacher."
```

**Implementation:**
```python
from spacy.matcher import DependencyMatcher

# Define transformation patterns
patterns = [
    {"passive_to_active": [...dependency pattern...]},
    {"split_relative_clause": [...dependency pattern...]}
]

def apply_rules(sentence):
    for pattern in patterns:
        if matches(sentence, pattern):
            return transform(sentence, pattern)
    return sentence
```

**Advantages:**
- Deterministic and controllable
- No external models needed
- 100% transparent
- Fast execution

**Disadvantages:**
- Requires extensive rule engineering
- Brittle (doesn't generalize well)
- Misses nuanced cases
- High maintenance cost

#### 2. Lexical Substitution Lists

**Technique:** Pre-compiled dictionary of complex → simple mappings

**Implementation:**
```python
# Medical terminology simplification
medical_simplifications = {
    'myocardial infarction': 'heart attack',
    'hypertension': 'high blood pressure',
    'contusion': 'bruise',
    'laceration': 'cut'
}

# Legal jargon simplification
legal_simplifications = {
    'hereinafter': 'from now on',
    'whereas': 'since',
    'forthwith': 'immediately'
}

def simplify_with_dict(text, domain_dict):
    for complex_term, simple_term in domain_dict.items():
        text = text.replace(complex_term, simple_term)
    return text
```

**Advantages:**
- Domain-specific accuracy
- No computation overhead
- Guaranteed correct replacements (if dictionary is curated)
- Easy to update

**Disadvantages:**
- Limited coverage
- Cannot handle variations/inflections automatically
- Requires manual curation
- Context-independent

### B. Statistical Methods

#### 1. N-gram Language Models

**Technique:** Use corpus statistics to rank simplifications

**Implementation:**
```python
from nltk import ngrams, FreqDist
import kenlm  # Efficient n-gram LM

# Train on simple text corpus (e.g., Simple Wikipedia)
lm = kenlm.Model('simple_wiki_5gram.bin')

def score_simplification(original_sent, simplified_sent):
    orig_score = lm.score(original_sent)
    simp_score = lm.score(simplified_sent)
    
    # Higher score = more likely in simple corpus
    return simp_score - orig_score

candidates = [replace_with_synonym1, replace_with_synonym2, ...]
best = max(candidates, key=lambda c: score_simplification(original, c))
```

**Advantages:**
- Captures fluency patterns from simple text
- No labeled pairs needed
- Fast inference
- Works offline

**Disadvantages:**
- Requires large simple text corpus
- Doesn't capture long-range dependencies
- May prefer common but incorrect simplifications

#### 2. Alignment-Based Methods (Monolingual Translation)

**Technique:** Treat simplification as translation from complex → simple English

**Data Required:**
- Parallel corpus: Complex sentences aligned with simplified versions
- Example: Wikipedia ↔ Simple Wikipedia

**Implementation:**
```python
from sentence_transformers import util

# Pre-compute embeddings for Simple Wikipedia sentences
simple_wiki_sentences = load_simple_wiki()
simple_embeddings = model.encode(simple_wiki_sentences, show_progress_bar=True)

def find_simple_equivalent(complex_sentence):
    complex_emb = model.encode(complex_sentence)
    
    # Find most similar simple sentence
    similarities = util.cos_sim(complex_emb, simple_embeddings)
    best_match_idx = similarities.argmax()
    
    return simple_wiki_sentences[best_match_idx]
```

**Advantages:**
- Learns from real simplification examples
- Captures natural simplification patterns
- Handles sentence-level transformations

**Disadvantages:**
- Requires parallel corpus (limited availability)
- Retrieval-based (may not fit context perfectly)
- Storage intensive (needs to store all simple sentences)

#### 3. Readability-Driven Optimization

**Technique:** Iteratively modify text to optimize readability scores

**Implementation:**
```python
from textstat import flesch_reading_ease
from itertools import permutations

def optimize_readability(sentence, max_iterations=100):
    best_sentence = sentence
    best_score = flesch_reading_ease(sentence)
    
    for _ in range(max_iterations):
        # Try different simplifications
        candidates = generate_single_word_replacements(best_sentence)
        
        for candidate in candidates:
            score = flesch_reading_ease(candidate)
            if score > best_score:
                best_score = score
                best_sentence = candidate
    
    return best_sentence
```

**Advantages:**
- Goal-driven optimization
- Quantifiable improvement
- Can combine multiple metrics (Flesch, SMOG, etc.)

**Disadvantages:**
- Readability formulas are crude (focus on word/sentence length)
- No semantic preservation guarantee
- Can produce unnatural text
- Computationally expensive for long texts

### C. Hybrid Approaches

#### 1. WordNet + Statistical Filtering

**Technique:** Combine symbolic knowledge with corpus statistics

**Implementation:**
```python
def get_statistically_validated_synonyms(word, pos):
    # Step 1: Get WordNet candidates
    wn_candidates = get_wordnet_synonyms(word, pos)
    
    # Step 2: Filter by corpus frequency
    freq_filtered = [c for c in wn_candidates 
                     if zipf_frequency(c, 'en') > zipf_frequency(word, 'en')]
    
    # Step 3: Rank by co-occurrence in simple text corpus
    simple_corpus = load_simple_corpus()
    scores = []
    for cand in freq_filtered:
        # How often does this word appear in simple text?
        simple_text_freq = simple_corpus.count(cand) / len(simple_corpus)
        scores.append((cand, simple_text_freq))
    
    # Return top candidates
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

**Advantages:**
- Combines linguistic knowledge + data-driven evidence
- Better coverage than pure rules
- More precise than pure statistics

#### 2. Clustering-Based Synonym Discovery

**Technique:** Group words by embedding similarity, use simpler cluster representatives

**Implementation:**
```python
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# Pre-processing: Cluster all English words by meaning
vocab = load_large_vocabulary()  # e.g., 50K words
embeddings = model.encode(vocab)

# Cluster words with similar meanings
clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)

# For each cluster, identify the simplest word (highest Zipf frequency)
cluster_representatives = {}
for cluster_id in set(clustering.labels_):
    cluster_words = [vocab[i] for i in np.where(clustering.labels_ == cluster_id)[0]]
    simplest_word = max(cluster_words, key=lambda w: zipf_frequency(w, 'en'))
    
    for word in cluster_words:
        cluster_representatives[word] = simplest_word

# Simplification: replace each word with its cluster representative
def simplify_via_clustering(text):
    tokens = tokenize(text)
    return [cluster_representatives.get(t, t) for t in tokens]
```

**Advantages:**
- Data-driven synonym discovery
- Can find non-WordNet synonyms
- Scalable to large vocabularies
- No manual curation

**Disadvantages:**
- Clustering quality varies
- May group antonyms if not careful
- Requires significant preprocessing
- No grammatical awareness

#### 3. Simplification by Exemplar Retrieval

**Technique:** Retrieve similar examples from a simplification database

**Implementation:**
```python
# Build database of simplification examples
examples_db = [
    {"complex": "utilize", "simple": "use", "context": "verb_action"},
    {"complex": "purchase", "simple": "buy", "context": "verb_transaction"},
    ...
]

# Index by embedding for fast retrieval
from annoy import AnnoyIndex

def simplify_by_example(word, context):
    # Find most similar example in database
    query_emb = encode(word + " " + context)
    similar_examples = index.get_nns_by_vector(query_emb, n=5)
    
    # Return the simplification from best match
    return examples_db[similar_examples[0]]['simple']
```

**Advantages:**
- Learns from human simplifications
- Context-aware
- Easy to add new examples

**Disadvantages:**
- Requires curated example database
- Limited to seen patterns
- Retrieval quality depends on embedding quality

### D. Neural Methods (Local, No API)

#### 1. Sequence-to-Sequence Models (Local BART/T5)

**Technique:** Fine-tune pre-trained models on simplification data, run locally

**Implementation:**
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Use a small T5 model that fits on consumer hardware
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Fine-tune on Wikipedia → Simple Wikipedia pairs (offline, once)
# Then use locally without API calls

def simplify_local_t5(text):
    input_text = "simplify: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    outputs = model.generate(input_ids, max_length=150, num_beams=4)
    simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return simplified
```

**Advantages:**
- High-quality simplification
- Learns complex patterns
- No API costs after setup
- Can be fine-tuned on domain data

**Disadvantages:**
- Requires GPU for reasonable speed
- Initial training requires parallel corpus
- Model size (even T5-small is ~200MB)
- Less explainable than rule-based methods

#### 2. Control Codes with Pretrained Models

**Technique:** Use controllable generation with local models

**Implementation:**
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Models like mT5 or BART with control codes
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

def controlled_simplify(text, target_readability='grade_5'):
    # Prepend control code
    input_text = f"<{target_readability}> {text}"
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Advantages:**
- Adjustable simplification level
- Single model for multiple targets
- Better than fixed-rule systems

**Disadvantages:**
- Requires special training with control codes
- Limited availability of such models
- Still computationally expensive

## Comparison Matrix

| Approach | Accuracy | Speed | Explainability | Coverage | Setup Complexity |
|----------|----------|-------|----------------|----------|------------------|
| **Current (WordNet+Embeddings)** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Rule-Based** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| **Dictionary Lookup** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **N-gram LM** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Alignment-Based** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Local T5/BART** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Exemplar Retrieval** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Clustering-Based** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

## Recommended Next Steps

### Immediate (Low-Hanging Fruit)
1. **Add Paraphrase Database (PPDB)** for better candidate generation
2. **Implement batch embedding computation** for 10x speedup
3. **Add readability metrics** (Flesch score) for before/after comparison
4. **Create domain-specific dictionaries** for news article terminology

### Short-Term (1-2 weeks)
1. **Switch to contextualized embeddings** (BERT) for better semantic similarity
2. **Implement phrasal verb detection** and compound simplification
3. **Add grammaticality checker** using perplexity scoring
4. **Build user feedback mechanism** for continuous improvement

### Long-Term (Research Direction)
1. **Fine-tune local T5-small** on Wikipedia → Simple Wikipedia pairs
2. **Implement multi-document simplification** for consistency across corpus
3. **Add control codes** for adjustable simplification levels
4. **Explore reinforcement learning** with human feedback

## Conclusion

The current approach provides a solid foundation for lexical simplification with good explainability and no dependency on external APIs. However, it can be significantly improved by:

1. **Expanding candidate sources** (PPDB, ConceptNet)
2. **Adding context awareness** (BERT embeddings)
3. **Supporting multi-word expressions**
4. **Optimizing for speed** (batch processing)
5. **Incorporating grammaticality checks**

For production use, a **hybrid approach** combining the current WordNet-based system with **local fine-tuned models** (T5-small) would provide the best balance of accuracy, speed, and explainability while avoiding API dependencies and costs.
