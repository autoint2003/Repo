"""
Analysis of Complex Word Identification Improvements

This script demonstrates the improvements made to complex word identification
and compares different threshold settings.
"""

import spacy
import csv
from pathlib import Path
from wordfreq import zipf_frequency
from collections import defaultdict

class ComplexWordAnalyzer:
    def __init__(self, threshold=4.0, min_word_length=4):
        self.nlp = spacy.load("en_core_web_sm")
        self.threshold = threshold
        self.min_word_length = min_word_length
        self.p4_exclusion_words = self._load_whitelist_words()

    def _get_token_lemma(self, token):
        """
        Return a normalized lemma for whitelist and frequency checks.
        """
        lemma = token.lemma_.strip().lower()
        if not lemma or lemma == "-pron-":
            return token.text.lower()
        return lemma

    def _get_easy_variant_forms(self, token):
        """
        Generate simple surface-form fallbacks for common inflections.
        """
        word = token.text.lower().strip()
        variants = set()

        if word.endswith("ies") and len(word) > 4:
            variants.add(word[:-3] + "y")
        if word.endswith("ied") and len(word) > 4:
            variants.add(word[:-3] + "y")
        if word.endswith("ing") and len(word) > 5:
            variants.add(word[:-3])
            variants.add(word[:-3] + "e")
            if len(word) > 6 and word[-4] == word[-5]:
                variants.add(word[:-4])
        if word.endswith("ed") and len(word) > 4:
            variants.add(word[:-2])
            variants.add(word[:-1])
            if word.endswith("tted") or word.endswith("pped") or word.endswith("nned"):
                variants.add(word[:-3])
        if word.endswith("es") and len(word) > 4:
            variants.add(word[:-2])
        if word.endswith("s") and len(word) > 3:
            variants.add(word[:-1])

        variants.discard("")
        return variants

    def _get_token_forms(self, token):
        """
        Return normalized candidate forms for whitelist and frequency checks.
        """
        forms = {token.text.lower(), self._get_token_lemma(token)}
        forms.update(self._get_easy_variant_forms(token))
        return {form for form in forms if form}

    def _get_matching_p4_word(self, token):
        """
        Return the matching whitelist form when the token is considered easy.
        """
        for form in self._get_token_forms(token):
            if form in self.p4_exclusion_words:
                return form
        return None

    def _get_word_frequency(self, token):
        """
        Use the easiest recognized form so inflections inherit base-word frequency.
        """
        return max(zipf_frequency(form, 'en') for form in self._get_token_forms(token))

    def _get_complexity_key(self, token):
        """
        Group variants by lemma to reduce duplicate review entries.
        """
        return self._get_token_lemma(token)

    def _load_whitelist_words(self):
        """
        Load words to exclude from complex-word detection.
        Sources:
        - P4wordlist.csv
        - sgwhitelist.csv
        Supports one word per line and comma-separated rows.
        """
        base_dir = Path(__file__).resolve().parent
        whitelist_paths = [
<<<<<<< HEAD:scripts/analyze_complex_word_identification.py
            base_dir / "data" / "P4wordlist.csv",
            base_dir / "data" / "sgwhitelist.csv",
=======
            base_dir / "P4wordlist.csv",
            base_dir / "sgwhitelist.csv",
>>>>>>> c2b0327218583b633f405b994dc03b186d1715ee:analyze_complex_word_identification.py
        ]
        exclusion_words = set()

        for whitelist_path in whitelist_paths:
            if not whitelist_path.exists():
                continue

            with whitelist_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    for cell in row:
                        word = cell.strip().lower()
                        if word:
                            exclusion_words.add(word)

        return exclusion_words
    
    def _is_complex_word(self, token, use_improved_filters=True):
        """
        Check if word is complex using improved or basic filters.
        """
        # Basic filters (always applied)
        if token.is_stop or token.is_punct:
            return False

        if self._get_matching_p4_word(token):
            return False
        
        word_freq = self._get_word_frequency(token)
        if word_freq >= self.threshold:
            return False
        
        if use_improved_filters: # Improved filters
            if token.ent_type_: # Skip named entities
                return False

            if token.pos_ == 'PROPN':  # Exclude proper nouns
                return False
            
            if token.like_num or token.pos_ == 'NUM':  # Exclude numbers
                return False
            
            if len(token.text) < self.min_word_length:  # Exclude short words
                return False
            
            if token.text.isupper() and len(token.text) > 1:  # Exclude acronyms
                return False
            
            # Only consider words with WordNet POS tags
            wn_pos_map = {"NOUN", "VERB", "ADJ", "ADV"}
            if token.pos_ not in wn_pos_map:
                return False
        
        return True
    
    def analyze_text(self, text, use_improved=True):
        """
        Analyze text and return statistics about complex words.
        Uses dictionaries to track unique words and their occurrence counts.
        """
        doc = self.nlp(text)
        
        stats = {
            'total_tokens': 0,
            'complex_words': {},  # {word: {'count': int, 'pos': str, 'freq': float}}
            'excluded_p4_words': {},
            'excluded_proper_nouns': {},  # {word: {'count': int, 'freq': float}}
            'excluded_numbers': {},
            'excluded_acronyms': {},
            'excluded_short': {},
            'excluded_wrong_pos': {},  # {word: {'count': int, 'pos': str, 'freq': float}}
            'borderline_words': {}  # Words close to threshold (within 0.3)
        }
        
        for token in doc:
            stats['total_tokens'] += 1
            
            if token.is_stop or token.is_punct:
                continue
            
            word_lower = self._get_complexity_key(token)
            word_freq = self._get_word_frequency(token)
            matched_p4_word = self._get_matching_p4_word(token)

            if matched_p4_word:
                if word_lower not in stats['excluded_p4_words']:
                    stats['excluded_p4_words'][word_lower] = {
                        'count': 0, 'freq': word_freq, 'original': token.text, 'matched_word': matched_p4_word
                    }
                stats['excluded_p4_words'][word_lower]['count'] += 1
                continue
            
            # Check if it's complex before filters
            if word_freq < self.threshold:
                # Categorize why it might be excluded
                if use_improved:
                    if token.pos_ == 'PROPN':
                        if word_lower not in stats['excluded_proper_nouns']:
                            stats['excluded_proper_nouns'][word_lower] = {
                                'count': 0, 'freq': word_freq, 'original': token.text
                            }
                        stats['excluded_proper_nouns'][word_lower]['count'] += 1
                        continue
                    
                    if token.pos_ == 'NUM':
                        if word_lower not in stats['excluded_numbers']:
                            stats['excluded_numbers'][word_lower] = {
                                'count': 0, 'freq': word_freq, 'original': token.text
                            }
                        stats['excluded_numbers'][word_lower]['count'] += 1
                        continue
                    
                    if len(token.text) < self.min_word_length:
                        if word_lower not in stats['excluded_short']:
                            stats['excluded_short'][word_lower] = {
                                'count': 0, 'freq': word_freq, 'original': token.text
                            }
                        stats['excluded_short'][word_lower]['count'] += 1
                        continue
                    
                    if token.text.isupper() and len(token.text) > 1:
                        if word_lower not in stats['excluded_acronyms']:
                            stats['excluded_acronyms'][word_lower] = {
                                'count': 0, 'freq': word_freq, 'original': token.text
                            }
                        stats['excluded_acronyms'][word_lower]['count'] += 1
                        continue
                    
                    wn_pos_map = {"NOUN", "VERB", "ADJ", "ADV"}
                    if token.pos_ not in wn_pos_map:
                        if word_lower not in stats['excluded_wrong_pos']:
                            stats['excluded_wrong_pos'][word_lower] = {
                                'count': 0, 'pos': token.pos_, 'freq': word_freq, 'original': token.text
                            }
                        stats['excluded_wrong_pos'][word_lower]['count'] += 1
                        continue
                
                # If we get here, it's identified as complex
                if word_lower not in stats['complex_words']:
                    stats['complex_words'][word_lower] = {
                        'count': 0, 'pos': token.pos_, 'freq': word_freq, 'original': token.text, 'lemma': self._get_token_lemma(token)
                    }
                stats['complex_words'][word_lower]['count'] += 1
                
                # Check if borderline
                if self.threshold - word_freq < 0.3:
                    if word_lower not in stats['borderline_words']:
                        stats['borderline_words'][word_lower] = {
                            'count': 0, 'freq': word_freq, 'original': token.text
                        }
                    stats['borderline_words'][word_lower]['count'] += 1
        
        return stats
    
    def print_analysis(self, stats):
        """
        Print detailed analysis of complex word identification.
        Now works with dictionary-based data structure.
        """
        print("\n" + "="*80)
        print("COMPLEX WORD IDENTIFICATION ANALYSIS")
        print("="*80)
        print(f"Threshold: {self.threshold} | Min word length: {self.min_word_length}")
        print(f"\nTotal tokens: {stats['total_tokens']}")
        
        # Calculate total occurrences of complex words
        total_complex_occurrences = sum(data['count'] for data in stats['complex_words'].values())
        unique_complex_words = len(stats['complex_words'])
        
        print(f"Unique complex words: {unique_complex_words}")
        print(f"Total complex word occurrences: {total_complex_occurrences} ({total_complex_occurrences/stats['total_tokens']*100:.1f}%)")
        
        print(f"\n--- IMPROVED FILTERS EXCLUDED ---")
        print(f"P4 wordlist matches (unique): {len(stats['excluded_p4_words'])}")
        if stats['excluded_p4_words']:
            sorted_p4 = sorted(stats['excluded_p4_words'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
            for word, data in sorted_p4:
                print(f"  - '{data['original']}' -> '{data['matched_word']}' (freq: {data['freq']:.2f}, occurrences: {data['count']})")
            if len(stats['excluded_p4_words']) > 5:
                print(f"  ... and {len(stats['excluded_p4_words'])-5} more")

        print(f"Proper nouns (unique): {len(stats['excluded_proper_nouns'])}")
        if stats['excluded_proper_nouns']:
            sorted_proper = sorted(stats['excluded_proper_nouns'].items(), 
                                  key=lambda x: x[1]['count'], reverse=True)[:5]
            for word, data in sorted_proper:
                print(f"  - '{data['original']}' (freq: {data['freq']:.2f}, occurrences: {data['count']})")
            if len(stats['excluded_proper_nouns']) > 5:
                print(f"  ... and {len(stats['excluded_proper_nouns'])-5} more")
        
        print(f"\nNumbers (unique): {len(stats['excluded_numbers'])}")
        if stats['excluded_numbers']:
            sorted_nums = sorted(stats['excluded_numbers'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
            for word, data in sorted_nums:
                print(f"  - '{data['original']}' (freq: {data['freq']:.2f}, occurrences: {data['count']})")
        
        print(f"\nAcronyms (unique): {len(stats['excluded_acronyms'])}")
        if stats['excluded_acronyms']:
            sorted_acro = sorted(stats['excluded_acronyms'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
            for word, data in sorted_acro:
                print(f"  - '{data['original']}' (freq: {data['freq']:.2f}, occurrences: {data['count']})")
        
        print(f"\nShort words (unique, < {self.min_word_length} chars): {len(stats['excluded_short'])}")
        if stats['excluded_short']:
            sorted_short = sorted(stats['excluded_short'].items(), 
                                key=lambda x: x[1]['count'], reverse=True)[:5]
            for word, data in sorted_short:
                print(f"  - '{data['original']}' (freq: {data['freq']:.2f}, occurrences: {data['count']})")
        
        print(f"\n--- COMPLEX WORDS BY FREQUENCY RANGE ---")
        freq_ranges = {
            'Very low (0-2)': [],
            'Low (2-3)': [],
            'Medium-low (3-4)': [],
            'Borderline (4-{})'.format(self.threshold): []
        }
        
        for word, data in stats['complex_words'].items():
            freq = data['freq']
            pos = data['pos']
            count = data['count']
            original = data['original']
            
            if freq < 2:
                freq_ranges['Very low (0-2)'].append((original, pos, freq, count))
            elif freq < 3:
                freq_ranges['Low (2-3)'].append((original, pos, freq, count))
            elif freq < 4:
                freq_ranges['Medium-low (3-4)'].append((original, pos, freq, count))
            else:
                freq_ranges[f'Borderline (4-{self.threshold})'].append((original, pos, freq, count))
        
        for range_name, words in freq_ranges.items():
            if words:
                print(f"\n{range_name}: {len(words)} unique words")
                # Sort by frequency, then by occurrence count
                for original, pos, freq, count in sorted(words, key=lambda x: (x[2], -x[3]))[:10]:
                    print(f"  - '{original}' ({pos}, freq: {freq:.2f}, occurrences: {count})")
                if len(words) > 10:
                    print(f"  ... and {len(words)-10} more")


if __name__ == "__main__":
    # Load row 3 from CSV
    csv_path = Path(__file__).resolve().parent.parent / "data" / "cna_articles.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        row_3 = rows[3]
    
    text = row_3["body_content"]
    title = row_3["title"]
    
    print("="*80)
    print("COMPLEX WORD IDENTIFICATION - Row 3 Analysis")
    print("="*80)
    print(f"\nTitle: {title}")
    print(f"\nText length: {len(text)} characters\n")
    
    # Analyze with default threshold
    analyzer = ComplexWordAnalyzer(threshold=4.5, min_word_length=4)
    stats = analyzer.analyze_text(text, use_improved=True)
    analyzer.print_analysis(stats)
