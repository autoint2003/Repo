"""
Analysis of Complex Word Identification Improvements

This script demonstrates the improvements made to complex word identification
and compares different threshold settings.
"""

import spacy
from wordfreq import zipf_frequency
from collections import defaultdict

class ComplexWordAnalyzer:
    def __init__(self, threshold=4.5, min_word_length=4):
        self.nlp = spacy.load("en_core_web_sm")
        self.threshold = threshold
        self.min_word_length = min_word_length
    
    def _is_complex_word(self, token, use_improved_filters=True):
        """
        Check if word is complex using improved or basic filters.
        """
        # Basic filters (always applied)
        if token.is_stop or token.is_punct:
            return False
        
        word_freq = zipf_frequency(token.text, 'en')
        if word_freq >= self.threshold:
            return False
        
        if use_improved_filters:
            # Improved filters
            if token.pos_ == 'PROPN':  # Exclude proper nouns
                return False
            
            if token.pos_ == 'NUM':  # Exclude numbers
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
        """
        doc = self.nlp(text)
        
        stats = {
            'total_tokens': 0,
            'complex_words': [],
            'excluded_proper_nouns': [],
            'excluded_numbers': [],
            'excluded_acronyms': [],
            'excluded_short': [],
            'excluded_wrong_pos': [],
            'borderline_words': []  # Words close to threshold (within 0.3)
        }
        
        for token in doc:
            stats['total_tokens'] += 1
            
            if token.is_stop or token.is_punct:
                continue
            
            word_freq = zipf_frequency(token.text, 'en')
            
            # Check if it's complex before filters
            if word_freq < self.threshold:
                # Categorize why it might be excluded
                if use_improved:
                    if token.pos_ == 'PROPN':
                        stats['excluded_proper_nouns'].append((token.text, word_freq))
                        continue
                    
                    if token.pos_ == 'NUM':
                        stats['excluded_numbers'].append((token.text, word_freq))
                        continue
                    
                    if len(token.text) < self.min_word_length:
                        stats['excluded_short'].append((token.text, word_freq))
                        continue
                    
                    if token.text.isupper() and len(token.text) > 1:
                        stats['excluded_acronyms'].append((token.text, word_freq))
                        continue
                    
                    wn_pos_map = {"NOUN", "VERB", "ADJ", "ADV"}
                    if token.pos_ not in wn_pos_map:
                        stats['excluded_wrong_pos'].append((token.text, token.pos_, word_freq))
                        continue
                
                # If we get here, it's identified as complex
                stats['complex_words'].append((token.text, token.pos_, word_freq))
                
                # Check if borderline
                if self.threshold - word_freq < 0.3:
                    stats['borderline_words'].append((token.text, word_freq))
        
        return stats
    
    def print_analysis(self, stats):
        """
        Print detailed analysis of complex word identification.
        """
        print("\n" + "="*80)
        print("COMPLEX WORD IDENTIFICATION ANALYSIS")
        print("="*80)
        print(f"Threshold: {self.threshold} | Min word length: {self.min_word_length}")
        print(f"\nTotal tokens: {stats['total_tokens']}")
        print(f"Complex words identified: {len(stats['complex_words'])} ({len(stats['complex_words'])/stats['total_tokens']*100:.1f}%)")
        
        print(f"\n--- IMPROVED FILTERS EXCLUDED ---")
        print(f"Proper nouns: {len(stats['excluded_proper_nouns'])}")
        if stats['excluded_proper_nouns']:
            for word, freq in stats['excluded_proper_nouns'][:5]:
                print(f"  - '{word}' (freq: {freq:.2f})")
            if len(stats['excluded_proper_nouns']) > 5:
                print(f"  ... and {len(stats['excluded_proper_nouns'])-5} more")
        
        print(f"\nNumbers: {len(stats['excluded_numbers'])}")
        if stats['excluded_numbers']:
            for word, freq in stats['excluded_numbers'][:5]:
                print(f"  - '{word}' (freq: {freq:.2f})")
        
        print(f"\nAcronyms (all caps): {len(stats['excluded_acronyms'])}")
        if stats['excluded_acronyms']:
            for word, freq in stats['excluded_acronyms'][:5]:
                print(f"  - '{word}' (freq: {freq:.2f})")
        
        print(f"\nShort words (< {self.min_word_length} chars): {len(stats['excluded_short'])}")
        if stats['excluded_short']:
            for word, freq in stats['excluded_short'][:5]:
                print(f"  - '{word}' (freq: {freq:.2f})")
        
        print(f"\n--- COMPLEX WORDS BY FREQUENCY RANGE ---")
        freq_ranges = {
            'Very low (0-2)': [],
            'Low (2-3)': [],
            'Medium-low (3-4)': [],
            'Borderline (4-{})'.format(self.threshold): []
        }
        
        for word, pos, freq in stats['complex_words']:
            if freq < 2:
                freq_ranges['Very low (0-2)'].append((word, pos, freq))
            elif freq < 3:
                freq_ranges['Low (2-3)'].append((word, pos, freq))
            elif freq < 4:
                freq_ranges['Medium-low (3-4)'].append((word, pos, freq))
            else:
                freq_ranges[f'Borderline (4-{self.threshold})'].append((word, pos, freq))
        
        for range_name, words in freq_ranges.items():
            if words:
                print(f"\n{range_name}: {len(words)} words")
                for word, pos, freq in sorted(words, key=lambda x: x[2])[:10]:
                    print(f"  - '{word}' ({pos}, freq: {freq:.2f})")
                if len(words) > 10:
                    print(f"  ... and {len(words)-10} more")


if __name__ == "__main__":
    # Test text
    test_article = """Global transportation company FedEx on Monday (Feb 23) filed a lawsuit in the United States Court of International Trade seeking a refund for President Donald Trump's emergency tariffs, one of the highest profile moves to recover funds since the US Supreme Court last week deemed the tariffs illegal. A flood of lawsuits to recover billions of dollars is expected by trade attorneys after the blockbuster ruling. The recovery process still has to be worked out by a lower court, however, complicating the matter. More than US$175 billion in US tariff collections are subject to potential refunds after the US Supreme Court last Friday ruled 6-3 that Trump overstepped his authority by using the International Emergency Economic Powers Act (IEEPA), a sanctions law, to impose tariffs on imported goods, Penn-Wharton Budget Model economists said."""
    
    print("="*80)
    print("COMPARING DIFFERENT THRESHOLD VALUES")
    print("="*80)
    
    # Test with different thresholds
    for threshold in [4.0, 4.5, 5.0]:
        analyzer = ComplexWordAnalyzer(threshold=threshold, min_word_length=4)
        stats = analyzer.analyze_text(test_article, use_improved=True)
        analyzer.print_analysis(stats)
        print("\n")
