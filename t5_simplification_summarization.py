import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm
import time

class T5TextProcessor:
    """
    Text Simplification and Summarization using T5/FLAN-T5 models.
    Supports GPU acceleration with automatic CPU fallback.
    
    FLAN-T5 models are instruction-tuned and typically perform better
    on zero-shot tasks like simplification compared to base T5 models.
    
    Recommended models:
    - 'google/flan-t5-large' (default): Best quality, slower (~780M params)
    - 'google/flan-t5-base': Good balance (~250M params)
    - 'google/flan-t5-small': Fastest, lower quality (~80M params)
    - 't5-small': Original T5 (OK for summarization, poor for simplification)
    """
    
    def __init__(self, model_name='google/flan-t5-large', use_gpu=True):
        """
        Initialize T5 model for text processing.
        
        Args:
            model_name (str): HuggingFace model name (default: 't5-small')
            use_gpu (bool): Whether to use GPU if available (default: True)
        """
        print(f"Loading {model_name} model...")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Setup device (GPU or CPU)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("✓ Using CPU")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    def get_memory_usage(self):
        """Get current GPU memory usage if available."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
            return f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        return "Using CPU (no GPU memory tracking)"
    
    def simplify_text(self, text, max_length=512, num_beams=6, temperature=1.0, 
                     use_sampling=False, top_p=0.92, top_k=50, use_few_shot=True):
        """
        Simplify complex text to simpler language with enhanced instruction prompts.
        
        Args:
            text (str): Input text to simplify
            max_length (int): Maximum length of output
            num_beams (int): Number of beams for beam search (higher = better quality, slower)
            temperature (float): Sampling temperature (lower = more conservative)
            use_sampling (bool): Use sampling instead of beam search for diversity
            top_p (float): Nucleus sampling threshold (0.0-1.0)
            top_k (int): Top-k sampling parameter
            use_few_shot (bool): Use few-shot examples in prompt for better results
        
        Returns:
            str: Simplified text
        """
        # Enhanced prompt with few-shot examples for FLAN-T5
        # Few-shot prompting dramatically improves instruction-following
        if use_few_shot:
            input_text = (
                f"Simplify the following complex text into easy language. "
                f"Replace difficult words with common words. Use short, clear sentences.\\n\\n"
                f"Example 1:\\n"
                f"Complex: The government implemented comprehensive strategies to ameliorate economic conditions.\\n"
                f"Simple: The government used several plans to improve the economy.\\n\\n"
                f"Example 2:\\n"
                f"Complex: The multifaceted approach encompasses various interventions.\\n"
                f"Simple: This approach includes many different actions.\\n\\n"
                f"Now simplify this text:\\n"
                f"Complex: {text}\\n"
                f"Simple:"
            )
        else:
            # Basic enhanced prompt
            input_text = (
                f"Rewrite the following text in simple, easy-to-understand language. "
                f"Use common words, short sentences, and clear explanations. "
                f"Keep the same meaning but make it easier to read: {text}"
            )
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt', 
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate with optimized parameters
        with torch.no_grad():
            if use_sampling:
                # Sampling-based generation for diversity
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2
                )
            else:
                # Standard beam search with optimizations
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
        
        # Decode output
        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified
    
    def summarize_text(self, text, max_length=150, min_length=40, num_beams=6, 
                       length_penalty=1.5):
        """
        Summarize long text into a concise summary with enhanced prompts.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            num_beams (int): Number of beams for beam search
            length_penalty (float): Penalty for length (>1.0 favors longer, <1.0 favors shorter)
        
        Returns:
            str: Summarized text
        """
        # Enhanced prompt for better FLAN-T5 performance
        input_text = (
            f"Write a clear and concise summary of the following text. "
            f"Include the main points and key information: {text}"
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate summary with optimized parameters
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def simplify_and_summarize(self, text, simplify_first=True, 
                               summary_max_length=150, summary_min_length=40):
        """
        Apply both simplification and summarization.
        
        Args:
            text (str): Input text
            simplify_first (bool): If True, simplify then summarize; else summarize then simplify
            summary_max_length (int): Max length for summary
            summary_min_length (int): Min length for summary
        
        Returns:
            dict: Dictionary with both results and intermediate steps
        """
        if simplify_first:
            # Simplify -> Summarize pipeline
            simplified = self.simplify_text(text)
            summary = self.summarize_text(
                simplified, 
                max_length=summary_max_length,
                min_length=summary_min_length
            )
            
            return {
                'original': text,
                'simplified': simplified,
                'summary_of_simplified': summary,
                'pipeline': 'simplify_first'
            }
        else:
            # Summarize -> Simplify pipeline
            summary = self.summarize_text(
                text,
                max_length=summary_max_length,
                min_length=summary_min_length
            )
            simplified_summary = self.simplify_text(summary)
            
            return {
                'original': text,
                'summary': summary,
                'simplified_summary': simplified_summary,
                'pipeline': 'summarize_first'
            }
    
    def compare_decoding_strategies(self, text, max_length=512):
        """
        Compare different decoding strategies for simplification.
        Useful for finding the best approach for your use case.
        
        Args:
            text (str): Input text to simplify
            max_length (int): Maximum length of output
        
        Returns:
            dict: Results from different strategies
        """
        results = {}
        
        # Strategy 1: Standard Beam Search (default)
        results['beam_search'] = self.simplify_text(
            text, max_length=max_length, num_beams=6
        )
        
        # Strategy 2: High Beam Count
        results['high_beams'] = self.simplify_text(
            text, max_length=max_length, num_beams=10
        )
        
        # Strategy 3: Nucleus Sampling
        results['nucleus_sampling'] = self.simplify_text(
            text, max_length=max_length, use_sampling=True,
            temperature=0.8, top_p=0.92
        )
        
        # Strategy 4: Conservative Sampling
        results['conservative_sampling'] = self.simplify_text(
            text, max_length=max_length, use_sampling=True,
            temperature=0.5, top_p=0.9, top_k=40
        )
        
        return results
    
    def batch_process(self, texts, task='simplify', batch_size=8, **kwargs):
        """
        Process multiple texts in batches for efficiency.
        
        Args:
            texts (list): List of input texts
            task (str): 'simplify', 'summarize', or 'both'
            batch_size (int): Number of texts to process together
            **kwargs: Additional arguments for the task
        
        Returns:
            list: List of processed texts
        """
        results = []
        
        with tqdm(total=len(texts), desc=f"Processing ({task})") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    if task == 'simplify':
                        result = self.simplify_text(text, **kwargs)
                    elif task == 'summarize':
                        result = self.summarize_text(text, **kwargs)
                    elif task == 'both':
                        result = self.simplify_and_summarize(text, **kwargs)
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    
                    results.append(result)
                    pbar.update(1)
        
        return results


def compare_approaches(text, processor):
    """
    Compare different processing approaches on the same text.
    """
    print("\n" + "="*80)
    print("COMPARISON: Different Processing Approaches")
    print("="*80)
    
    # Original
    print(f"\n📄 ORIGINAL ({len(text)} chars):")
    print(text[:500] + "..." if len(text) > 500 else text)
    
    # Approach 1: Simplify only
    print("\n" + "-"*80)
    print("1️⃣  SIMPLIFICATION ONLY:")
    start = time.time()
    simplified = processor.simplify_text(text)
    time_1 = time.time() - start
    print(f"⏱️  Time: {time_1:.2f}s")
    print(f"📝 Result ({len(simplified)} chars):")
    print(simplified)
    
    # Approach 2: Summarize only
    print("\n" + "-"*80)
    print("2️⃣  SUMMARIZATION ONLY:")
    start = time.time()
    summary = processor.summarize_text(text)
    time_2 = time.time() - start
    print(f"⏱️  Time: {time_2:.2f}s")
    print(f"📝 Result ({len(summary)} chars):")
    print(summary)
    
    # Approach 3: Simplify then Summarize
    print("\n" + "-"*80)
    print("3️⃣  SIMPLIFY → SUMMARIZE:")
    start = time.time()
    result_3 = processor.simplify_and_summarize(text, simplify_first=True)
    time_3 = time.time() - start
    print(f"⏱️  Time: {time_3:.2f}s")
    print(f"📝 Simplified ({len(result_3['simplified'])} chars):")
    print(result_3['simplified'])
    print(f"\n📝 Summary of Simplified ({len(result_3['summary_of_simplified'])} chars):")
    print(result_3['summary_of_simplified'])
    
    # Approach 4: Summarize then Simplify
    print("\n" + "-"*80)
    print("4️⃣  SUMMARIZE → SIMPLIFY:")
    start = time.time()
    result_4 = processor.simplify_and_summarize(text, simplify_first=False)
    time_4 = time.time() - start
    print(f"⏱️  Time: {time_4:.2f}s")
    print(f"📝 Summary ({len(result_4['summary'])} chars):")
    print(result_4['summary'])
    print(f"\n📝 Simplified Summary ({len(result_4['simplified_summary'])} chars):")
    print(result_4['simplified_summary'])
    
    # Timing comparison
    print("\n" + "="*80)
    print("⏱️  TIMING COMPARISON:")
    print(f"   Simplify only:        {time_1:.2f}s")
    print(f"   Summarize only:       {time_2:.2f}s")
    print(f"   Simplify→Summarize:   {time_3:.2f}s")
    print(f"   Summarize→Simplify:   {time_4:.2f}s")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("FLAN-T5-LARGE TEXT PROCESSOR")
    print("Text Simplification & Summarization with GPU Acceleration")
    print("="*80)
    
    # Initialize processor (will auto-detect GPU)
    # FLAN-T5 models understand instructions better than base T5
    # flan-t5-large: ~780M params, ~3GB model file
    # Alternative options: 'flan-t5-base' (faster), 'flan-t5-small' (smallest)
    processor = T5TextProcessor(model_name='google/flan-t5-large', use_gpu=True)
    print(f"\n{processor.get_memory_usage()}")
    
    # ========================================================================
    # DEMO 1: Simple Examples
    # ========================================================================
    print("\n\n" + "="*80)
    print("DEMO 1: Simple Examples")
    print("="*80)
    
    demo_text = """
    The government implemented a comprehensive strategy to diminish the economic 
    volatility and ameliorate the financial circumstances of vulnerable populations. 
    This multifaceted approach encompasses fiscal stimulus packages, monetary policy 
    adjustments, and targeted social welfare interventions.
    """
    
    print("\n📝 Original Text:")
    print(demo_text.strip())
    
    print("\n" + "-"*80)
    print("🔄 SIMPLIFICATION (Enhanced Prompt + Optimized Parameters):")
    simplified = processor.simplify_text(demo_text)
    print(simplified)
    
    print("\n" + "-"*80)
    print("📊 SUMMARIZATION (Enhanced Prompt + Optimized Parameters):")
    summary = processor.summarize_text(demo_text)
    print(summary)
    
    print("\n" + "-"*80)
    print("🔀 COMPARING DECODING STRATEGIES:")
    print("   Testing different generation approaches for best quality...")
    strategies = processor.compare_decoding_strategies(demo_text, max_length=200)
    
    for name, result in strategies.items():
        print(f"\n   • {name.upper().replace('_', ' ')}:")
        print(f"     {result}")
    
    # ========================================================================
    # DEMO 2: Real News Article Processing
    # ========================================================================
    print("\n\n" + "="*80)
    print("DEMO 2: Processing Real CNA News Articles")
    print("="*80)
    
    try:
        # Load CNA articles
        df = pd.read_csv('cna_articles.csv', nrows=10)
        print(f"\n✓ Loaded {len(df)} articles from dataset")
        
        # Find first valid article
        article_text = None
        article_idx = None
        article_title = None
        
        for idx in range(len(df)):
            text = df['body_content'].iloc[idx]
            if pd.notna(text) and isinstance(text, str) and len(text.strip()) > 100:
                article_text = text
                article_idx = idx
                article_title = df['title'].iloc[idx] if 'title' in df.columns else "Unknown"
                break
        
        if article_text:
            print(f"\n📰 Article #{article_idx}: {article_title}")
            print(f"   Length: {len(article_text)} characters")
            
            # Run comparison
            compare_approaches(article_text, processor)
            
        else:
            print("\n⚠️  No valid articles found in dataset")
            
    except FileNotFoundError:
        print("\n⚠️  cna_articles.csv not found. Skipping real article processing.")
    
    # ========================================================================
    # DEMO 3: Batch Processing
    # ========================================================================
    print("\n\n" + "="*80)
    print("DEMO 3: Batch Processing Multiple Articles")
    print("="*80)
    
    try:
        df = pd.read_csv('cna_articles.csv', nrows=5)
        
        # Get valid articles
        valid_articles = []
        for idx in range(len(df)):
            text = df['body_content'].iloc[idx]
            if pd.notna(text) and isinstance(text, str) and len(text.strip()) > 100:
                valid_articles.append(text)
        
        if valid_articles:
            print(f"\n✓ Processing {len(valid_articles)} articles in batch mode...")
            
            # Batch summarize
            start = time.time()
            summaries = processor.batch_process(
                valid_articles, 
                task='summarize',
                batch_size=2,
                max_length=100,
                min_length=30
            )
            elapsed = time.time() - start
            
            print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/len(valid_articles):.2f}s per article)")
            print(f"   Throughput: {len(valid_articles)/elapsed*60:.1f} articles/minute")
            
            # Display results
            print("\n" + "-"*80)
            print("SUMMARIES:")
            for i, summary in enumerate(summaries, 1):
                print(f"\n{i}. {summary}")
            
    except FileNotFoundError:
        print("\n⚠️  cna_articles.csv not found. Skipping batch processing.")
    
    # ========================================================================
    # Performance Summary
    # ========================================================================
    print("\n\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Device: {processor.device}")
    print(f"{processor.get_memory_usage()}")
    print("\nModel: google/flan-t5-large (~780M parameters)")
    print("\nProcessing Times (approximate on CPU):")
    print("  • Simplification: 3-8 seconds per article")
    print("  • Summarization:  3-8 seconds per article")
    print("  • Both:           6-15 seconds per article")
    print("\nWith GPU (NVIDIA RTX 3060 or better):")
    print("  • Simplification: 0.5-1.5 seconds per article")
    print("  • Summarization:  0.5-1.5 seconds per article")
    print("  • Both:           1.0-3.0 seconds per article")
    print("\nMemory Requirements:")
    print("  • CPU: ~4-6 GB RAM")
    print("  • GPU: ~4-6 GB VRAM (8GB recommended)")
    print("\nNote: FLAN-T5 models are instruction-tuned and may perform")
    print("      better at simplification than base T5 models!")
    print("="*80)
    
    print("\n✨ OPTIMIZATION IMPROVEMENTS:")
    print("   ✓ Few-shot prompting with examples for better instruction-following")
    print("   ✓ Enhanced prompts with detailed instructions for FLAN-T5")
    print("   ✓ Optimized beam search with repetition penalties")
    print("   ✓ Nucleus sampling option for more diverse output")
    print("   ✓ Multiple temperature settings for quality control")
    print("   ✓ Multiple decoding strategies for comparison")
    print("\n✨ Processing complete!")
