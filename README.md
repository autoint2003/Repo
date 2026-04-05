# Text Mining

Python repository for crawling Channel NewsAsia (CNA) articles, simplifying article text, and exporting evaluation results for readability and lexical complexity analysis.

## What this repository does

This project has two main workflows:

1. Crawl CNA articles into structured datasets in `data/`
2. Run a text simplification pipeline over those articles and save timestamped experiment outputs in `results/`

The simplification pipeline currently combines:

- Lexical simplification with WordNet, BERT embeddings, Zipf frequency, and inflection matching
- Syntactic simplification with constituency parsing via Stanza
- Readability scoring with Flesch-Kincaid Grade Level and Flesch Reading Ease
- A complexity comparison step using an LLM judge
- Result visualization with line charts for readability and complexity comparisons

## Repository structure

```text
text_mining/
|- main.py
|- crawl_cna.py
|- requirements.txt
|- data/
|- results/
|- docs/
`- scripts/
```

Key files:

- `main.py`: main pipeline that reads a source CSV, simplifies article text, computes readability metrics, and writes a timestamped CSV to `results/`
- `crawl_cna.py`: crawler for CNA section pages and article pages
- `scripts/text_simplification.py`: lexical simplifier
- `scripts/syntax_simplifier.py`: syntactic simplifier
- `scripts/syntactic_evaluation.py`: readability scoring helpers
- `scripts/complexity_evaluation.py`: LLM-based lexical complexity comparison
- `scripts/plot_results.py`: generates three line-chart PNGs from a results CSV
- `docs/`: methodology and implementation notes

## Data flow

### 1. Crawl source articles

`crawl_cna.py` scrapes CNA article pages and exports fields such as:

- `url`
- `title`
- `section`
- `description`
- `location`
- `body_content`
- `publication_date`
- `category`
- `scraped_at`

By default, source datasets are stored in:

- `data/cna_articles.csv`
- `data/cna_articles.json`

### 2. Run simplification and evaluation

`main.py` reads the source CSV defined by the `SOURCE_DOC` environment variable, then for each row:

- simplifies `body_content` lexically
- simplifies the lexical output syntactically
- evaluates readability before and after simplification
- stores a complexity-comparison result
- writes a new timestamped CSV to `results/`

Typical output columns include:

- `text_simplified`
- `syntactic_simplified`
- `text_vs_text_simplified_complexity_result`
- `text_grade_level`
- `text_reading_ease`
- `text_simplified_grade_level`
- `text_simplified_reading_ease`
- `syntactic_simplified_grade_level`
- `syntactic_simplified_reading_ease`

### 3. Generate plots from a results CSV

`scripts/plot_results.py` reads a timestamped results CSV and writes three line charts:

- grade level comparison for original, lexical simplification, and syntactic simplification
- reading ease comparison for original, lexical simplification, and syntactic simplification
- LLM judge complexity score comparison for `Text A` vs `Text B`

The script takes only the CSV path as input. It uses article titles on the x-axis and normalizes the grade-level and reading-ease plots so the original article score is `100` for each row.

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Core dependencies listed in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Some scripts currently import additional packages that are not listed in `requirements.txt`. Depending on which workflow you run, you may also need:

```powershell
pip install requests beautifulsoup4 textstat tqdm
```

Notes:

- `main.py` will download `nltk` WordNet data if missing
- `main.py` will attempt to download the spaCy model `en_core_web_sm` if it is not installed
- `scripts/syntax_simplifier.py` uses `stanza`, which may download models on first run
- `scripts/complexity_evaluation.py` depends on a package imported as `copilot` and an authenticated Copilot setup
- `scripts/plot_results.py` uses `matplotlib` with the non-interactive `Agg` backend, so it can render PNGs in headless environments

## Environment configuration

The repository uses `.env` for the source dataset path.

Current example:

```env
SOURCE_DOC="data/cna_articles.csv"
```

`main.py` loads this value and reads the CSV from that path.

## Usage

### Crawl CNA articles

Run the crawler:

```powershell
python .\crawl_cna.py
```

The example block at the bottom of the file currently crawls:

- section: `singapore`
- listing pages: `2`
- max articles: `10`

If you want different crawl settings, edit the values in the `if __name__ == "__main__":` block in `crawl_cna.py`.

### Run the simplification pipeline

Make sure `SOURCE_DOC` points to a valid CSV with a `body_content` column, then run:

```powershell
python .\main.py
```

Output is written to:

```text
results/test_YYYYMMDD_HHMMSS.csv
```

### Run individual components

Several scripts include local demo or test entry points:

- `python .\scripts\text_simplification.py`
- `python .\scripts\syntax_simplifier.py`
- `python .\scripts\syntactic_evaluation.py`

### Generate plots for an experiment result

Run the plotting script on a results CSV:

```powershell
.\.venv\Scripts\python .\scripts\plot_results.py .\results\test_20260405_163659.csv
```

Generated images are written to:

```text
results/plots/<results_csv_stem>/
```

The script currently produces:

- `01_grade_levels.png`
- `02_reading_ease.png`
- `03_complexity_scores.png`

## Expected input format

The simplification pipeline assumes the source CSV contains article rows with at least:

- `body_content`

It is designed around the schema produced by `crawl_cna.py`, so using the crawler output as input is the simplest path.

## Outputs

Source data:

- `data/cna_articles.csv`
- `data/cna_articles.json`

Experiment outputs:

- `results/test_*.csv`
- `results/plots/<results_csv_stem>/01_grade_levels.png`
- `results/plots/<results_csv_stem>/02_reading_ease.png`
- `results/plots/<results_csv_stem>/03_complexity_scores.png`

Documentation:

- `docs/CRAWLER_DOCUMENTATION.md`
- `docs/TEXT_SIMPLIFICATION_METHODOLOGY.md`
- `docs/T5_SUMMARIZATION_SIMPLIFICATION_REPORT.md`
- `docs/COMPLEX_WORD_IDENTIFICATION_IMPROVEMENTS.md`

## Limitations

- `main.py` calls an LLM-based complexity judge, so that step is not fully offline
- first-run model downloads can be large and slow
- the lexical simplifier can produce awkward substitutions because it relies on WordNet candidates plus embedding similarity
- the syntactic simplifier may over-split sentences in some cases
- crawling depends on the current CNA page structure and may need updates if the site changes

## Notes for contributors

- main processing assumes a CSV source configured through `.env`
- results are written as new timestamped files rather than overwriting previous experiment outputs
- existing docs in `docs/` contain the detailed design rationale for the crawler and simplification pipeline

## License

No license file is currently present in this repository.
