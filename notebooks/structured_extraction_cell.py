# =============================================================================
# Cell 8.5: Structured Bond Extraction (v10.17)
# =============================================================================
# This cell extracts symbolic moral bonds using LLM-based structured extraction
# and compares results with embedding-based predictions.
#
# Insert this cell between Cell 8 (Geometric Analysis) and Cell 9 (Fuzz Testing)
# in your BIP notebook.
#
# Prerequisites:
#   - corpus_cache from Cell 2
#   - model from Cell 7
#   - ANTHROPIC_API_KEY environment variable
# =============================================================================

# @title Cell 8.5: Structured Extraction Analysis {display-mode: "form"}
# @markdown **Structured Moral Bond Extraction**
# @markdown
# @markdown Uses Claude to extract symbolic moral structures, enabling:
# @markdown - Language-neutral comparison
# @markdown - Mathematical analysis of moral space
# @markdown - Comparison with embedding-based predictions

import os
import sys
from pathlib import Path

# Ensure src is importable
if str(Path('.').resolve()) not in sys.path:
    sys.path.insert(0, str(Path('.').resolve()))

from src.bip import (
    MoralBond,
    BondType,
    RoleType,
    ActionCategory,
    CorpusPassage,
    extract_from_corpus_cache,
    extract_from_passage,
    BondDatabase,
    find_universal_patterns,
    compute_tradition_similarity,
    print_algebra_report,
    plot_tradition_heatmap_text,
    plot_bond_space_text,
    cluster_bonds,
    compare_extraction_methods,
    print_comparison_report,
    generate_analysis_report,
)

# =============================================================================
# Configuration
# =============================================================================

# @markdown ---
# @markdown **Extraction Settings**
EXTRACTION_SAMPLE_SIZE = 50  # @param {type:"integer"}
# @markdown - Number of passages to extract (start small, ~$0.15 for 50)
EXTRACTION_LANGUAGES = ["hebrew", "classical_chinese", "arabic", "sanskrit", "english"]  # @param
# @markdown - Languages to include
USE_HAIKU_MODEL = True  # @param {type:"boolean"}
# @markdown - Use claude-3-haiku (cheaper) vs claude-sonnet-4 (better)
COMPARE_WITH_EMBEDDINGS = True  # @param {type:"boolean"}
# @markdown - Compare structured vs embedding predictions

# Model selection
EXTRACTION_MODEL = "claude-3-haiku-20240307" if USE_HAIKU_MODEL else "claude-sonnet-4-20250514"

# =============================================================================
# Run Extraction
# =============================================================================

print("=" * 70)
print("STRUCTURED MORAL BOND EXTRACTION")
print("=" * 70)

# Check for API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("\nWARNING: ANTHROPIC_API_KEY not set!")
    print("Set it with: import os; os.environ['ANTHROPIC_API_KEY'] = 'your-key'")
    print("\nUsing demo bonds instead of live extraction...")

    # Demo bonds for testing without API
    structured_bonds = [
        MoralBond(BondType.OBLIGATION, RoleType.CHILD, RoleType.PARENT,
                  ActionCategory.HONOR, source_tradition="biblical", source_language="hebrew"),
        MoralBond(BondType.OBLIGATION, RoleType.CHILD, RoleType.PARENT,
                  ActionCategory.HONOR, source_tradition="confucian", source_language="classical_chinese"),
        MoralBond(BondType.OBLIGATION, RoleType.CHILD, RoleType.PARENT,
                  ActionCategory.HONOR, source_tradition="dharma", source_language="sanskrit"),
        MoralBond(BondType.OBLIGATION, RoleType.CHILD, RoleType.PARENT,
                  ActionCategory.HONOR, source_tradition="quranic", source_language="arabic"),
        MoralBond(BondType.PROHIBITION, RoleType.AGENT, RoleType.PATIENT,
                  ActionCategory.KILL, source_tradition="biblical", source_language="hebrew"),
        MoralBond(BondType.PROHIBITION, RoleType.AGENT, RoleType.PATIENT,
                  ActionCategory.KILL, source_tradition="dharma", source_language="sanskrit"),
        MoralBond(BondType.PROHIBITION, RoleType.AGENT, RoleType.PATIENT,
                  ActionCategory.KILL, source_tradition="buddhist", source_language="pali"),
        MoralBond(BondType.OBLIGATION, RoleType.PROMISER, RoleType.PROMISEE,
                  ActionCategory.PROMISE, source_tradition="rabbinic", source_language="aramaic"),
        MoralBond(BondType.OBLIGATION, RoleType.DEBTOR, RoleType.CREDITOR,
                  ActionCategory.REPAY, source_tradition="modern_american", source_language="english"),
        MoralBond(BondType.OBLIGATION, RoleType.PARENT, RoleType.CHILD,
                  ActionCategory.NURTURE, source_tradition="biblical", source_language="hebrew"),
    ]
    print(f"\nLoaded {len(structured_bonds)} demo bonds")

else:
    try:
        import anthropic
        client = anthropic.Anthropic()

        print(f"\nConfiguration:")
        print(f"  Sample size: {EXTRACTION_SAMPLE_SIZE}")
        print(f"  Languages: {EXTRACTION_LANGUAGES}")
        print(f"  Model: {EXTRACTION_MODEL}")

        # Check if corpus_cache exists
        if 'corpus_cache' not in dir():
            print("\nWARNING: corpus_cache not found!")
            print("Run Cell 2 first to load the corpus.")
            structured_bonds = []
        else:
            print(f"\nExtracting from corpus_cache...")
            structured_bonds = extract_from_corpus_cache(
                corpus_cache,
                client=client,
                sample_size=EXTRACTION_SAMPLE_SIZE,
                languages=EXTRACTION_LANGUAGES,
                model=EXTRACTION_MODEL,
                delay=0.5,
                verbose=True,
            )

    except ImportError:
        print("\nERROR: anthropic package not installed")
        print("Install with: !pip install anthropic")
        structured_bonds = []

# =============================================================================
# Analysis
# =============================================================================

if structured_bonds:
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # Basic report
    print(generate_analysis_report(structured_bonds))

    # Algebraic analysis
    print()
    print_algebra_report(structured_bonds)

    # Visualizations
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    print("\n## Tradition Similarity Heatmap")
    print(plot_tradition_heatmap_text(structured_bonds))

    print("\n## Bond Space (colored by tradition)")
    print(plot_bond_space_text(structured_bonds, color_by="tradition"))

    # Clustering
    print("\n## Clustering Analysis")
    cluster_result = cluster_bonds(structured_bonds, n_clusters=4)
    if "error" not in cluster_result:
        for cluster_id, stats in cluster_result.get("cluster_stats", {}).items():
            print(f"\nCluster {cluster_id} (n={stats['size']}):")
            if stats["bond_types"]:
                print(f"  Top bond types: {stats['bond_types'][:3]}")
            if stats["traditions"]:
                print(f"  Top traditions: {stats['traditions'][:3]}")

# =============================================================================
# Comparison with Embeddings (if available)
# =============================================================================

if COMPARE_WITH_EMBEDDINGS and structured_bonds:
    print("\n" + "=" * 70)
    print("EMBEDDING vs STRUCTURED COMPARISON")
    print("=" * 70)

    # Check if we have embedding predictions
    if 'model' in dir() and 'test_loader' in dir():
        print("\nComparing with embedding model predictions...")

        # Get embedding predictions
        embedding_predictions = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                bond_preds = outputs['bond_pred'].argmax(dim=-1).cpu().tolist()

                for i, pred in enumerate(bond_preds):
                    embedding_predictions.append({
                        'id': batch.get('id', [None])[i] if 'id' in batch else None,
                        'bond_type': pred,
                        'text': batch.get('text', [''])[i] if 'text' in batch else '',
                    })

        # Compare
        comparison = compare_extraction_methods(structured_bonds, embedding_predictions)
        print_comparison_report(comparison)

    else:
        print("\nNOTE: Model/test_loader not found - skipping embedding comparison.")
        print("Run Cells 6-7 first to train the model.")

# =============================================================================
# Save Results
# =============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

if structured_bonds:
    from src.bip import save_bonds_jsonl

    output_path = Path("data/structured_bonds.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_bonds_jsonl(structured_bonds, output_path)
    print(f"\nSaved {len(structured_bonds)} bonds to {output_path}")

    # Also save to database
    db_path = Path("data/structured_bonds.db")
    db = BondDatabase(db_path)
    db.add_bonds(structured_bonds)
    print(f"Saved to database: {db_path}")
    print(f"Total bonds in database: {db.count()}")

    # Query universal patterns
    universal = db.query_universal(min_traditions=2)
    print(f"\nUniversal patterns (2+ traditions): {len(universal)}")
    for pattern, traditions, count in universal[:5]:
        print(f"  {pattern}")
        print(f"    Traditions: {', '.join(traditions)}")

    db.close()

print("\n" + "=" * 70)
print("STRUCTURED EXTRACTION COMPLETE")
print("=" * 70)
