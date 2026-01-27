# BIP Temporal Invariance Experiment
# Bond Invariance Principle Testing

# Core moral structure extraction
from .moral_structure import (
    MoralBond,
    BondType,
    RoleType,
    ActionCategory,
    ContextType,
    ModalStrength,
    extract_bonds_sync,
    find_universal_patterns,
    compute_symmetry_group,
    compute_bond_algebra,
    find_cultural_variations,
    generate_analysis_report,
)

# Corpus extraction
from .corpus_extraction import (
    CorpusPassage,
    extract_from_corpus_cache,
    extract_from_passage,
    save_bonds_jsonl,
    load_bonds_jsonl,
    compute_extraction_stats,
    print_extraction_stats,
)

# Validation metrics
from .extraction_metrics import (
    GoldStandardEntry,
    ExtractionMetrics,
    load_gold_standard,
    compute_extraction_metrics,
    print_metrics_report,
    evaluate_on_gold_standard,
)

# Bond database
from .bond_database import (
    BondDatabase,
    print_database_stats,
)

# Mathematical analysis
from .bond_algebra import (
    compute_role_transition_matrix,
    compute_action_bond_matrix,
    compute_tradition_vectors,
    compute_tradition_similarity,
    compute_bond_composition,
    compute_symmetry_structure,
    compute_group_structure,
    find_moral_isomorphisms,
    print_algebra_report,
)

# Visualization
from .bond_visualization import (
    cluster_bonds,
    reduce_dimensions,
    plot_tradition_heatmap_text,
    plot_cluster_summary_text,
    plot_bond_space_text,
)

# Method comparison
from .method_comparison import (
    ComparisonResult,
    compare_extraction_methods,
    print_comparison_report,
    analyze_systematic_disagreements,
)

__all__ = [
    # Core types
    "MoralBond",
    "BondType",
    "RoleType",
    "ActionCategory",
    "ContextType",
    "ModalStrength",
    # Extraction
    "extract_bonds_sync",
    "CorpusPassage",
    "extract_from_corpus_cache",
    "extract_from_passage",
    "save_bonds_jsonl",
    "load_bonds_jsonl",
    # Analysis
    "find_universal_patterns",
    "compute_symmetry_group",
    "compute_bond_algebra",
    "find_cultural_variations",
    "generate_analysis_report",
    # Metrics
    "GoldStandardEntry",
    "ExtractionMetrics",
    "load_gold_standard",
    "compute_extraction_metrics",
    "print_metrics_report",
    "evaluate_on_gold_standard",
    # Database
    "BondDatabase",
    "print_database_stats",
    # Math
    "compute_role_transition_matrix",
    "compute_action_bond_matrix",
    "compute_tradition_vectors",
    "compute_tradition_similarity",
    "compute_group_structure",
    "find_moral_isomorphisms",
    "print_algebra_report",
    # Visualization
    "cluster_bonds",
    "reduce_dimensions",
    "plot_tradition_heatmap_text",
    "plot_bond_space_text",
    # Comparison
    "ComparisonResult",
    "compare_extraction_methods",
    "print_comparison_report",
]
