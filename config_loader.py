# config_loader.py
# Load YAML configuration files for BIP experiment

import yaml
import re
from pathlib import Path
from enum import Enum, auto
from typing import Dict, List, Any

def load_yaml(filepath: str) -> dict:
    """Load a YAML file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_configs(config_dir: str = "config") -> dict:
    """Load all configuration files."""
    config_dir = Path(config_dir)
    
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        name = yaml_file.stem
        configs[name] = load_yaml(yaml_file)
        print(f"Loaded {name}.yaml")
    
    return configs

# ============================================================
# BOND AND HOHFELD ENUMS
# ============================================================

class BondType(Enum):
    HARM_PREVENTION = auto()
    RECIPROCITY = auto()
    AUTONOMY = auto()
    PROPERTY = auto()
    FAMILY = auto()
    AUTHORITY = auto()
    CARE = auto()
    FAIRNESS = auto()
    CONTRACT = auto()
    NONE = auto()

class HohfeldState(Enum):
    OBLIGATION = auto()
    RIGHT = auto()
    LIBERTY = auto()
    NO_RIGHT = auto()

# ============================================================
# PATTERN LOADER
# ============================================================

def load_patterns(patterns_config: dict) -> tuple:
    """
    Convert YAML patterns to the format expected by the experiment.
    
    Returns:
        (ALL_BOND_PATTERNS, ALL_HOHFELD_PATTERNS) dicts
    """
    ALL_BOND_PATTERNS = {}
    ALL_HOHFELD_PATTERNS = {}
    
    # Load bond patterns
    for lang, bond_dict in patterns_config.get('bond_patterns', {}).items():
        if bond_dict is None:
            continue
        ALL_BOND_PATTERNS[lang] = {}
        for bond_name, patterns in bond_dict.items():
            if patterns is None:
                continue
            try:
                bond_type = BondType[bond_name]
                ALL_BOND_PATTERNS[lang][bond_type] = patterns
            except KeyError:
                print(f"Warning: Unknown bond type {bond_name}")
    
    # Load Hohfeld patterns
    for lang, hohfeld_dict in patterns_config.get('hohfeld_patterns', {}).items():
        if hohfeld_dict is None:
            continue
        ALL_HOHFELD_PATTERNS[lang] = {}
        for state_name, patterns in hohfeld_dict.items():
            if patterns is None:
                continue
            try:
                state = HohfeldState[state_name]
                ALL_HOHFELD_PATTERNS[lang][state] = patterns
            except KeyError:
                print(f"Warning: Unknown Hohfeld state {state_name}")
    
    return ALL_BOND_PATTERNS, ALL_HOHFELD_PATTERNS

# ============================================================
# CORPUS LOADER
# ============================================================

def get_enabled_corpora(corpora_config: dict) -> List[dict]:
    """Get list of enabled corpora."""
    enabled = []
    for name, config in corpora_config.get('corpora', {}).items():
        if config.get('enabled', False):
            config['name'] = name
            enabled.append(config)
    return enabled

def get_period_centuries(corpora_config: dict) -> dict:
    """Get period to century mapping."""
    return corpora_config.get('period_centuries', {})

# ============================================================
# EXPERIMENT CONFIG LOADER
# ============================================================

def get_training_config(experiment_config: dict, gpu_name: str = "T4") -> dict:
    """Get training configuration for specific GPU."""
    training = experiment_config.get('training', {})
    
    # Get batch size for this GPU
    batch_sizes = training.get('batch_sizes', {})
    batch_size = batch_sizes.get(gpu_name, batch_sizes.get('default', 128))
    
    # Get gradient accumulation
    grad_accum = training.get('gradient_accumulation', {})
    grad_steps = grad_accum.get(gpu_name, grad_accum.get('default', 1))
    
    return {
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_steps,
        'learning_rate': training.get('learning_rate', 2e-5),
        'weight_decay': training.get('weight_decay', 0.01),
        'n_epochs': training.get('n_epochs', 5),
        'loss_weights': training.get('loss_weights', {'bond': 1.0, 'language': 0.01, 'period': 0.01}),
        'use_amp': training.get('use_amp', True),
        'adversarial_warmup': training.get('adversarial_warmup', {}),
    }

def get_enabled_splits(experiment_config: dict) -> List[str]:
    """Get list of enabled split names."""
    splits = experiment_config.get('splits', {})
    return [name for name, config in splits.items() if config.get('enabled', False)]

def get_evaluation_thresholds(experiment_config: dict) -> dict:
    """Get evaluation thresholds."""
    return experiment_config.get('evaluation', {})

# ============================================================
# INDEX MAPPINGS
# ============================================================

def create_index_mappings() -> tuple:
    """Create mappings from labels to indices."""
    BOND_TO_IDX = {bt.name: i for i, bt in enumerate(BondType)}
    IDX_TO_BOND = {i: bt.name for i, bt in enumerate(BondType)}
    
    HOHFELD_TO_IDX = {hs.name: i for i, hs in enumerate(HohfeldState)}
    IDX_TO_HOHFELD = {i: hs.name for i, hs in enumerate(HohfeldState)}
    
    # Default language mapping (can be overridden)
    LANG_TO_IDX = {
        'hebrew': 0,
        'aramaic': 1, 
        'classical_chinese': 2,
        'arabic': 3,
        'english': 4,
    }
    IDX_TO_LANG = {i: lang for lang, i in LANG_TO_IDX.items()}
    
    return {
        'BOND_TO_IDX': BOND_TO_IDX,
        'IDX_TO_BOND': IDX_TO_BOND,
        'HOHFELD_TO_IDX': HOHFELD_TO_IDX,
        'IDX_TO_HOHFELD': IDX_TO_HOHFELD,
        'LANG_TO_IDX': LANG_TO_IDX,
        'IDX_TO_LANG': IDX_TO_LANG,
    }

# ============================================================
# QUICK START
# ============================================================

def quick_load(config_dir: str = "config"):
    """
    Quick load all configs and return everything needed.
    
    Usage:
        cfg = quick_load("config")
        ALL_BOND_PATTERNS = cfg['bond_patterns']
        ALL_HOHFELD_PATTERNS = cfg['hohfeld_patterns']
        training_config = cfg['training']
    """
    configs = load_all_configs(config_dir)
    
    # Load patterns
    patterns_config = configs.get('patterns', {})
    ALL_BOND_PATTERNS, ALL_HOHFELD_PATTERNS = load_patterns(patterns_config)
    
    # Load corpora info
    corpora_config = configs.get('corpora', {})
    enabled_corpora = get_enabled_corpora(corpora_config)
    period_centuries = get_period_centuries(corpora_config)
    
    # Load experiment config
    experiment_config = configs.get('experiment', {})
    
    # Create index mappings
    mappings = create_index_mappings()
    
    return {
        'bond_patterns': ALL_BOND_PATTERNS,
        'hohfeld_patterns': ALL_HOHFELD_PATTERNS,
        'enabled_corpora': enabled_corpora,
        'period_centuries': period_centuries,
        'experiment_config': experiment_config,
        'mappings': mappings,
        'raw_configs': configs,
    }


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    # Test loading
    cfg = quick_load("config")
    
    print("\n=== Bond Patterns ===")
    for lang, patterns in cfg['bond_patterns'].items():
        n = sum(len(p) for p in patterns.values())
        print(f"  {lang}: {n} patterns")
    
    print("\n=== Enabled Corpora ===")
    for corpus in cfg['enabled_corpora']:
        print(f"  {corpus['name']}: {corpus.get('languages', [])}")
    
    print("\n=== Training Config ===")
    training = get_training_config(cfg['experiment_config'], "T4")
    for k, v in training.items():
        print(f"  {k}: {v}")
