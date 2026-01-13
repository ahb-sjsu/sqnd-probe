# Add this after DRIVE_FOLDER parameter in Cell 1

BACKBONE_ADDITION = '''
#@markdown ---
#@markdown ## Model Backbone
BACKBONE = "MiniLM"  #@param ["MiniLM", "LaBSE", "XLM-R-base", "XLM-R-large"]
#@markdown - **MiniLM**: Fast, 118M params, good baseline (paraphrase-multilingual-MiniLM-L12-v2)
#@markdown - **LaBSE**: Best cross-lingual alignment, 471M params (LaBSE)
#@markdown - **XLM-R-base**: Strong multilingual, 270M params (xlm-roberta-base)
#@markdown - **XLM-R-large**: Strongest representations, 550M params (xlm-roberta-large)

# Backbone configurations
BACKBONE_CONFIGS = {
    "MiniLM": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "hidden_size": 384,
        "recommended_batch": {"L4/A100": 512, "T4": 256, "2xT4": 512, "SMALL": 128, "MINIMAL/CPU": 64},
    },
    "LaBSE": {
        "model_name": "sentence-transformers/LaBSE",
        "hidden_size": 768,
        "recommended_batch": {"L4/A100": 256, "T4": 128, "2xT4": 256, "SMALL": 64, "MINIMAL/CPU": 32},
    },
    "XLM-R-base": {
        "model_name": "xlm-roberta-base",
        "hidden_size": 768,
        "recommended_batch": {"L4/A100": 256, "T4": 128, "2xT4": 256, "SMALL": 64, "MINIMAL/CPU": 32},
    },
    "XLM-R-large": {
        "model_name": "xlm-roberta-large",
        "hidden_size": 1024,
        "recommended_batch": {"L4/A100": 128, "T4": 64, "2xT4": 128, "SMALL": 32, "MINIMAL/CPU": 16},
    },
}

BACKBONE_CONFIG = BACKBONE_CONFIGS[BACKBONE]
MODEL_NAME = BACKBONE_CONFIG["model_name"]
BACKBONE_HIDDEN = BACKBONE_CONFIG["hidden_size"]
'''

print("Cell 1 backbone parameter addition ready")
print(BACKBONE_ADDITION)
