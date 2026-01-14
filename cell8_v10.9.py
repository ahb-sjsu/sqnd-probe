# @title 8. Geometric Analysis & Linear Probe { display-mode: "form" }
# @markdown v10.9: New geometric analysis module + linear probe test
# @markdown Tests latent space structure (axis discovery, role swap analysis)
# @markdown Tests if z_bond encodes language/period (should be low = invariant)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
from typing import List, Dict, Tuple


# ===== v10.9: GEOMETRIC ANALYZER CLASS =====
class GeometricAnalyzer:
    """
    Probe the latent space geometry to discover moral structure.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        z = self.model.get_bond_embedding(inputs["input_ids"], inputs["attention_mask"])
        return z.cpu().numpy().flatten()

    def find_direction(self, positive_texts: List[str], negative_texts: List[str]) -> np.ndarray:
        """
        Find the direction in z-space that separates two concepts.
        E.g., obligation vs permission, harm vs care.
        """
        pos_embs = np.array([self.get_embedding(t) for t in positive_texts])
        neg_embs = np.array([self.get_embedding(t) for t in negative_texts])

        pos_mean = pos_embs.mean(axis=0)
        neg_mean = neg_embs.mean(axis=0)

        direction = pos_mean - neg_mean
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        return direction

    def test_direction_transfer(
        self, direction: np.ndarray, test_pairs: List[Tuple[str, str]]
    ) -> float:
        """
        Test if a direction generalizes to new examples.
        Returns accuracy of direction-based classification.
        """
        scores = []
        for pos_text, neg_text in test_pairs:
            pos_proj = np.dot(self.get_embedding(pos_text), direction)
            neg_proj = np.dot(self.get_embedding(neg_text), direction)
            scores.append(1.0 if pos_proj > neg_proj else 0.0)
        return np.mean(scores)

    def pca_on_pairs(self, concept_pairs: Dict[str, List[Tuple[str, str]]]) -> Dict:
        """
        Run PCA on difference vectors to find dominant axes.

        concept_pairs: {"obligation_permission": [(obl1, perm1), ...], ...}
        """
        all_diffs = []
        labels = []

        for concept, pairs in concept_pairs.items():
            for pos, neg in pairs:
                diff = self.get_embedding(pos) - self.get_embedding(neg)
                all_diffs.append(diff)
                labels.append(concept)

        X = np.array(all_diffs)

        pca = PCA(n_components=min(10, len(X)))
        pca.fit(X)

        return {
            "components": pca.components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "labels": labels,
            "transformed": pca.transform(X),
        }

    def role_swap_analysis(self, agent_patient_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Test if swapping agent/patient produces consistent transformation.

        agent_patient_pairs: [("A harmed B", "B harmed A"), ...]
        """
        transformations = []

        for original, swapped in agent_patient_pairs:
            orig_emb = self.get_embedding(original)
            swap_emb = self.get_embedding(swapped)
            transformations.append(swap_emb - orig_emb)

        T = np.array(transformations)

        # Check consistency: are all transformations similar?
        mean_transform = T.mean(axis=0)
        cosines = [
            np.dot(t, mean_transform) / (np.linalg.norm(t) * np.linalg.norm(mean_transform) + 1e-9)
            for t in T
        ]

        return {
            "mean_transform": mean_transform,
            "consistency": np.mean(cosines),
            "consistency_std": np.std(cosines),
        }


print("=" * 60)
print("LINEAR PROBE TEST")
print("=" * 60)
print("\nIf probe accuracy is NEAR CHANCE, representation is INVARIANT")
print("(This is what we want for BIP)")

probe_results = {}

for split_name in ["hebrew_to_others", "semitic_to_non_semitic"]:
    model_path = f"{SAVE_DIR}/best_{split_name}.pt"
    if not os.path.exists(model_path):
        print(f"\nSkipping {split_name} - no saved model")
        continue

    print(f"\n{'='*50}")
    print(f"PROBE: {split_name}")
    print("=" * 50)

    model = BIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ids = set(all_splits[split_name]["test_ids"][:5000])
    test_dataset = NativeDataset(
        test_ids, "data/processed/passages.jsonl", "data/processed/bonds.jsonl", tokenizer
    )

    if len(test_dataset) < 50:
        print(f"  Skip - only {len(test_dataset)} samples")
        continue

    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn, num_workers=0)

    all_z, all_lang, all_period = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extract"):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), 0)
            all_z.append(out["z"].cpu().numpy())
            all_lang.extend(batch["language_labels"].tolist())
            all_period.extend(batch["period_labels"].tolist())

    X = np.vstack(all_z)
    y_lang = np.array(all_lang)
    y_period = np.array(all_period)

    scaler_probe = StandardScaler()
    X_scaled = scaler_probe.fit_transform(X)

    # Train/test split for probes
    n = len(X)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[: int(0.7 * n)], idx[int(0.7 * n) :]

    # Language probe - check for multiple classes
    unique_langs = np.unique(y_lang[test_idx])
    if len(unique_langs) < 2:
        print(f"  SKIP language probe - only {len(unique_langs)} class")
        lang_acc = 1.0 / max(1, len(np.unique(y_lang)))
        lang_chance = lang_acc
    else:
        lang_probe = LogisticRegression(max_iter=1000, n_jobs=-1)
        lang_probe.fit(X_scaled[train_idx], y_lang[train_idx])
        lang_acc = (lang_probe.predict(X_scaled[test_idx]) == y_lang[test_idx]).mean()
        lang_chance = 1.0 / len(unique_langs)

    # Period probe - same check
    unique_periods = np.unique(y_period[test_idx])
    if len(unique_periods) < 2:
        print(f"  SKIP period probe - only {len(unique_periods)} class")
        period_acc = 1.0 / max(1, len(np.unique(y_period)))
        period_chance = period_acc
    else:
        period_probe = LogisticRegression(max_iter=1000, n_jobs=-1)
        period_probe.fit(X_scaled[train_idx], y_period[train_idx])
        period_acc = (period_probe.predict(X_scaled[test_idx]) == y_period[test_idx]).mean()
        period_chance = 1.0 / len(unique_periods)

    lang_status = "INVARIANT" if lang_acc < lang_chance + 0.15 else "NOT invariant"
    period_status = "INVARIANT" if period_acc < period_chance + 0.15 else "NOT invariant"

    probe_results[split_name] = {
        "language_acc": lang_acc,
        "language_chance": lang_chance,
        "language_status": lang_status,
        "period_acc": period_acc,
        "period_chance": period_chance,
        "period_status": period_status,
    }

    print(f"\nRESULTS:")
    print(f"  Language: {lang_acc:.1%} (chance: {lang_chance:.1%}) -> {lang_status}")
    print(f"  Period:   {period_acc:.1%} (chance: {period_chance:.1%}) -> {period_status}")

    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Probe tests complete")
print("=" * 60)

# ===== v10.9: GEOMETRIC ANALYSIS =====
print("\n" + "=" * 60)
print("GEOMETRIC ANALYSIS (v10.9)")
print("=" * 60)
print("\nDiscovering interpretable axes in latent space...")

# Test pairs for axis discovery (cross-lingual)
OBLIGATION_PERMISSION_TRAIN = [
    # English - training set
    ("You must help the elderly", "You may help the elderly"),
    ("He is required to pay", "He is allowed to pay"),
    ("Parents must protect children", "Parents may protect children"),
]

OBLIGATION_PERMISSION_TEST = [
    # Chinese
    ("君子必孝", "君子可孝"),  # Gentleman must/may be filial
    ("民必從法", "民可從法"),  # People must/may follow law
    # Arabic
    ("يجب عليك أن تساعد", "يجوز لك أن تساعد"),  # You must/may help
    # Hebrew
    ("חייב לכבד", "מותר לכבד"),  # Obligated/permitted to honor
    # English - held out
    ("She must attend", "She may attend"),
]

HARM_CARE_PAIRS = [
    ("He injured the child", "He protected the child"),
    ("殺人者", "救人者"),  # One who kills / one who saves
    ("ظلم الضعيف", "رحم الضعيف"),  # Oppressed / showed mercy to the weak
    ("She hurt the patient", "She healed the patient"),
]

ROLE_SWAP_PAIRS = [
    ("The master commands the servant", "The servant commands the master"),
    ("君命臣", "臣命君"),  # Lord commands minister / minister commands lord
    ("الأب يأمر الابن", "الابن يأمر الأب"),  # Father commands son / son commands father
    ("The parent guides the child", "The child guides the parent"),
]

geometry_results = {}

# Use the best model from mixed_baseline split for geometric analysis
model_path = f"{SAVE_DIR}/best_mixed_baseline.pt"
if os.path.exists(model_path):
    print("\nLoading model for geometric analysis...")
    model = BIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    analyzer = GeometricAnalyzer(model, tokenizer, device)

    # 1. Find obligation/permission axis
    print("\n--- Obligation/Permission Axis ---")
    obl_texts = [p[0] for p in OBLIGATION_PERMISSION_TRAIN]
    perm_texts = [p[1] for p in OBLIGATION_PERMISSION_TRAIN]
    obl_perm_axis = analyzer.find_direction(obl_texts, perm_texts)

    # Test transfer to other languages
    transfer_acc = analyzer.test_direction_transfer(obl_perm_axis, OBLIGATION_PERMISSION_TEST)
    print(f"  Direction found from English training pairs")
    print(f"  Transfer accuracy to other languages: {transfer_acc:.1%}")
    axis_status = "STRONG" if transfer_acc > 0.8 else "WEAK" if transfer_acc > 0.5 else "FAILED"
    print(f"  Status: {axis_status} deontic axis")

    geometry_results["obligation_permission"] = {
        "transfer_accuracy": transfer_acc,
        "status": axis_status,
    }

    # 2. Find harm/care axis
    print("\n--- Harm/Care Axis ---")
    harm_texts = [p[0] for p in HARM_CARE_PAIRS]
    care_texts = [p[1] for p in HARM_CARE_PAIRS]
    harm_care_axis = analyzer.find_direction(harm_texts, care_texts)

    # Check axis orthogonality
    axis_correlation = abs(np.dot(obl_perm_axis, harm_care_axis))
    print(f"  Axis found")
    print(f"  Correlation with obl/perm axis: {axis_correlation:.3f}")
    orthogonal = "ORTHOGONAL" if axis_correlation < 0.3 else "CORRELATED"
    print(f"  Status: {orthogonal}")

    geometry_results["harm_care"] = {
        "axis_correlation": axis_correlation,
        "orthogonal": axis_correlation < 0.3,
    }

    # 3. Role swap analysis
    print("\n--- Role Swap Analysis ---")
    role_analysis = analyzer.role_swap_analysis(ROLE_SWAP_PAIRS)
    print(
        f"  Mean consistency: {role_analysis['consistency']:.3f} +/- {role_analysis['consistency_std']:.3f}"
    )
    role_status = "CONSISTENT" if role_analysis["consistency"] > 0.9 else "VARIABLE"
    print(f"  Status: {role_status} agent/patient transformation")

    geometry_results["role_swap"] = {
        "consistency": role_analysis["consistency"],
        "consistency_std": role_analysis["consistency_std"],
        "status": role_status,
    }

    # 4. PCA on all structural pairs
    print("\n--- PCA Analysis ---")
    all_concept_pairs = {
        "obligation_permission": OBLIGATION_PERMISSION_TRAIN + OBLIGATION_PERMISSION_TEST,
        "harm_care": HARM_CARE_PAIRS,
    }
    pca_results = analyzer.pca_on_pairs(all_concept_pairs)

    cumsum = np.cumsum(pca_results["explained_variance_ratio"])
    n_components_90 = np.argmax(cumsum > 0.9) + 1 if any(cumsum > 0.9) else len(cumsum)

    print(f"  Explained variance ratio: {pca_results['explained_variance_ratio'][:5]}")
    print(f"  Components for 90% variance: {n_components_90}")
    pca_status = "LOW-DIM" if n_components_90 <= 3 else "HIGH-DIM"
    print(f"  Status: {pca_status} moral structure")

    geometry_results["pca"] = {
        "explained_variance": pca_results["explained_variance_ratio"].tolist(),
        "n_components_90pct": n_components_90,
        "status": pca_status,
    }

    del model
    torch.cuda.empty_cache()
else:
    print(f"\nSkipping geometric analysis - no model at {model_path}")
    geometry_results = {"error": "No model available"}

print("\n" + "=" * 60)
print("Geometric analysis complete")
print("=" * 60)
