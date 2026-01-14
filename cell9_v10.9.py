# @title 9. Fuzz Testing & Final Results { display-mode: "form" }
# @markdown v10.9: Structural vs Surface fuzz testing + comprehensive summary
# @markdown Tests if model responds to moral structure (good) vs surface features (bad)

import json
import shutil
from scipy import stats

print("=" * 60)
print("FUZZ TESTING (v10.9)")
print("=" * 60)
print("\nTesting: structural changes should move embeddings,")
print("         surface changes should NOT move embeddings.")


# ===== STRUCTURAL FUZZ TEST CLASS =====
class StructuralFuzzTest:
    """
    Extended fuzz testing with cross-lingual pairs.
    """

    # Structural perturbation templates (language-agnostic concepts)
    STRUCTURAL_PAIRS = {
        "obligation_to_permission": [
            # English
            ("You must help the elderly", "You may help the elderly"),
            ("He is required to pay", "He is allowed to pay"),
            ("Parents must protect children", "Parents may protect children"),
            # Chinese
            ("君子必孝", "君子可孝"),  # Gentleman must/may be filial
            ("民必從法", "民可從法"),  # People must/may follow law
            # Arabic
            ("يجب عليك أن تساعد", "يجوز لك أن تساعد"),  # You must/may help
            # Hebrew
            ("חייב לכבד", "מותר לכבד"),  # Obligated/permitted to honor
        ],
        "harm_to_care": [
            ("He injured the child", "He protected the child"),
            ("殺人者", "救人者"),  # One who kills / one who saves
            ("ظلم الضعيف", "رحم الضعيف"),  # Oppressed / showed mercy to the weak
        ],
        "role_swap": [
            ("The master commands the servant", "The servant commands the master"),
            ("君命臣", "臣命君"),  # Lord commands minister / minister commands lord
            ("الأب يأمر الابن", "الابن يأمر الأب"),  # Father commands son / son commands father
        ],
        "violation_to_fulfillment": [
            ("He broke his promise", "He kept his promise"),
            ("違約", "守約"),  # Violate contract / keep contract
            ("نقض العهد", "وفى بالعهد"),  # Broke covenant / fulfilled covenant
        ],
    }

    # Surface perturbation templates (should NOT move embeddings)
    SURFACE_PERTURBATIONS = {
        "name_change": lambda t: t.replace("John", "Michael").replace("Mary", "Lisa"),
        "irrelevant_detail": lambda t: t + " It was Tuesday.",
        "add_location": lambda t: t + " in the city.",
    }

    def run_comprehensive_test(self, analyzer) -> dict:
        """
        Run full structural vs surface test battery.
        """
        results = {}

        # Test structural perturbations
        for perturbation_type, pairs in self.STRUCTURAL_PAIRS.items():
            distances = []
            for text1, text2 in pairs:
                emb1 = analyzer.get_embedding(text1)
                emb2 = analyzer.get_embedding(text2)
                # Cosine distance
                dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
                distances.append(dist)

            results[f"structural_{perturbation_type}"] = {
                "mean_distance": np.mean(distances),
                "std": np.std(distances),
                "n": len(distances),
            }

        # Surface perturbations on base sentences
        base_sentences = [
            "John borrowed money from Mary and must repay it.",
            "The doctor has a duty to help patients.",
            "Parents should protect their children.",
        ]

        surface_distances = []
        for base in base_sentences:
            base_emb = analyzer.get_embedding(base)
            for name, perturb_fn in self.SURFACE_PERTURBATIONS.items():
                perturbed = perturb_fn(base)
                if perturbed != base:
                    perturbed_emb = analyzer.get_embedding(perturbed)
                    dist = 1 - np.dot(base_emb, perturbed_emb) / (
                        np.linalg.norm(base_emb) * np.linalg.norm(perturbed_emb) + 1e-9
                    )
                    surface_distances.append(dist)

        results["surface_all"] = {
            "mean_distance": np.mean(surface_distances) if surface_distances else 0,
            "std": np.std(surface_distances) if surface_distances else 0,
            "n": len(surface_distances),
        }

        # Statistical comparison
        structural_all = []
        for k, v in results.items():
            if k.startswith("structural_"):
                structural_all.extend([v["mean_distance"]] * v["n"])

        if structural_all and surface_distances:
            t_stat, p_value = stats.ttest_ind(structural_all, surface_distances)
        else:
            t_stat, p_value = 0, 1.0

        results["comparison"] = {
            "structural_mean": np.mean(structural_all) if structural_all else 0,
            "surface_mean": np.mean(surface_distances) if surface_distances else 0,
            "ratio": (
                np.mean(structural_all) / (np.mean(surface_distances) + 1e-9)
                if structural_all
                else 0
            ),
            "t_statistic": t_stat,
            "p_value": p_value,
        }

        return results


# Run fuzz test if model is available
fuzz_results = {}
model_path = f"{SAVE_DIR}/best_mixed_baseline.pt"

if os.path.exists(model_path):
    print("\nLoading model for fuzz testing...")
    model = BIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Reuse GeometricAnalyzer from cell 8
    analyzer = GeometricAnalyzer(model, tokenizer, device)
    fuzz_test = StructuralFuzzTest()

    print("\nRunning structural vs surface comparison...")
    fuzz_results = fuzz_test.run_comprehensive_test(analyzer)

    print("\n--- Structural Perturbations (should be HIGH) ---")
    for k, v in fuzz_results.items():
        if k.startswith("structural_"):
            print(f"  {k}: distance={v['mean_distance']:.4f} +/- {v['std']:.4f} (n={v['n']})")

    print("\n--- Surface Perturbations (should be LOW) ---")
    v = fuzz_results["surface_all"]
    print(f"  surface_all: distance={v['mean_distance']:.4f} +/- {v['std']:.4f} (n={v['n']})")

    print("\n--- Statistical Comparison ---")
    c = fuzz_results["comparison"]
    print(f"  Structural mean: {c['structural_mean']:.4f}")
    print(f"  Surface mean:    {c['surface_mean']:.4f}")
    print(f"  Ratio:           {c['ratio']:.2f}x")
    print(f"  t-statistic:     {c['t_statistic']:.2f}")
    print(f"  p-value:         {c['p_value']:.4f}")

    # Interpret results
    if c["ratio"] > 2.0 and c["p_value"] < 0.05:
        fuzz_status = "EXCELLENT"
        fuzz_msg = "Model strongly distinguishes structural from surface"
    elif c["ratio"] > 1.5:
        fuzz_status = "GOOD"
        fuzz_msg = "Model distinguishes structural from surface"
    elif c["ratio"] > 1.0:
        fuzz_status = "MARGINAL"
        fuzz_msg = "Some structural sensitivity"
    else:
        fuzz_status = "FAILED"
        fuzz_msg = "Model may be using surface features"

    print(f"\n  FUZZ STATUS: {fuzz_status}")
    print(f"  {fuzz_msg}")

    del model
    torch.cuda.empty_cache()
else:
    print(f"\nSkipping fuzz test - no model at {model_path}")
    fuzz_status = "SKIPPED"

print("\n" + "=" * 60)
print("FINAL BIP EVALUATION (v10.9)")
print("=" * 60)

print(f"\nHardware: {GPU_TIER} ({VRAM_GB:.0f}GB VRAM, {RAM_GB:.0f}GB RAM)")

print("\n" + "-" * 60)
print("CROSS-DOMAIN TRANSFER RESULTS")
print("-" * 60)

successful_splits = []
for name, r in all_results.items():
    ratio = r["bond_f1_macro"] / 0.1
    lang_acc = r["language_acc"]

    transfer_ok = ratio > 1.3
    invariant_ok = lang_acc < 0.35  # Near chance (20%)

    status = "SUCCESS" if (transfer_ok and invariant_ok) else "PARTIAL" if transfer_ok else "FAIL"

    print(f"\n{name}:")
    print(
        f"  Bond F1:     {r['bond_f1_macro']:.3f} ({ratio:.1f}x chance) {'OK' if transfer_ok else 'WEAK'}"
    )
    print(f"  Language:    {lang_acc:.1%} {'INVARIANT' if invariant_ok else 'LEAKING'}")
    print(f"  -> {status}")

    if transfer_ok and invariant_ok:
        successful_splits.append(name)

print("\n" + "-" * 60)
print("VERDICT")
print("-" * 60)

n_success = len(successful_splits)
if n_success >= 3:
    verdict = "STRONGLY_SUPPORTED"
    msg = "Multiple independent transfer paths demonstrate universal structure"
elif n_success >= 2:
    verdict = "SUPPORTED"
    msg = "Multiple transfer paths work"
elif n_success >= 1:
    verdict = "PARTIALLY_SUPPORTED"
    msg = "At least one transfer path works"
elif any(r["bond_f1_macro"] > 0.13 for r in all_results.values()):
    verdict = "WEAK"
    msg = "Some transfer signal, but not robust"
else:
    verdict = "INCONCLUSIVE"
    msg = "No clear transfer demonstrated"

print(f"\n  Successful transfers: {n_success}/{len(all_results)}")
print(f"  Splits: {successful_splits if successful_splits else 'None'}")
print(f"\n  VERDICT: {verdict}")
print(f"  {msg}")

# v10.9 specific checks
print("\n" + "-" * 60)
print("v10.9 SPECIFIC CRITERIA")
print("-" * 60)

# Check key v10.9 splits
v109_checks = {
    "confucian_to_buddhist": "Chinese diversity test",
    "all_to_sanskrit": "Sanskrit transfer test",
    "quran_to_fiqh": "Arabic improvement test",
}

for split_name, test_name in v109_checks.items():
    if split_name in all_results:
        r = all_results[split_name]
        f1 = r["bond_f1_macro"]
        threshold = 0.4 if "sanskrit" in split_name else 0.5
        status = "PASS" if f1 >= threshold else "FAIL"
        print(f"  {test_name}: F1={f1:.3f} (need {threshold}) -> {status}")
    else:
        print(f"  {test_name}: NOT RUN")

# Geometry results
if "geometry_results" in dir() and geometry_results:
    print("\n  Geometric Analysis:")
    if "obligation_permission" in geometry_results:
        acc = geometry_results["obligation_permission"].get("transfer_accuracy", 0)
        print(f"    Deontic axis transfer: {acc:.1%} (need 80%)")
    if "pca" in geometry_results:
        n_comp = geometry_results["pca"].get("n_components_90pct", 0)
        print(f"    PCA components for 90%: {n_comp} (need ≤3)")

# Fuzz results
if fuzz_results and "comparison" in fuzz_results:
    print(f"\n  Fuzz Test: {fuzz_status}")
    print(f"    Structural/Surface ratio: {fuzz_results['comparison']['ratio']:.2f}x")

# Save results
final_results = {
    "version": "v10.9",
    "all_results": all_results,
    "probe_results": probe_results if "probe_results" in dir() else {},
    "geometry_results": geometry_results if "geometry_results" in dir() else {},
    "fuzz_results": fuzz_results,
    "successful_splits": successful_splits,
    "verdict": verdict,
    "hardware": {"gpu": GPU_TIER, "vram_gb": VRAM_GB, "ram_gb": RAM_GB},
    "settings": {
        "batch_size": BATCH_SIZE,
        "max_per_lang": MAX_PER_LANG,
        "num_workers": NUM_WORKERS,
    },
    "experiment_time": time.time() - EXPERIMENT_START,
}

with open("results/final_results.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)
shutil.copy("results/final_results.json", f"{SAVE_DIR}/final_results.json")

print(f"\nTotal time: {(time.time() - EXPERIMENT_START)/60:.1f} minutes")
print("Results saved to Drive!")
print("\n" + "=" * 60)
