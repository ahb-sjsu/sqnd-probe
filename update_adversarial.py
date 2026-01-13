import json

nb = json.load(open('BIP_v10.5_expanded.ipynb', encoding='utf-8'))
cell7 = ''.join(nb['cells'][7]['source'])

# Update hyperparameters
replacements = [
    # Increase adversarial weights significantly
    ('LANG_WEIGHT = 0.01  #@param {type:"number"}',
     'LANG_WEIGHT = 1.0  #@param {type:"number"}'),
    ('PERIOD_WEIGHT = 0.01  #@param {type:"number"}',
     'PERIOD_WEIGHT = 0.5  #@param {type:"number"}'),
    # More epochs for adversarial training to converge
    ('N_EPOCHS = 5  #@param {type:"integer"}',
     'N_EPOCHS = 10  #@param {type:"integer"}'),
]

# Update get_adv_lambda to ramp higher
old_adv_lambda = '''def get_adv_lambda(epoch, warmup=2):
        if epoch <= warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        return 1.0'''

new_adv_lambda = '''def get_adv_lambda(epoch, warmup=3):
        """Ramp adversarial strength: 0.1 -> 2.0 over warmup, then hold at 2.0"""
        if epoch <= warmup:
            return 0.1 + 1.9 * (epoch / warmup)
        return 2.0'''

replacements.append((old_adv_lambda, new_adv_lambda))

for old, new in replacements:
    if old in cell7:
        cell7 = cell7.replace(old, new)
        print(f"Replaced: {old[:50]}...")
    else:
        print(f"NOT FOUND: {old[:50]}...")

# Save back
lines = cell7.split('\n')
nb['cells'][7]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('BIP_v10.5_expanded.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n=== Updated hyperparameters ===")
print("LANG_WEIGHT: 0.01 -> 1.0 (100x increase)")
print("PERIOD_WEIGHT: 0.01 -> 0.5 (50x increase)")
print("N_EPOCHS: 5 -> 10 (2x more training)")
print("adv_lambda max: 1.0 -> 2.0 (stronger gradient reversal)")
print("\nDone!")
