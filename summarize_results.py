import json
import glob
import os

results = []
json_files = glob.glob("artifacts/analysis/4h_trade_decision_probe/replay_sweep/*.json")

for f in json_files:
    with open(f, 'r') as jf:
        data = json.load(jf)
        original_prob = data.get("trade_probability") # The script calls it trade_probability in the json
        recon_prob = data.get("reconstructed_probability")
        recon_gap = data.get("reconstructed_threshold_gap")
        
        # Check if it clears threshold based on gap > 0
        clears = recon_gap > 0 if recon_gap is not None else False
        
        results.append({
            "log": os.path.basename(f).replace(".json", ""),
            "original_prob": original_prob,
            "recon_prob": recon_prob,
            "recon_gap": recon_gap,
            "clears": clears
        })

print(f"{'Log Name':<80} | {'Orig Prob':<10} | {'Recon Prob':<10} | {'Gap':<10} | {'Clears'}")
print("-" * 125)
clears_count = 0
for r in results:
    orig = f"{r['original_prob']:.4f}" if r['original_prob'] is not None else "N/A"
    recon = f"{r['recon_prob']:.4f}" if r['recon_prob'] is not None else "N/A"
    gap = f"{r['recon_gap']:.4f}" if r['recon_gap'] is not None else "N/A"
    print(f"{r['log']:<80} | {orig:<10} | {recon:<10} | {gap:<10} | {r['clears']}")
    if r['clears']:
        clears_count += 1

print("-" * 125)
print(f"Total logs clearing threshold: {clears_count}/{len(results)}")
