import json
import yaml
import math
import os

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def apply_platt(raw_p, a, b):
    # Platt scaling: sigmoid(a * logit(raw_p) + b)
    # Clip raw_p to avoid log(0)
    raw_p = max(min(raw_p, 0.9999), 0.0001)
    logit = math.log(raw_p / (1 - raw_p))
    return sigmoid(a * logit + b)

def get_calibration_key(regime_state):
    return f"{regime_state['primary_regime']}_{regime_state['secondary_regime']}"

def main():
    try:
        with open('last_valid_log.json', 'r') as f:
            last_log = json.load(f)
    except Exception as e:
        # Fallback: try to read line by line if grep failed to get a single object
        print(f"Error loading log: {e}")
        return

    pred_4h = last_log.get('predictions', {}).get('4h')
    if not pred_4h:
        print("No 4h prediction found in log.")
        return

    raw_p_up = pred_4h['raw_p_up']
    regime_state = pred_4h['regime_state']
    
    with open('configs/run_refresh_and_predict.shadow_4h_ultra_conservative.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('artifacts/models_4h_candidate_ultra_conservative/platt_calibration_4h_candidate.json', 'r') as f:
        calibration_map = json.load(f)
    
    cal_key = get_calibration_key(regime_state)
    cal_params = calibration_map.get(cal_key)
    if not cal_params:
        # Fallback logic
        cal_params = calibration_map.get('default', list(calibration_map.values())[0])

    recomputed_p_up = apply_platt(raw_p_up, cal_params['a'], cal_params['b'])
    logged_p_up = pred_4h['p_up']
    alignment_gap = abs(recomputed_p_up - logged_p_up)
    
    # Trust hardening check
    # Check if we have trust_hardening_policy in config
    policy = config.get('trust_hardening_policy', {})
    
    # Evaluate divergence
    suspicious_div = alignment_gap > 0.01
    
    trust_status = "TRUSTED"
    trust_reasons = []
    if suspicious_div:
        trust_status = "LOW_TRUST"
        trust_reasons.append(f"Calibration divergence: {alignment_gap:.6f}")
    
    results = {
        "raw_p_up": raw_p_up,
        "logged_p_up": logged_p_up,
        "recomputed_p_up": recomputed_p_up,
        "calibration_key_used": cal_key,
        "probability_alignment_gap": alignment_gap,
        "metadata_reasons": [],
        "suspicious_calibration_divergence": suspicious_div,
        "final_trust_reasons": trust_reasons,
        "final_trust_status": trust_status,
        "indicates_clears_runtime_block": trust_status == "TRUSTED"
    }
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
