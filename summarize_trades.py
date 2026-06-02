import json
import os

def get_predictions_summary():
    preds_path = 'artifacts/predictions/latest.json'
    if not os.path.exists(preds_path):
        return "latest.json not found"
    
    with open(preds_path, 'r') as f:
        data = json.load(f)
    
    horizons = ['15m', '1h', '4h', '8h', '12h']
    summary = {}
    
    predictions = data.get('predictions', {})
    for h in horizons:
        p = predictions.get(h, {})
        summary[h] = {
            "timestamp": p.get("timestamp"),
            "close": p.get("close"),
            "entry_price": p.get("entry_price"),
            "direction_next": p.get("direction_next"),
            "trade_action": p.get("trade_action"),
            "p_up": round(p.get("p_up", 0), 4) if p.get("p_up") is not None else None,
            "confidence_score": round(p.get("confidence_score", 0), 4) if p.get("confidence_score") is not None else None,
            "projected_price": round(p.get("projected_price", 0), 2) if p.get("projected_price") is not None else None,
            "stop_loss": p.get("stop_loss"),
            "take_profit": p.get("take_profit"),
            "regime_state": p.get("regime_state"),
            "expected_value": round(p.get("expected_value", 0), 6) if p.get("expected_value") is not None else None,
            "gates": p.get("gates", {})
        }
    return summary

def get_trade_ready_summary():
    summary_path = 'artifacts/monitoring/trade_ready_summary.json'
    if not os.path.exists(summary_path):
        return "trade_ready_summary.json not found"
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    prompt_ready = data.get('prompt_ready_summary', {})
    mos = prompt_ready.get('market_outlook_strategy', {})
    tep = prompt_ready.get('trade_execution_plan_usd', {})
    ans = prompt_ready.get('analysis_summary', {})
    
    return {
        "selected_direction": mos.get("selected_direction"),
        "preferred_horizon": mos.get("preferred_horizon"),
        "confidence": mos.get("confidence_level"),
        "entry": tep.get("entry_point"),
        "stop": tep.get("stop_loss"),
        "take_profit": tep.get("take_profit"),
        "risk_reward": round(tep.get("risk_reward_ratio", 0), 2) if tep.get("risk_reward_ratio") is not None else None,
        "blockers": ans.get("blocking_factors", [])
    }

preds = get_predictions_summary()
trade_ready = get_trade_ready_summary()

output = {
    "horizons": preds,
    "top_level_summary": trade_ready
}

print(json.dumps(output, indent=2))
