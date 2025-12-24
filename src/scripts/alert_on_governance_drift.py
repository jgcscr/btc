import os
import json
from pathlib import Path
import smtplib
from email.message import EmailMessage

GOV_DIR = Path('artifacts/analysis/governance')
ALERT_EMAIL = os.environ.get('GOV_ALERT_EMAIL')
THRESHOLDS = {'equity_ensemble_net': 0.01, 'hit_rate': 0.05, 'drawdown': 0.05, 'sharpe_like': 0.1}

def find_latest_diff():
    dated = sorted(GOV_DIR.glob('[0-9]*'), reverse=True)
    for d in dated:
        diff = d / 'walkforward_diff.json'
        if diff.exists():
            with open(diff) as f:
                data = json.load(f)
            return d, data
    return None, None

def alert_message(diff):
    lines = []
    for k, v in diff.items():
        if v.get('flag'):
            lines.append(f"ALERT: {k} deviation {v['delta']:.4f} (current {v['current']}, baseline {v['baseline']})")
    return '\n'.join(lines)

def send_email_alert(body):
    if not ALERT_EMAIL:
        print('No GOV_ALERT_EMAIL set, skipping email.')
        return
    msg = EmailMessage()
    msg['Subject'] = 'Governance Drift Alert'
    msg['From'] = ALERT_EMAIL
    msg['To'] = ALERT_EMAIL
    msg.set_content(body)
    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)

def main():
    d, diff = find_latest_diff()
    if not diff:
        print('No walkforward_diff.json found.')
        return
    body = alert_message(diff)
    if body:
        print(body)
        send_email_alert(body)
    else:
        print('No governance drift detected.')

if __name__ == '__main__':
    main()
