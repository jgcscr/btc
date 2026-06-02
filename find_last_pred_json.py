import json

with open('artifacts/tmp/shadow_4h_ultra_conservative_live_replay.log', 'r') as f:
    for line in reversed(f.readlines()):
        if '{' in line and '}' in line and 'predictions' in line:
            try:
                # Try to extract the JSON part
                start = line.find('{')
                end = line.rfind('}') + 1
                data = json.loads(line[start:end])
                if 'predictions' in data and '4h' in data['predictions']:
                    with open('last_valid_log.json', 'w') as out:
                        json.dump(data, out)
                    print("Found valid log with 4h prediction.")
                    exit(0)
            except:
                continue
print("Could not find valid log with 4h prediction.")
