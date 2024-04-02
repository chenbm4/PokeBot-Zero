import json

def load_config(filepath):
    try:
        with open(filepath, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f'Config file not found at {filepath}')
    except json.JSONDecodeError:
        print(f'Error decoding JSON file at {filepath}')
    except Exception as e:
        print(f'Error loading config file: {e}')
    return None
    