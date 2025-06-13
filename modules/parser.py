import pandas as pd
import json
import os
import zipfile

def load_dataset(file):
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file), {}
        elif ext == ".json":
            data = json.load(file)
            return pd.json_normalize(data), {}
        elif ext == ".jsonl":
            lines = file.read().decode("utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            return pd.DataFrame(records), {}
        elif ext == ".xml":
            import xml.etree.ElementTree as ET
            tree = ET.parse(file)
            root = tree.getroot()
            data = [{elem.tag: elem.text for elem in item} for item in root]
            return pd.DataFrame(data), {}
        elif ext == ".zip":
            with zipfile.ZipFile(file, 'r') as z:
                metadata = {}
                for name in z.namelist():
                    if name.endswith('.json') or name.endswith('.csv'):
                        with z.open(name) as f:
                            if name.endswith('.json'):
                                data = json.load(f)
                                return pd.json_normalize(data), metadata
                            else:
                                return pd.read_csv(f), metadata
        return None, {}
    except Exception as e:
        return None, {"error": str(e)}
