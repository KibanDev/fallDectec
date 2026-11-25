"""Download and verify the annotated GMDCSA24 v2.1 RGB recordings."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1] / 'data' / 'gmdcsa'
REPO = 'ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos'
REVISION = 'v2.1'


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == entry['sha']


def download(entry):
    path = ROOT / entry['path']
    if matches(path, entry):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part')
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{quote(entry['path'])}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response, partial.open('wb') as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
            if not matches(partial, entry):
                raise OSError(f"Content failed verification: {entry['path']}")
            partial.replace(path)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    index = ROOT / 'tree.json'
    if not index.exists():
        url = f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1'
        with urlopen(url, timeout=30) as response:
            index.write_bytes(response.read())
    entries = [entry for entry in json.loads(index.read_text(encoding='utf-8'))['tree']
               if entry['path'].startswith('Subject ') and entry['path'].endswith(('.csv', '.mp4'))]
    entries.sort(key=lambda row: (not row['path'].endswith('.csv'), row['path']))
    if len(entries) != 168:
        raise ValueError(f'Expected 160 videos and 8 label files, found {len(entries)} files')
    with ThreadPoolExecutor(max_workers=10) as pool:
        for count, _ in enumerate(pool.map(download, entries), 1):
            if count % 10 == 0 or count == len(entries):
                print(f'GMDCSA24 verified {count}/{len(entries)}', flush=True)


if __name__ == '__main__':
    main()
