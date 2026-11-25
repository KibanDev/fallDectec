"""Evaluate both versions on a compact, annotated Le2i Home sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlencode
from zipfile import ZipFile, is_zipfile

from cache_pose import MODEL_PATH, ROOT, SAMPLE_FPS, cache_video, cv2
from train_pose import Clip, LearnedPose, baseline_engine, compare

API = 'https://www.kaggle.com/api/v1/datasets'
DATASET = 'tuyenldvn/falldataset-imvia'


def download(root, entry):
    name, size = entry['name'], entry['totalBytes']
    path = root / name
    if path.is_file() and path.stat().st_size == size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    url = f'{API}/download/{DATASET}/{quote(name, safe="")}'
    if not is_zipfile(temporary):
        subprocess.run(['curl.exe', '-fLsS', '--retry', '2', '--max-time', '120',
                        '-o', str(temporary), url], check=True)
    if is_zipfile(temporary):
        with ZipFile(temporary) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if (len(members) != 1 or members[0].file_size != size
                    or Path(members[0].filename).name != path.name):
                raise ValueError(f'Unexpected archive contents for {name}')
            unpacked = path.with_suffix('.unpacked')
            with archive.open(members[0]) as source, unpacked.open('wb') as target:
                shutil.copyfileobj(source, target)
        unpacked.replace(temporary)
    if temporary.stat().st_size != size:
        raise ValueError(f'Unexpected download size for {name}')
    temporary.replace(path)
    return path


def list_files():
    entries = {}
    params = {'pageSize': 200}
    while True:
        payload = subprocess.check_output(['curl.exe', '-fLsS', '--max-time', '30',
                                           f'{API}/list/{DATASET}?{urlencode(params)}'])
        page = json.loads(payload)
        entries.update((row['name'], row) for row in page['datasetFiles'])
        token = page.get('nextPageTokenNullable')
        if not token:
            return entries
        params['pageToken'] = token


def prepare(root):
    files = list_files()
    annotations = [row for name, row in files.items()
                   if re.fullmatch(r'Home_0[12]/Home_0[12]/Annotation_files/video \(\d+\)\.txt', name)]
    groups = {True: [], False: []}
    for row in annotations:
        label = download(root, row)
        start, end, *_ = label.read_text(encoding='utf-8-sig').splitlines()
        start, end = int(start), int(end)
        if (start, end) != (0, 0) and not 0 < start <= end:
            raise ValueError(f'Invalid Le2i annotation: {label}')
        video = row['name'].replace('/Annotation_files/', '/Videos/').removesuffix('.txt')+'.avi'
        groups[start > 0].append((files[video], start, end))
    chosen = []
    for group in groups.values():
        if len(group) < 6:
            raise ValueError('Need at least six annotated clips for each class')
        chosen.extend(sorted(group, key=lambda item: (item[0]['totalBytes'], item[0]['name']))[:6])
    print(f'Downloading {len(chosen)} clips, {sum(row[0]["totalBytes"] for row in chosen)/1e6:.1f} MB', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda item: download(root, item[0]), chosen))
    manifest = [dict(video=entry['name'], bytes=entry['totalBytes'], start_frame=start, end_frame=end)
                for entry, start, end in chosen]
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = ROOT/'data'/'le2i'
    if args.download:
        prepare(root)
        return

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    detector = YOLO(str(MODEL_PATH))
    model = LearnedPose.load()
    if model is None:
        raise FileNotFoundError('The active learned pose weights are required')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    with TemporaryDirectory(prefix='le2i-poses-') as temporary:
        clips = []
        for index, row in enumerate(manifest):
            video = root/row['video']
            capture = cv2.VideoCapture(str(video))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            finally:
                capture.release()
            if not fps > 0 or row['end_frame'] > frame_count:
                raise ValueError(f'Annotation does not match video: {video}')
            onset = (row['start_frame']-1)/fps if row['start_frame'] else None
            cache = Path(temporary)/f'{index}.json'
            cache.write_text(json.dumps(cache_video(video, detector, args.device)), encoding='utf-8')
            clips.append(Clip(video, cache, 0, onset is not None, onset, ()))
            print(f'Le2i pose {index+1}/{len(manifest)}: {row["video"]}', flush=True)
        summary, recordings = compare(clips, model, baseline_engine(ROOT.parent/'fallDetection_0'))
    report = dict(dataset=DATASET, sample_fps=SAMPLE_FPS, policy=model.policy, selection=manifest,
                  onset_window_seconds=[-.25, 1.5], results=summary, recordings=recordings)
    destination = ROOT/'results'/'le2i_comparison.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith('_posture')}, indent=2))


if __name__ == '__main__':
    main()
