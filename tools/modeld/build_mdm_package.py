"""Build the pinned MDM v1 NAS package from verified upstream downloads."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

MODEL_SHA = '4fe66440ef38c94ddbe315e14174912975014a96384201a8a632aabbe2302d66'
MODEL_SIZE = 773090994
SOURCE = 'bea3fd4f6e8429bb50e1e0fcb0a202a036c39335'
TINYGRAD = '9d0446a4ba8a532c8b674fb6ad795af015cd9dcf'
SOURCE_TAR_SHA = 'ca794475451a9d4f94b3d34a814efd72b6338fe5b59a2be329a0dbf9df8cc62a'
CHECKPOINT = 'f22e06e5-3b3f-48d2-a6b9-2f32dc20e707/56320/ea1600cb-629e-40a8-ad29-c29b6bfce06e/12864'


def digest(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(model, source_tar, output):
  if model.stat().st_size != MODEL_SIZE or digest(model) != MODEL_SHA:
    raise ValueError('MDM model identity mismatch')
  if digest(source_tar) != SOURCE_TAR_SHA:
    raise ValueError('pinned tinygrad source archive mismatch')
  output.mkdir(parents=True, exist_ok=True)
  runtime = output / 'precompiled-runtime.tar.gz'
  with source_tar.open('rb') as source, tarfile.open(fileobj=source, mode='r:gz') as upstream:
    with runtime.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
      with tarfile.open(fileobj=compressed, mode='w') as archive:
        for member in sorted(upstream.getmembers(), key=lambda m: m.name):
          name = member.name.partition('/')[2]
          if not member.isfile() or not (name.startswith(('tinygrad/', 'examples/openpilot/')) or name == 'LICENSE'):
            continue
          if '..' in Path(name).parts:
            raise ValueError('invalid source archive path')
          data = upstream.extractfile(member).read()
          info = tarfile.TarInfo(name)
          info.size, info.mode, info.mtime = len(data), 0o644, 0
          archive.addfile(info, io.BytesIO(data))
  manifest = {'model_id': 'comma-pr39047-mdm-v1-bea3fd4f-4fe66440', 'filename': 'big_driving_tinygrad.pkl',
              'size': MODEL_SIZE, 'sha256': MODEL_SHA, 'url': 'big_driving_tinygrad.pkl'}
  catalog = {'protocol': 1, 'format': 'comma-generic-onnx', 'serialization': 'persistent-buffer-v1',
             'gpu_arch': 'gfx1200', 'frame_skip': 4, 'camera_resolutions': [[1928, 1208], [1344, 760]],
             'model_sha256': MODEL_SHA, 'model_checkpoint': CHECKPOINT, 'source_commit': SOURCE, 'tinygrad_commit': TINYGRAD,
             'pickle': {'url': manifest['url'], 'size': MODEL_SIZE, 'sha256': MODEL_SHA},
             'runtime': {'url': runtime.name, 'size': runtime.stat().st_size, 'sha256': digest(runtime)}}
  for name, value in [('manifest.json', manifest), ('precompiled.json', catalog)]:
    (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
  shutil.copyfile(model, output / manifest['filename'])
  print(json.dumps(catalog, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('model', type=Path)
  parser.add_argument('source_tar', type=Path)
  parser.add_argument('output', type=Path)
  args = parser.parse_args()
  build(args.model, args.source_tar, args.output)
