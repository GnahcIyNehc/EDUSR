"""检查源码语法、来源哈希及可选的原目录完整性，不导入训练依赖。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check source syntax and release integrity; optionally verify original source files.")
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['copied_sha256']:
            errors.append('Release hash mismatch: ' + entry['path'])
        if args.source_root:
            original = args.source_root / entry['path']
            if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != entry['source_sha256']:
                errors.append('Source hash mismatch: ' + entry['path'])
    sources = list(root.rglob('*.py'))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Passed: {len(manifest['files'])} source files; {len(sources)} Python files parsed successfully.")
    if args.source_root:
        print('All listed original files match the recorded source hashes.')


if __name__ == '__main__':
    main()
