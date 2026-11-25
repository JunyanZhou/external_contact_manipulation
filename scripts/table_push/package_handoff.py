"""Create a self-checking complete task handoff from a file mapping JSON."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True,
                   help="JSON list of {source, archive} entries; source may be a directory")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mapping = json.loads(args.manifest.read_text())
    files = {}
    for item in mapping:
        source = Path(item["source"])
        archive = Path(item["archive"])
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file():
                continue
            name = (archive / path.relative_to(source)).as_posix() if source.is_dir() else archive.as_posix()
            if name == "SHA256SUMS" or name in files:
                raise ValueError(f"Duplicate or reserved archive path: {name}")
            files[name] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    checks = []
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            archive.writestr(name, data)
            checks.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        archive.writestr("SHA256SUMS", "\n".join(checks) + "\n")
    temporary.replace(args.output)
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for line in archive.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    print(args.output.resolve(), "files", len(files) + 1, "bytes", args.output.stat().st_size)


if __name__ == "__main__":
    main()
