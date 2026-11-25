"""Package the final drawer-pull experiment and its reproduction inputs."""

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/drawer_task/drawer_pull_teacher_student_baseline_complete.zip"
PREFIX = "drawer_pull_complete"


def add_tree(files: set[Path], relative: str, suffixes=None):
    root = ROOT / relative
    if not root.exists():
        raise FileNotFoundError(root)
    for path in root.rglob("*"):
        if path.is_file() and (suffixes is None or path.suffix in suffixes):
            files.add(path)


def main():
    files: set[Path] = set()
    for relative in (
        "docs/drawer_pull_external_support_teacher_student.md",
        "setup.py",
        "README.md",
        "artifacts/table_push/static_poses_hand08.npz",
        "scripts/train.py",
        "scripts/table_push/eval_task_policy.py",
        "scripts/table_push/check_five_target_pair.py",
        "artifacts/drawer_task/assets/drawer_cabinet_v7.usd",
        "artifacts/drawer_task/checkpoints/teacher_selected.pt",
        "artifacts/drawer_task/checkpoints/student_adapt_final.pt",
        "artifacts/drawer_task/checkpoints/student_selected.pt",
        "artifacts/door_task/checkpoints/ground_only_reach_baseline.pt",
        "artifacts/drawer_task/teacher_final_train.log",
        "artifacts/drawer_task/teacher_pull_finetune.log",
        "artifacts/drawer_task/student_adapt.log",
        "artifacts/drawer_task/student_finetune.log",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)

    for relative in (
        "artifacts/drawer_task/references_v12",
        "artifacts/drawer_task/references_feasible",
        "artifacts/drawer_task/waypoints",
        "artifacts/drawer_task/config_snapshots",
        "dataset/drawer_pull_v10",
        "dataset/drawer_pull_feasible",
        "artifacts/drawer_task/videos",
        "cfg",
    ):
        if relative.endswith("/videos"):
            continue
        add_tree(files, relative)
    add_tree(files, "active_adaptation", {".py", ".usd"})
    add_tree(files, "scripts", {".py"})

    for kind in ("teacher", "student", "baseline"):
        for i in range(4):
            stem = f"artifacts/drawer_task/{kind}_selected_val{i:02d}_final"
            for suffix in (".json", ".log"):
                path = ROOT / f"{stem}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)
                files.add(path)
    for i in range(4):
        suffix = "" if i == 0 else f"_val{i:02d}"
        path = ROOT / f"artifacts/drawer_task/student_baseline_final_pair{suffix}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
    for name in (
        "teacher_drawer_pull_1080p.mp4",
        "student_drawer_pull_1080p.mp4",
        "ground_only_baseline_same_goals_1080p.mp4",
        "student_vs_ground_only_baseline_3840x1080.mp4",
    ):
        path = ROOT / "artifacts/drawer_task/videos" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)

    manifest = {"task": "drawer_pull", "files": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", allowZip64=True) as archive:
        for path in sorted(files):
            rel = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"][rel] = {"sha256": digest, "bytes": path.stat().st_size}
            compression = zipfile.ZIP_STORED if path.suffix in {".pt", ".mp4", ".npz", ".usd"} else zipfile.ZIP_DEFLATED
            archive.write(path, f"{PREFIX}/{rel}", compress_type=compression)
        archive.writestr(f"{PREFIX}/MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(OUT) as archive:
        error = archive.testzip()
        if error is not None:
            raise RuntimeError(f"Archive CRC failure: {error}")
    print(f"{OUT}: {len(files)} files, {OUT.stat().st_size / 2**20:.1f} MiB; verified")


if __name__ == "__main__":
    main()
