"""Check asset path handling without importing the simulator."""
import ast
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "active_adaptation/envs/objects.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
BUILDERS = ("create_drawer_cabinet_cfg", "create_spring_pedal_cfg", "create_spring_door_cfg")


def resolve_from_builder(name, value):
    function = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "usd_path" for target in node.targets))
    scope = {"Path": Path, "__file__": str(SOURCE), "usd_path": value}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["usd_path"]


class ObjectAssetPaths(unittest.TestCase):
    def test_relative_assets_ignore_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for builder in BUILDERS:
                    with self.subTest(builder=builder):
                        self.assertEqual(resolve_from_builder(builder, "artifacts/custom.usd"),
                                         ROOT / "artifacts/custom.usd")
            finally:
                os.chdir(previous)

    def test_absolute_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "custom.usd"
            for builder in BUILDERS:
                with self.subTest(builder=builder):
                    self.assertEqual(resolve_from_builder(builder, str(absolute)), absolute)

    def test_default_assets_are_repository_relative(self):
        expected = {
            "create_drawer_cabinet_cfg": "drawer_task/assets/drawer_cabinet.usd",
            "create_spring_pedal_cfg": "spring_pedal/assets/spring_pedal.usd",
            "create_spring_door_cfg": "door_task/assets/spring_door.usd",
        }
        for builder, suffix in expected.items():
            with self.subTest(builder=builder):
                self.assertEqual(resolve_from_builder(builder, None), ROOT / "artifacts" / suffix)


if __name__ == "__main__":
    unittest.main()
