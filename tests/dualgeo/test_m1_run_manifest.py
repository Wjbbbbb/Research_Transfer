import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qwen_vl.dualgeo.prepare_m1_run import write_manifest


class M1RunManifestTest(unittest.TestCase):
    def test_manifest_is_write_once_and_records_the_split_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split"
            split.mkdir()
            (split / "manifest.json").write_text('{"seed": 0}', encoding="utf-8")
            run = root / "run"
            run.mkdir()
            with patch("qwen_vl.dualgeo.prepare_m1_run.git_revision", return_value="abc123"):
                manifest = write_manifest(
                    run,
                    order="self_then_seek",
                    steps=2,
                    total_batch_size=8,
                    world_size=2,
                    qwen_checkpoint=root / "qwen",
                    vggt_checkpoint=root / "vggt",
                    sidecar_checkpoint=root / "sidecar",
                    split_root=split,
                    repo_root=root,
                )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["git_commit"], "abc123")
            self.assertEqual(payload["stage_max_steps"], 2)
            with self.assertRaises(FileExistsError):
                write_manifest(
                    run,
                    order="self_then_seek",
                    steps=2,
                    total_batch_size=8,
                    world_size=2,
                    qwen_checkpoint=root / "qwen",
                    vggt_checkpoint=root / "vggt",
                    sidecar_checkpoint=root / "sidecar",
                    split_root=split,
                    repo_root=root,
                )
