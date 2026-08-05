"""Tests for the web UI (FastAPI TestClient, heavy deps mocked).

Run: .venv/Scripts/python.exe -m unittest tests.test_web -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from master_agent.web.app import app
from master_agent.web.jobs import MANAGER, Job


class TestStudioHtml(unittest.TestCase):
    def test_voice_chat_ui_present(self):
        from master_agent.web.app import STATIC_DIR

        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-tab="voice"', html)
        self.assertIn("SpeechRecognition", html)
        self.assertIn("webkitSpeechRecognition", html)
        self.assertIn("speechSynthesis", html)
        self.assertIn("c-intake-mic", html)
        self.assertIn("v-start", html)
        self.assertIn("Hands-free", html)


class TestHealth(unittest.TestCase):
    def test_health_shape(self):
        with patch("master_agent.comfy.client.ComfyClient.health") as h, patch(
            "master_agent.web.app.collection_count" if hasattr(app, "collection_count") else "master_agent.kb.store.collection_count",
            return_value=3,
        ), patch("master_agent.llm.provider_available", return_value=True):
            h.return_value = {"devices": [{"name": "gpu0", "vram_free": 16e9, "vram_total": 17e9}]}
            client = TestClient(app)
            r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["comfyui"]["up"])
        self.assertEqual(data["comfyui"]["gpus"][0]["vram_free_gb"], 16.0)
        self.assertTrue(data["ollama"])


class TestJobs(unittest.TestCase):
    def setUp(self):
        MANAGER._jobs.clear()

    def test_validation(self):
        client = TestClient(app)
        r = client.post("/api/jobs", json={"request": "  "})
        self.assertEqual(r.status_code, 400)
        r = client.post("/api/jobs", json={"request": "x", "quality": "nope"})
        self.assertEqual(r.status_code, 400)

    def test_rejects_media_outside_uploads(self):
        client = TestClient(app)
        r = client.post(
            "/api/jobs",
            json={"request": "rain", "image_path": "C:/Windows/notepad.exe"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("uploads", (r.json().get("detail") or "").lower())

    def test_accepts_media_under_uploads(self):
        from master_agent.config import STATE_DIR

        uploads = STATE_DIR / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        img = uploads / "test_ui_img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        try:
            with patch(
                "master_agent.orchestrator.pipeline._plan_storyboard",
                return_value=([], "style", {}),
            ):
                client = TestClient(app)
                r = client.post(
                    "/api/jobs",
                    json={
                        "request": "rain on neon",
                        "dry_run": True,
                        "image_path": str(img),
                        "llm_panel": "local",
                        "storyboard": "always",
                    },
                )
            self.assertEqual(r.status_code, 200, r.text)
            job = MANAGER.get(r.json()["id"])
            self.assertEqual(job.params.get("image_path"), str(img.resolve()))
            self.assertEqual(job.params.get("storyboard"), "always")
        finally:
            img.unlink(missing_ok=True)

    def test_upload_endpoint(self):
        client = TestClient(app)
        r = client.post(
            "/api/upload",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["path"])
        self.assertEqual(data["name"], "clip.wav")
        self.assertGreater(data["size_bytes"], 0)
        Path(data["path"]).unlink(missing_ok=True)

    def test_dry_run_job_lifecycle(self):
        from master_agent.storyboard.storyboard import ShotCard

        cards = [ShotCard(index=0, title="Hook", duration_s=4.0, ltx_prompt="p")]
        with patch(
            "master_agent.orchestrator.pipeline._plan_storyboard",
            return_value=(cards, "style x", {"winner": "ollama"}),
        ):
            client = TestClient(app)
            r = client.post("/api/jobs", json={"request": "coffee ad", "dry_run": True})
            self.assertEqual(r.status_code, 200)
            job_id = r.json()["id"]
            job = MANAGER.get(job_id)
            for _ in range(100):
                if job.status in ("done", "error"):
                    break
                import time

                time.sleep(0.05)
            detail = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(detail["status"], "done", detail.get("error"))
        self.assertEqual(detail["result"]["shots"][0]["title"], "Hook")
        self.assertEqual(detail["result"]["global_style"], "style x")

    def test_run_job_lifecycle(self):
        fake = MagicMock()
        fake.run_id = "x"
        fake.to_dict.return_value = {
            "run_id": "x",
            "status": "done",
            "video_path": "outputs/x/y.mp4",
            "full_judge_score": 0.9,
        }
        fake.status = "done"
        fake.error = None
        with patch("master_agent.orchestrator.pipeline.run_pipeline", return_value=fake), patch(
            "master_agent.comfy.client.ComfyClient"
        ) as mock_client:
            mock_client.return_value.upload_image.return_value = "start.png"
            mock_client.return_value.upload_audio.return_value = "vo.wav"
            client = TestClient(app)
            r = client.post("/api/jobs", json={"request": "rain", "duration_s": 3})
            job_id = r.json()["id"]
            job = MANAGER.get(job_id)
            for _ in range(100):
                if job.status in ("done", "error"):
                    break
                import time

                time.sleep(0.05)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["full_judge_score"], 0.9)

    def test_run_job_uploads_media_to_comfy(self):
        from master_agent.config import STATE_DIR

        uploads = STATE_DIR / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        img = uploads / "ui_start.png"
        aud = uploads / "ui_vo.wav"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        aud.write_bytes(b"RIFF")
        fake = MagicMock()
        fake.run_id = "m1"
        fake.to_dict.return_value = {
            "run_id": "m1",
            "status": "done",
            "video_path": str(STATE_DIR.parent / "outputs" / "m1.mp4"),
            "full_judge_score": 0.8,
        }
        fake.status = "done"
        fake.error = None
        try:
            with patch("master_agent.orchestrator.pipeline.run_pipeline", return_value=fake) as rp, patch(
                "master_agent.comfy.client.ComfyClient"
            ) as mock_client:
                inst = mock_client.return_value
                inst.upload_image.return_value = "start.png"
                inst.upload_audio.return_value = "vo.wav"
                client = TestClient(app)
                r = client.post(
                    "/api/jobs",
                    json={
                        "request": "talking head",
                        "image_path": str(img),
                        "audio_path": str(aud),
                        "variant": "lipsync",
                    },
                )
                self.assertEqual(r.status_code, 200, r.text)
                job = MANAGER.get(r.json()["id"])
                for _ in range(100):
                    if job.status in ("done", "error"):
                        break
                    import time

                    time.sleep(0.05)
            self.assertEqual(job.status, "done", job.error)
            kwargs = rp.call_args.kwargs
            self.assertEqual(kwargs.get("image_name"), "start.png")
            self.assertEqual(kwargs.get("audio_name"), "vo.wav")
            self.assertEqual(kwargs.get("variant"), "lipsync")
            self.assertEqual(job.result.get("media", {}).get("image_name"), "start.png")
        finally:
            img.unlink(missing_ok=True)
            aud.unlink(missing_ok=True)


class TestRunsAndGuards(unittest.TestCase):
    def test_runs_lists_records(self):
        with TemporaryDirectory() as td:
            rec = Path(td) / "20260730T000000Z_abc123_pipeline.json"
            rec.write_text(
                json.dumps(
                    {
                        "run_id": "abc123",
                        "request": "coffee ad",
                        "segment_paths": ["a.mp4"],
                        "full_judge_score": 0.88,
                        "full_judge_pass": True,
                        "status": "done",
                    }
                ),
                encoding="utf-8",
            )
            with patch("master_agent.web.app.RUNS_DIR", Path(td)):
                client = TestClient(app)
                r = client.get("/api/runs")
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request"], "coffee ad")
        self.assertEqual(rows[0]["kind"], "pipeline")
        self.assertTrue(rows[0]["passed"])

    def test_videos_path_guard(self):
        client = TestClient(app)
        r = client.get("/videos/..%2F..%2F.env")
        self.assertIn(r.status_code, (404, 422))

    def test_kb_search_shape(self):
        with patch(
            "master_agent.kb.store.search",
            return_value=[{"id": "run:x", "text": "t", "metadata": {}, "distance": 0.1}],
        ):
            client = TestClient(app)
            r = client.get("/api/kb/search?q=coffee&k=1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()[0]["id"], "run:x")


if __name__ == "__main__":
    unittest.main()
