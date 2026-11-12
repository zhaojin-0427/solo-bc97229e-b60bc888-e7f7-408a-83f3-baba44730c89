from __future__ import annotations

import gzip
import importlib
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

try:
    import flask  # noqa: F401
except ImportError:  # pragma: no cover - core tests do not need Flask
    flask = None

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@unittest.skipIf(flask is None, "Flask is not installed; skip web integration tests")
class WebIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data_dir = self.root / "data"
        self._old_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = str(self.data_dir)
        import app as app_module

        importlib.reload(app_module)
        self.app_module = app_module
        self.app = app_module.create_app(
            {
                "TESTING": True,
                "DATA_DIR": self.data_dir,
                "JOB_WORKERS": 1,
                "MAX_UPLOAD_BYTES": 64 * 1024 * 1024,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if self._old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old_data_dir
        self._tmp.cleanup()

    def _upload(self, sample_id: str, example: str):
        base = EXAMPLES / example
        with open(base / "R1.fastq.gz", "rb") as r1, open(
            base / "R2.fastq.gz", "rb"
        ) as r2:
            return self.client.post(
                "/samples",
                data={
                    "sample_id": sample_id,
                    "r1": (r1, "R1.fastq.gz"),
                    "r2": (r2, "R2.fastq.gz"),
                },
                content_type="multipart/form-data",
                headers={"Accept": "application/json"},
            )

    def _wait(self, job_id: str, timeout: float = 20.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            job = response.get_json()
            if job["status"] in ("succeeded", "failed"):
                return job
            time.sleep(0.05)
        self.fail(f"job {job_id} did not finish in time")

    def _upload_and_wait(self, sample_id: str, example: str = "valid-disordered") -> dict:
        response = self._upload(sample_id, example)
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        job = self._wait(response.get_json()["id"])
        self.assertEqual(job["status"], "succeeded", job)
        return job

    def _release_dir(self, sample_id: str, release_id: str) -> Path:
        return self.data_dir / "packages" / sample_id / "releases" / release_id

    def test_upload_disordered_sample_and_download_package(self) -> None:
        response = self._upload("DEMO-ORDERED", "valid-disordered")
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        job_id = response.get_json()["id"]
        job = self._wait(job_id)
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["r1_reads"], 4)
        self.assertEqual(job["r2_reads"], 4)
        self.assertIsNotNone(job["current_package"])

        download = self.client.get("/samples/DEMO-ORDERED/download")
        self.assertEqual(download.status_code, 200)
        payload = download.data
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            r1_bytes = archive.read("DEMO-ORDERED_R1.fastq.gz")
            r2_bytes = archive.read("DEMO-ORDERED_R2.fastq.gz")
        with gzip.open(io.BytesIO(r1_bytes), "rb") as gz:
            headers = [
                line.decode().split()[0]
                for line in gz
                if line.startswith(b"@")
            ]
        with gzip.open(io.BytesIO(r2_bytes), "rb") as gz:
            headers2 = [
                line.decode().split()[0]
                for line in gz
                if line.startswith(b"@")
            ]
        self.assertEqual(
            headers,
            ["@DEMO-03/1", "@DEMO-01/1", "@DEMO-04/1", "@DEMO-02/1"],
        )
        self.assertEqual(
            headers2,
            ["@DEMO-03/2", "@DEMO-01/2", "@DEMO-04/2", "@DEMO-02/2"],
        )

    def test_reupload_failure_keeps_existing_package_downloadable(self) -> None:
        first = self._upload("DEMO-KEEP", "valid-disordered")
        self.assertEqual(first.status_code, 202)
        first_job = self._wait(first.get_json()["id"])
        self.assertEqual(first_job["status"], "succeeded")
        release_id = first_job["current_package"]["release_id"]
        before = self.client.get("/samples/DEMO-KEEP/download").data

        second = self._upload("DEMO-KEEP", "tail-corrupt-gzip")
        self.assertEqual(second.status_code, 202)
        second_job = self._wait(second.get_json()["id"])
        self.assertEqual(second_job["status"], "failed", second_job)
        self.assertIn("gzip", second_job["error"])
        self.assertEqual(
            second_job["current_package"]["release_id"], release_id
        )

        after = self.client.get("/samples/DEMO-KEEP/download")
        self.assertEqual(after.status_code, 200)
        self.assertEqual(after.data, before)

    def test_invalid_sample_id_and_missing_files_are_rejected(self) -> None:
        bad = self.client.post(
            "/samples",
            data={"sample_id": "../bad"},
            headers={"Accept": "application/json"},
            content_type="multipart/form-data",
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("样本号无效", bad.get_json()["error"])

        empty = self.client.post(
            "/samples",
            data={"sample_id": "OK"},
            headers={"Accept": "application/json"},
            content_type="multipart/form-data",
        )
        self.assertEqual(empty.status_code, 400)

    def test_open_download_survives_republish_and_prune(self) -> None:
        first = self._upload_and_wait("DEMO-STREAM")
        release_a = first["current_package"]["release_id"]
        dir_a = self._release_dir("DEMO-STREAM", release_a)
        expected_response = self.client.get("/samples/DEMO-STREAM/download")
        expected_zip = expected_response.data
        expected_response.close()

        # Start a download of version A and keep the response open: the
        # request has selected A but the transfer is still in flight.
        stream = self.client.get("/samples/DEMO-STREAM/download", buffered=False)
        self.assertEqual(stream.status_code, 200)
        try:
            # Two more publishes push A beyond KEEP_RELEASES=2.  Publishing
            # the same sample must keep working during the slow download.
            second = self._upload_and_wait("DEMO-STREAM")
            third = self._upload_and_wait("DEMO-STREAM")
            release_c = third["current_package"]["release_id"]
            self.assertNotEqual(release_a, release_c)

            # Cleanup of A is deferred while the download holds its lease.
            self.assertTrue(dir_a.is_dir())

            # Other samples stay operable during the slow download.
            self._upload_and_wait("DEMO-OTHER")

            # A brand-new request is served the new current version C.
            fresh = self.client.get("/samples/DEMO-STREAM/download")
            with zipfile.ZipFile(io.BytesIO(fresh.data)) as archive:
                fresh_manifest = json.loads(archive.read("manifest.json"))
            fresh.close()
            self.assertEqual(fresh_manifest["release_id"], release_c)

            # The in-flight request still streams exactly version A.
            body = b"".join(stream.iter_encoded())
        finally:
            stream.close()

        self.assertEqual(body, expected_zip)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["release_id"], release_a)

        # Once the download is done, deferred cleanup reclaims A while the
        # two newest releases stay available.
        self.assertFalse(dir_a.exists())
        release_b = second["current_package"]["release_id"]
        self.assertTrue(self._release_dir("DEMO-STREAM", release_b).is_dir())
        self.assertTrue(self._release_dir("DEMO-STREAM", release_c).is_dir())

    def test_abandoned_download_releases_lease_and_cleanup_runs(self) -> None:
        first = self._upload_and_wait("DEMO-ABANDON")
        release_a = first["current_package"]["release_id"]
        dir_a = self._release_dir("DEMO-ABANDON", release_a)

        stream = self.client.get("/samples/DEMO-ABANDON/download", buffered=False)
        self.assertEqual(stream.status_code, 200)

        self._upload_and_wait("DEMO-ABANDON")
        self._upload_and_wait("DEMO-ABANDON")
        # Pinned by the open response, A survives pruning...
        self.assertTrue(dir_a.is_dir())

        # ...until the client goes away without reading the body.
        stream.close()
        self.assertFalse(dir_a.exists())

    def test_client_disconnect_mid_download_releases_lease(self) -> None:
        import http.client
        import threading

        from werkzeug.serving import make_server

        first = self._upload_and_wait("DEMO-DISCONNECT")
        release_a = first["current_package"]["release_id"]
        dir_a = self._release_dir("DEMO-DISCONNECT", release_a)

        server = make_server("127.0.0.1", 0, self.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            conn.request("GET", "/samples/DEMO-DISCONNECT/download")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read(1)  # start the transfer, then walk away
            conn.close()

            # Republish twice; the aborted download must not pin A forever.
            self._upload_and_wait("DEMO-DISCONNECT")
            self._upload_and_wait("DEMO-DISCONNECT")

            deadline = time.time() + 10
            while dir_a.exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertFalse(dir_a.exists())
        finally:
            server.shutdown()
            thread.join(timeout=5)

    def test_range_and_conditional_downloads_still_work(self) -> None:
        self._upload_and_wait("DEMO-RANGE")
        full = self.client.get("/samples/DEMO-RANGE/download/r1")
        self.assertEqual(full.status_code, 200)
        data = full.data
        self.assertGreater(len(data), 40)

        ranged = self.client.get(
            "/samples/DEMO-RANGE/download/r1",
            headers={"Range": "bytes=10-29"},
        )
        self.assertEqual(ranged.status_code, 206)
        self.assertEqual(ranged.data, data[10:30])
        self.assertEqual(
            ranged.headers["Content-Range"], f"bytes 10-29/{len(data)}"
        )

        etag = full.headers["ETag"]
        last_modified = full.headers["Last-Modified"]

        not_modified = self.client.get(
            "/samples/DEMO-RANGE/download/r1",
            headers={"If-None-Match": etag},
        )
        self.assertEqual(not_modified.status_code, 304)

        not_modified_since = self.client.get(
            "/samples/DEMO-RANGE/download/r1",
            headers={"If-Modified-Since": last_modified},
        )
        self.assertEqual(not_modified_since.status_code, 304)

        if_range = self.client.get(
            "/samples/DEMO-RANGE/download/r1",
            headers={"Range": "bytes=0-9", "If-Range": etag},
        )
        self.assertEqual(if_range.status_code, 206)
        self.assertEqual(if_range.data, data[:10])


if __name__ == "__main__":
    unittest.main()
