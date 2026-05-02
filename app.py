"""
Flask backend for the Churn Prediction Dashboard.

Routes:
  POST /api/upload          — upload CSV, kick off ML pipeline in background
  GET  /api/status/<job_id> — poll job status
  GET  /api/results/<job_id>— fetch results JSON when complete
  POST /api/autorun         — run pipeline on bundled Telco CSV
"""

import io
import os
import uuid
import threading

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory job store  {job_id: {"status": ..., "results": ..., "error": ...}}
jobs: dict = {}


def _run_pipeline(job_id: str, file_data: io.BytesIO) -> None:
    try:
        from ml_pipeline import run_pipeline
        results = run_pipeline(file_data)
        jobs[job_id]["results"] = results
        jobs[job_id]["status"] = "complete"
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "message": "Churn Prediction API is running",
        "endpoints": {
            "upload_csv":  "POST /api/upload",
            "autorun":     "POST /api/autorun",
            "job_status":  "GET  /api/status/<job_id>",
            "job_results": "GET  /api/results/<job_id>",
        }
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    job_id = str(uuid.uuid4())
    file_data = io.BytesIO(file.read())  # read entirely into memory — no disk write

    jobs[job_id] = {"status": "processing", "results": None, "error": None}
    threading.Thread(target=_run_pipeline, args=(job_id, file_data), daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": job["status"], "error": job.get("error")})


@app.route("/api/results/<job_id>")
def results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "complete":
        return jsonify({"error": "Job not complete yet"}), 400
    return jsonify(job["results"])


@app.route("/api/autorun", methods=["POST"])
def autorun():
    default_csv = os.path.join(os.path.dirname(__file__), "Telco-Customer-Churn.csv")
    if not os.path.exists(default_csv):
        return jsonify({"error": "Default dataset not found"}), 404

    with open(default_csv, "rb") as f:
        file_data = io.BytesIO(f.read())

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "results": None, "error": None}
    threading.Thread(target=_run_pipeline, args=(job_id, file_data), daemon=True).start()
    return jsonify({"job_id": job_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print("Churn Prediction Dashboard — Flask Backend")
    print("=" * 60)
    print(f"API running at:  http://localhost:{port}")
    print("Frontend (Vite): http://localhost:5173")
    print("=" * 60 + "\n")
    app.run(debug=False, port=port, host="0.0.0.0")
