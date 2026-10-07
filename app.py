import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import pandas as pd
from analytics_engine import (
    normalize_dataframe,
    compute_overview_analytics,
    analyze_sentiment_text,
    optimize_draft_caption
)

app = Flask(__name__)
app.secret_key = "social-media-analytics-secret-key-2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

# In-memory storage for active working dataset
ACTIVE_DATASET = {
    "name": "Global Brand Campaign (Demo)",
    "df": None,
    "analytics": None,
    "last_updated": None
}

def load_default_dataset():
    """Loads default sample dataset into active memory on startup."""
    default_csv = os.path.join(SAMPLE_DIR, "brand_campaign.csv")
    if os.path.exists(default_csv):
        raw_df = pd.read_csv(default_csv)
        norm_df = normalize_dataframe(raw_df)
        analytics = compute_overview_analytics(norm_df)
        ACTIVE_DATASET["name"] = "Global Brand Campaign (ApexActive)"
        ACTIVE_DATASET["df"] = norm_df
        ACTIVE_DATASET["analytics"] = analytics
        ACTIVE_DATASET["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

load_default_dataset()

# ----------------- WEB PAGES ----------------- #

@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    dataset_name = ACTIVE_DATASET.get("name", "Custom Dataset")
    last_updated = ACTIVE_DATASET.get("last_updated", "")
    analytics = ACTIVE_DATASET.get("analytics") or {}
    return render_template(
        "dashboard.html",
        active_tab="dashboard",
        dataset_name=dataset_name,
        last_updated=last_updated,
        analytics=analytics
    )

@app.route("/upload", methods=["GET", "POST"])
def upload():
    dataset_name = ACTIVE_DATASET.get("name", "Custom Dataset")
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose a valid CSV or JSON file.", "error")
            return redirect(url_for("upload"))

        try:
            filename = file.filename.lower()
            if filename.endswith(".csv"):
                raw_df = pd.read_csv(file)
            elif filename.endswith(".json"):
                raw_df = pd.read_json(file)
            else:
                flash("Unsupported file format. Please upload a .csv or .json file.", "error")
                return redirect(url_for("upload"))

            if raw_df.empty:
                flash("The uploaded file is empty.", "error")
                return redirect(url_for("upload"))

            norm_df = normalize_dataframe(raw_df)
            analytics = compute_overview_analytics(norm_df)

            ACTIVE_DATASET["name"] = file.filename
            ACTIVE_DATASET["df"] = norm_df
            ACTIVE_DATASET["analytics"] = analytics
            ACTIVE_DATASET["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            flash(f"Successfully loaded {len(norm_df)} records from {file.filename}!", "success")
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            return redirect(url_for("upload"))

    return render_template("upload.html", active_tab="upload", dataset_name=dataset_name)

@app.route("/sentiment")
def sentiment():
    dataset_name = ACTIVE_DATASET.get("name", "Custom Dataset")
    return render_template("sentiment.html", active_tab="sentiment", dataset_name=dataset_name)

@app.route("/optimizer")
def optimizer():
    dataset_name = ACTIVE_DATASET.get("name", "Custom Dataset")
    return render_template("optimizer.html", active_tab="optimizer", dataset_name=dataset_name)

@app.route("/reports")
def reports():
    dataset_name = ACTIVE_DATASET.get("name", "Custom Dataset")
    analytics = ACTIVE_DATASET.get("analytics") or {}
    return render_template("reports.html", active_tab="reports", dataset_name=dataset_name, analytics=analytics)


# ----------------- API ENDPOINTS ----------------- #

@app.route("/api/analytics")
def get_analytics():
    """Returns current processed analytics data as JSON."""
    analytics = ACTIVE_DATASET.get("analytics")
    if not analytics:
        return jsonify({"error": "No active dataset loaded"}), 404
    return jsonify({
        "dataset_name": ACTIVE_DATASET["name"],
        "last_updated": ACTIVE_DATASET["last_updated"],
        "data": analytics
    })

@app.route("/api/load-preset", methods=["POST"])
def load_preset():
    """Loads a built-in preset sample dataset."""
    preset = request.json.get("preset", "brand_campaign")
    mapping = {
        "brand_campaign": ("brand_campaign.csv", "Global Brand Campaign (ApexActive)"),
        "tech_launch": ("tech_launch.csv", "Tech SaaS Launch (DevPulseAI)")
    }

    if preset not in mapping:
        return jsonify({"status": "error", "message": "Unknown preset"}), 400

    filename, display_name = mapping[preset]
    file_path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Preset file not found"}), 404

    try:
        raw_df = pd.read_csv(file_path)
        norm_df = normalize_dataframe(raw_df)
        analytics = compute_overview_analytics(norm_df)

        ACTIVE_DATASET["name"] = display_name
        ACTIVE_DATASET["df"] = norm_df
        ACTIVE_DATASET["analytics"] = analytics
        ACTIVE_DATASET["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        return jsonify({
            "status": "success",
            "message": f"Successfully activated preset: {display_name}",
            "dataset_name": display_name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/analyze-sentiment", methods=["POST"])
def api_analyze_sentiment():
    """Performs real-time sentiment & emotion analysis on given text."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    result = analyze_sentiment_text(text)
    return jsonify(result)

@app.route("/api/optimize-post", methods=["POST"])
def api_optimize_post():
    """Predicts engagement and provides actionable post optimization feedback."""
    payload = request.get_json(silent=True) or {}
    caption = payload.get("caption", "")
    platform = payload.get("platform", "Instagram")
    hour = int(payload.get("hour", 14))
    
    result = optimize_draft_caption(caption, platform=platform, target_hour=hour)
    return jsonify(result)

@app.route("/api/download-template")
def download_template():
    """Download template CSV for formatting custom uploads."""
    template_path = os.path.join(SAMPLE_DIR, "template.csv")
    if os.path.exists(template_path):
        return send_file(
            template_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="social_media_analytics_template.csv"
        )
    return jsonify({"error": "Template not found"}), 404

@app.route("/api/export-csv")
def export_csv():
    """Exports current active dataset as a CSV."""
    df = ACTIVE_DATASET.get("df")
    if df is None or df.empty:
        return jsonify({"error": "No data available to export"}), 400

    out = io.StringIO()
    # Export clean columns
    export_cols = [c for c in df.columns if c not in ["parsed_date"]]
    df[export_cols].to_csv(out, index=False)
    out.seek(0)

    filename = f"social_analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        io.BytesIO(out.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

@app.route("/api/export-report-json")
def export_report_json():
    """Exports structured analytics summary as a JSON file."""
    analytics = ACTIVE_DATASET.get("analytics")
    if not analytics:
        return jsonify({"error": "No data to export"}), 400

    report_payload = {
        "title": "Social Media Performance & Sentiment Analytics Report",
        "dataset_name": ACTIVE_DATASET.get("name"),
        "generated_at": datetime.now().isoformat(),
        "summary": analytics.get("kpis"),
        "platform_performance": analytics.get("platforms"),
        "sentiment_overview": analytics.get("sentiment_distribution"),
        "top_performing_posts": analytics.get("top_posts"),
        "top_hashtags": analytics.get("top_hashtags")
    }

    json_str = json.dumps(report_payload, indent=2)
    filename = f"social_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return send_file(
        io.BytesIO(json_str.encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Social Media Analytics server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

