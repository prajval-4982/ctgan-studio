"""
Flask Web Application — Synthetic Data Generation using CTGAN
==============================================================
Main entry-point. Provides routes for uploading datasets,
training CTGAN, generating synthetic data, evaluating results,
and downloading output files.
"""

import os
import glob
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, session,
)

# --- Module imports -----------------------------------------------------------
from modules.dataset_loader import load_dataset, preview_dataset, validate_dataset
from modules.preprocessing import preprocess
from modules.ctgan_trainer import train_model, save_model, load_model
from modules.synthetic_generator import generate_synthetic_data, export_data
from modules.statistical_evaluator import (
    compare_statistics, plot_distributions, plot_correlation_heatmaps,
    calculate_scientific_scores, get_histogram_data,
)
from modules.ml_evaluator import evaluate_ml
from modules.privacy_evaluator import calculate_privacy_metrics
from modules.report_visualizer import (
    generate_comparison_chart, format_ml_results_table, format_stats_table,
    plot_feature_importance,
)
from modules.audit_report import generate_audit_report

# --- App setup ----------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "ctgan-synthetic-data-2024"

# Directories
UPLOAD_DIR   = os.path.join("data", "real")
SYNTH_DIR    = os.path.join("data", "synthetic")
MODEL_DIR    = "models"
PLOT_DIR     = os.path.join("static", "plots")

for d in (UPLOAD_DIR, SYNTH_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# --- Helpers ------------------------------------------------------------------

def _clear_plots():
    """Remove old plot images from the plots directory."""
    for f in glob.glob(os.path.join(PLOT_DIR, "*.png")):
        os.remove(f)


# =============================================================================
#  ROUTES
# =============================================================================

@app.route("/")
def index():
    """Home page — project overview."""
    return render_template("index.html")


# ---------- Dataset Upload ----------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload and preview a CSV dataset."""
    preview_html = None
    info = None

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("Please select a CSV file.", "warning")
            return redirect(url_for("upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "danger")
            return redirect(url_for("upload"))

        # Save uploaded file
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        try:
            df = load_dataset(filepath)
            validation = validate_dataset(df)

            if not validation["valid"]:
                flash(validation["message"], "danger")
                return redirect(url_for("upload"))

            # Store path in session for later stages
            session["dataset_path"] = filepath
            session["dataset_name"] = file.filename

            preview = preview_dataset(df)
            preview_html = preview.to_html(
                classes="table table-dark table-striped table-hover",
                border=0, index=False,
            )
            info = {
                "rows": df.shape[0],
                "cols": df.shape[1],
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "message": validation["message"],
            }
            flash("Dataset uploaded and validated successfully!", "success")
        except Exception as e:
            flash(f"Error loading dataset: {e}", "danger")
            return redirect(url_for("upload"))

    return render_template("upload.html", preview_html=preview_html, info=info)


# ---------- Synthetic Generation ----------------------------------------------

@app.route("/generate", methods=["GET", "POST"])
def generate():
    """Configure record count, train CTGAN, and generate synthetic data."""
    if "dataset_path" not in session:
        flash("Upload a dataset first.", "warning")
        return redirect(url_for("upload"))

    if request.method == "POST":
        try:
            num_rows = int(request.form.get("num_rows", 1000))
            epochs = int(request.form.get("epochs", 300))
            model_type = request.form.get("model_type", "CTGAN").upper()
        except ValueError:
            flash("Please enter valid numbers.", "danger")
            return redirect(url_for("generate"))

        try:
            # Load & preprocess
            df = load_dataset(session["dataset_path"])
            df_clean, meta = preprocess(df)

            # Train model (CTGAN or TVAE)
            model = train_model(df_clean, model_type=model_type, epochs=epochs)
            save_model(model)

            # Generate synthetic data
            synthetic_df = generate_synthetic_data(model, num_rows)

            # Export in all formats
            export_data(synthetic_df, "csv", SYNTH_DIR)
            export_data(synthetic_df, "xlsx", SYNTH_DIR)
            export_data(synthetic_df, "json", SYNTH_DIR)
            export_data(synthetic_df, "docx", SYNTH_DIR)

            # ---- Evaluation ----
            _clear_plots()

            # Statistical
            stats = compare_statistics(df_clean, synthetic_df)
            dist_plots = plot_distributions(df_clean, synthetic_df, PLOT_DIR)
            corr_plot = plot_correlation_heatmaps(df_clean, synthetic_df, PLOT_DIR)

            # ML evaluation
            ml_results = evaluate_ml(df_clean, synthetic_df)

            # Privacy evaluation
            privacy = calculate_privacy_metrics(df_clean, synthetic_df)

            # Scientific quality scores (KS Test, JS Distance, Corr Similarity)
            scientific_scores = calculate_scientific_scores(df_clean, synthetic_df)

            # Raw Histograms for Chart.js
            hist_data = get_histogram_data(df_clean, synthetic_df)

            # Visualization
            ml_chart = generate_comparison_chart(ml_results, PLOT_DIR)

            # Feature importance plot
            fi_chart = None
            if ml_results.get("feature_importance"):
                fi_chart = plot_feature_importance(ml_results["feature_importance"], PLOT_DIR)

            mean_html, std_html = format_stats_table(stats)
            real_tbl, synth_tbl = format_ml_results_table(ml_results)

            # Synthetic preview
            synth_preview = synthetic_df.head(15).to_html(
                classes="table table-dark table-striped table-hover",
                border=0, index=False,
            )

            # Collect plot filenames for the template
            plot_files = [os.path.basename(p) for p in dist_plots]
            plot_files.append(os.path.basename(corr_plot))
            plot_files.append(os.path.basename(ml_chart))

            # Store results in session-safe manner (keep simple strings)
            session["generated"] = True
            session["num_rows"] = num_rows
            session["model_type"] = model_type

            flash(f"Successfully generated {num_rows} synthetic records!", "success")

            return render_template(
                "results.html",
                synth_preview=synth_preview,
                mean_html=mean_html,
                std_html=std_html,
                real_tbl=real_tbl,
                synth_tbl=synth_tbl,
                plot_files=plot_files,
                num_rows=num_rows,
                model_type=model_type,
                privacy=privacy,
                feature_importance=ml_results.get("feature_importance"),
                scientific_scores=scientific_scores,
                hist_data=hist_data,
            )

        except Exception as e:
            flash(f"Generation failed: {e}", "danger")
            import traceback; traceback.print_exc()
            return redirect(url_for("generate"))

    return render_template("generate.html", dataset_name=session.get("dataset_name"))


# ---------- Results -----------------------------------------------------------

@app.route("/results")
def results():
    """
    Show the most recent generation results.
    Re-loads saved synthetic data and re-generates plots if needed.
    """
    synth_path = os.path.join(SYNTH_DIR, "synthetic_data.csv")

    if not os.path.exists(synth_path) or "dataset_path" not in session:
        flash("No synthetic data found. Generate data first.", "warning")
        return redirect(url_for("generate"))

    try:
        df_real = load_dataset(session["dataset_path"])
        df_clean, _ = preprocess(df_real)
        synthetic_df = pd.read_csv(synth_path)

        # Re-generate plots if missing
        plot_files_on_disk = glob.glob(os.path.join(PLOT_DIR, "*.png"))
        if not plot_files_on_disk:
            _clear_plots()
            dist_plots = plot_distributions(df_clean, synthetic_df, PLOT_DIR)
            corr_plot = plot_correlation_heatmaps(df_clean, synthetic_df, PLOT_DIR)
            ml_results = evaluate_ml(df_clean, synthetic_df)
            ml_chart = generate_comparison_chart(ml_results, PLOT_DIR)
        else:
            ml_results = evaluate_ml(df_clean, synthetic_df)

        # Feature importance plot
        if ml_results.get("feature_importance"):
            plot_feature_importance(ml_results["feature_importance"], PLOT_DIR)

        stats = compare_statistics(df_clean, synthetic_df)
        mean_html, std_html = format_stats_table(stats)
        real_tbl, synth_tbl = format_ml_results_table(ml_results)

        synth_preview = synthetic_df.head(15).to_html(
            classes="table table-dark table-striped table-hover",
            border=0, index=False,
        )

        plot_files = [os.path.basename(f) for f in glob.glob(os.path.join(PLOT_DIR, "*.png"))]

        # Privacy evaluation
        privacy = calculate_privacy_metrics(df_clean, synthetic_df)

        # Scientific quality scores
        scientific_scores = calculate_scientific_scores(df_clean, synthetic_df)

        # Raw Histograms for Chart.js
        hist_data = get_histogram_data(df_clean, synthetic_df)

        return render_template(
            "results.html",
            synth_preview=synth_preview,
            mean_html=mean_html,
            std_html=std_html,
            real_tbl=real_tbl,
            synth_tbl=synth_tbl,
            plot_files=plot_files,
            num_rows=len(synthetic_df),
            model_type=session.get("model_type", "CTGAN"),
            privacy=privacy,
            feature_importance=ml_results.get("feature_importance"),
            scientific_scores=scientific_scores,
            hist_data=hist_data,
        )
    except Exception as e:
        flash(f"Error loading results: {e}", "danger")
        import traceback; traceback.print_exc()
        return redirect(url_for("generate"))


# ---------- Downloads ---------------------------------------------------------

@app.route("/download/<filetype>")
def download(filetype):
    """Download synthetic data in the requested format."""
    filemap = {
        "csv":  "synthetic_data.csv",
        "xlsx": "synthetic_data.xlsx",
        "json": "synthetic_data.json",
        "docx": "synthetic_data.docx",
    }

    filename = filemap.get(filetype)
    if not filename:
        flash("Unsupported file type.", "danger")
        return redirect(url_for("results"))

    path = os.path.join(SYNTH_DIR, filename)
    if not os.path.exists(path):
        flash("File not found. Generate data first.", "warning")
        return redirect(url_for("generate"))

    return send_file(os.path.abspath(path), as_attachment=True)


@app.route("/download/audit-report")
def download_audit_report():
    """Generate and download a Technical Audit Report (DOCX)."""
    synth_path = os.path.join(SYNTH_DIR, "synthetic_data.csv")

    if not os.path.exists(synth_path) or "dataset_path" not in session:
        flash("No synthetic data found. Generate data first.", "warning")
        return redirect(url_for("generate"))

    try:
        df_real = load_dataset(session["dataset_path"])
        df_clean, _ = preprocess(df_real)
        synthetic_df = pd.read_csv(synth_path)

        stats = compare_statistics(df_clean, synthetic_df)
        ml_results = evaluate_ml(df_clean, synthetic_df)
        privacy = calculate_privacy_metrics(df_clean, synthetic_df)
        scientific_scores = calculate_scientific_scores(df_clean, synthetic_df)

        dataset_name = os.path.basename(session.get("dataset_path", "Unknown"))
        model_type = session.get("model_type", "CTGAN")
        num_rows = len(synthetic_df)

        report_path = generate_audit_report(
            stats=stats,
            ml_results=ml_results,
            privacy=privacy,
            model_type=model_type,
            num_rows=num_rows,
            dataset_name=dataset_name,
            scientific_scores=scientific_scores,
        )

        return send_file(report_path, as_attachment=True,
                         download_name="CTGAN_Technical_Audit_Report.docx")

    except Exception as e:
        flash(f"Error generating audit report: {e}", "danger")
        import traceback; traceback.print_exc()
        return redirect(url_for("results"))


# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
