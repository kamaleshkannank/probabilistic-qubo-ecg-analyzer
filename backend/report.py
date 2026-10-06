"""PDF report generation (reportlab + matplotlib waveform)."""
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image)
from reportlab.lib.styles import getSampleStyleSheet
from .explain import VALUE_GUIDE, FINDING_GUIDE, SCORE_GUIDE, flag_measurement


def waveform_png(raw, filtered, fs, r_peaks=(), path="/tmp/ecg_wave.png"):
    n = min(len(raw), int(10 * fs))
    t = np.arange(n) / fs
    plt.figure(figsize=(10, 3.2))
    plt.plot(t, np.asarray(raw)[:n], lw=0.8, alpha=0.6, label="Raw")
    plt.plot(t, np.asarray(filtered)[:n], lw=1.1, label="Filtered (0.5-45Hz + 50Hz notch)")
    for r in (r_peaks or []):
        if 0 <= r < n:
            plt.axvline(r / fs, color="r", alpha=0.35, lw=0.8)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(fontsize=8)
    plt.title("ECG - raw vs filtered with detected R-peaks (first 10 s)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def build_pdf(payload: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=14 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("<b>Probabilistic Randomized QUBO-Based ECG Analyzer - Report</b>", styles["Title"]))
    story.append(Paragraph("RESEARCH PROTOTYPE - Physician assistance only. NOT an autonomous diagnosis. Probabilities from a trained, calibrated MIT-BIH model + deterministic measurement criteria (see training summary below).", styles["Normal"]))
    story.append(Spacer(1, 6))

    model = payload.get("model", {})
    if model and isinstance(model, dict) and model.get("metrics"):
        story.append(Paragraph("<b>Model training summary (real data)</b>", styles["Heading3"]))
        mrows = [["Target", "Windows (pos)", "5-fold AUROC", "5-fold Brier"]]
        for t, m in model["metrics"].items():
            mrows.append([t, f"{m['n']} ({m['positives']})",
                          str(m["auroc_5fold"]), str(m["brier_5fold"])])
        mt = Table(mrows, colWidths=[40 * mm, 45 * mm, 40 * mm, 40 * mm])
        mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e3a5d")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(mt)
        story.append(Paragraph(
            f"Model: {model.get('model', '')} Records: {len(model.get('records', []))} MIT-BIH records, "
            f"{model.get('windows', '')} expert-annotated windows. "
            "BRADY/TACHY/QT/ST are deterministic clinical thresholds on real measurements.",
            styles["Normal"]))
        story.append(Spacer(1, 6))

    meta = payload.get("meta", {})
    rows = [[k, str(v)] for k, v in meta.items()]
    t = Table([["Item", "Value"]] + rows, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e3a5d")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(t); story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Measurements with doctor-level explanation (estimates)</b>", styles["Heading3"]))
    story.append(Paragraph("Each row: patient value, normal reference, what it means, and how to read an abnormal value. Flags are heuristic - confirm on 12-lead.", styles["Normal"]))
    story.append(Spacer(1, 4))
    feat = payload.get("features", {})
    frows = [["Measurement", "Value", "Flag", "Normal ref / meaning"]]
    for k, v in feat.items():
        if k == "r_peaks" or k == "fs":
            continue
        g = VALUE_GUIDE.get(k)
        if g:
            title, normal, _interp = g
            cell = f"<b>{title}</b><br/>{normal}<br/><i>{_interp[:220]}</i>"
        else:
            title, cell = k, str(v)
            normal = ""
        frows.append([title, str(v), flag_measurement(k, v), Paragraph(cell, styles["Normal"])])
    t2 = Table(frows, colWidths=[32 * mm, 22 * mm, 32 * mm, 84 * mm])
    t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#147a8a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t2); story.append(Spacer(1, 6))

    # Score guide: P, S, QUBO, SQI
    story.append(Paragraph("<b>How to read the scores</b>", styles["Heading3"]))
    for _k in ("probability", "stability", "qubo", "sqi"):
        title, what, how = SCORE_GUIDE[_k]
        story.append(Paragraph(f"* <b>{title}</b> - {what} <i>{how}</i>", styles["Normal"]))
    story.append(Spacer(1, 6))

    img_path = payload.get("waveform_png")
    if img_path and os.path.exists(img_path):
        story.append(Paragraph("<b>Waveform</b>", styles["Heading3"]))
        story.append(Image(img_path, width=170 * mm, height=54 * mm))
        story.append(Spacer(1, 6))

    probs = payload.get("probabilities", [])
    story.append(Paragraph("<b>Candidate findings - P(F|E) trained + stability + QUBO, each explained</b>", styles["Heading3"]))
    prows = [["Finding", "P (trained)", "Method", "Stability S", "QUBO x"]]
    for p in probs:
        prows.append([p["name"], str(p["probability"]), str(p.get("method", "")),
                      str(p.get("stability", "")), str(p.get("qubo_selected", ""))])
    t3 = Table(prows, colWidths=[62 * mm, 22 * mm, 32 * mm, 24 * mm, 24 * mm])
    t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e3a5d")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(t3); story.append(Spacer(1, 4))
    for p in probs:
        g = FINDING_GUIDE.get(p.get("code", ""), "")
        story.append(Paragraph(
            f"* <b>{p['name']}</b> (P={p['probability']}, S={p.get('stability')}, QUBO={p.get('qubo_selected')}) - {g}",
            styles["Normal"]))

    story.append(Paragraph("<b>Possible next diagnostic steps (informational, not medical advice)</b>", styles["Heading3"]))
    for p in probs:
        if p.get("qubo_selected") == 1 or p.get("probability", 0) >= 0.5:
            for s in p.get("next_steps", []):
                story.append(Paragraph(f"* <b>{p['code']}</b>: {s}", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Method: Butterworth HP 0.5Hz (4th) + 50Hz notch Q=30 + LP 45Hz (4th), zero-phase filtfilt. "
                           "NSR/Rhythm/Ectopic probabilities: calibrated RandomForest trained on expert-annotated MIT-BIH windows. "
                           "BRADY/TACHY/QT/ST: deterministic clinical thresholds on real measurements. "
                           "Uncertainty: nonparametric bootstrap over the record's own beats (seeded, reproducible). "
                           "QUBO: classical exhaustive/simulated-annealing. S(F)=N_F/N. "
                           "Research prototype - requires independent validation before clinical use.",
                           styles["Normal"]))
    doc.build(story)
    return buf.getvalue()
