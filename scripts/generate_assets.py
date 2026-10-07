"""Script to generate high-resolution PNG and SVG presentation & UI assets."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

def create_hero_scope():
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor('#0B1B2B')
    ax.set_facecolor('#0B1B2B')
    
    # Grid lines
    for x in np.arange(0, 10, 0.2):
        ax.axvline(x, color='#1E293B', linewidth=0.5, alpha=0.7)
    for y in np.arange(-1.5, 2.5, 0.5):
        ax.axhline(y, color='#1E293B', linewidth=0.5, alpha=0.7)
        
    t = np.linspace(0, 10, 2000)
    # Synthetic clean ECG waveform
    sig = np.zeros_like(t)
    for i in range(10):
        t0 = i * 1.0 + 0.3
        # P wave
        sig += 0.15 * np.exp(-((t - t0)**2) / 0.003)
        # Q wave
        sig -= 0.15 * np.exp(-((t - (t0 + 0.05))**2) / 0.0005)
        # R wave
        sig += 1.4 * np.exp(-((t - (t0 + 0.08))**2) / 0.0003)
        # S wave
        sig -= 0.35 * np.exp(-((t - (t0 + 0.11))**2) / 0.0006)
        # T wave
        sig += 0.3 * np.exp(-((t - (t0 + 0.25))**2) / 0.008)
        
    ax.plot(t, sig, color='#22D3EE', linewidth=2.0, label='Filtered Signal (0.5-45 Hz)')
    
    # R-peak annotations
    for i in range(10):
        r_time = i * 1.0 + 0.3 + 0.08
        ax.scatter([r_time], [1.4], color='#EF4444', s=60, zorder=5)
        ax.text(r_time, 1.6, 'R', color='#F87171', fontsize=9, fontweight='bold', ha='center')
        
    ax.set_xlim(0, 8)
    ax.set_ylim(-1.0, 2.0)
    ax.set_title("Real-Time Waveform & Peak Detection — MIT-BIH Record 101", color='#F8FAFC', fontsize=12, pad=12)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
        
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'hero_ecg_scope.png'), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(os.path.join(ASSETS_DIR, 'hero_ecg_scope.svg'), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def create_workflow_diagram():
    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.axis('off')
    
    steps = [
        ("01", "Signal Preprocessing", "Butterworth + Notch\nZero-phase filtfilt"),
        ("02", "ECG Measurement", "Pan-Tompkins R-peaks\nHRV, QTc, ST metrics"),
        ("03", "Probabilistic ML", "300-tree RandomForest\nPlatt Calibration P(F|E)"),
        ("04", "Bootstrap Resampling", "100 Beat States\nStability Score S"),
        ("05", "QUBO Optimization", "Conflict Matrix B\nClassical Solver x_i"),
        ("06", "Clinical Synthesis", "Automated Insight\nNext Diagnostic Steps"),
        ("07", "PDF Report", "Publication Grade\nDoctor Vector PDF")
    ]
    
    n = len(steps)
    box_w = 1.4
    gap = 0.4
    total_w = n * box_w + (n - 1) * gap
    start_x = 0.5
    
    for i, (num, title, desc) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        rect = patches.FancyBboxPatch((x, 1.0), box_w, 1.8, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     linewidth=1.2, edgecolor="#0284C7", facecolor="#FFFFFF")
        ax.add_patch(rect)
        
        ax.text(x + box_w/2, 2.45, num, color="#0284C7", fontsize=11, fontweight="bold", ha="center")
        ax.text(x + box_w/2, 2.15, title, color="#0F172A", fontsize=9, fontweight="bold", ha="center")
        ax.text(x + box_w/2, 1.45, desc, color="#475569", fontsize=7.5, ha="center", va="center")
        
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.05, 1.9), xytext=(x + box_w + 0.05, 1.9),
                        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.5))
            
    ax.set_xlim(0, start_x * 2 + total_w)
    ax.set_ylim(0.5, 3.2)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'workflow_pipeline_diagram.png'))
    plt.savefig(os.path.join(ASSETS_DIR, 'workflow_pipeline_diagram.svg'))
    plt.close()

def create_qubo_diagram():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    
    nodes = {
        "NSR": (2, 3.5, "Normal Sinus Rhythm\nP=0.904, S=1.00", "#16A34A", True),
        "RHYTHM": (6, 3.5, "Rhythm Abnormality\nP=0.042, S=0.00", "#64748B", False),
        "BRADY": (2, 1.5, "Bradycardia (<60 bpm)\nP=0.000, S=0.00", "#64748B", False),
        "TACHY": (6, 1.5, "Tachycardia (>100 bpm)\nP=0.000, S=0.00", "#64748B", False)
    }
    
    # Conflict edges
    conflicts = [("NSR", "RHYTHM"), ("BRADY", "TACHY"), ("NSR", "BRADY")]
    for u, v in conflicts:
        x1, y1 = nodes[u][0], nodes[u][1]
        x2, y2 = nodes[v][0], nodes[v][1]
        ax.plot([x1, x2], [y1, y2], color="#EF4444", linestyle="--", linewidth=1.5, zorder=1)
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y, "Conflict\nPenalty b_ij", color="#DC2626", fontsize=7.5,
                ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#FCA5A5", lw=0.5))
        
    for key, (x, y, label, col, selected) in nodes.items():
        border_col = "#16A34A" if selected else "#CBD5E1"
        bg_col = "#DCFCE7" if selected else "#F8FAFC"
        rect = patches.FancyBboxPatch((x - 1.2, y - 0.4), 2.4, 0.8, boxstyle="round,pad=0.1",
                                     linewidth=1.5, edgecolor=border_col, facecolor=bg_col, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, key, color="#0F172A", fontsize=10, fontweight="bold", ha="center", zorder=3)
        ax.text(x, y - 0.2, label, color="#475569", fontsize=7.5, ha="center", zorder=3)
        
        status_text = "SELECTED (x_i = 1)" if selected else "REJECTED (x_i = 0)"
        ax.text(x, y - 0.55, status_text, color=col, fontsize=7, fontweight="bold", ha="center")
        
    ax.set_title("QUBO Conflict Resolution Matrix E(x) = ∑ a_i x_i + ∑ b_ij x_i x_j", color="#0F172A", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 4.2)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'qubo_conflict_resolution_diagram.png'))
    plt.savefig(os.path.join(ASSETS_DIR, 'qubo_conflict_resolution_diagram.svg'))
    plt.close()

def create_tech_stack():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    
    layers = [
        ("Frontend", "React 18 + Modular Vanilla CSS UI Components", "#0284C7"),
        ("REST API", "FastAPI + Pydantic + Asynchronous Pipelines", "#0F2942"),
        ("ECG Signal Processing", "SciPy Butterworth + 50Hz Notch + Zero-Phase filtfilt", "#0D9488"),
        ("Machine Learning", "RandomForest (300 trees) + Sigmoidal Platt Calibration", "#16A34A"),
        ("Uncertainty Estimation", "Nonparametric Bootstrap Resampling (N=100 states)", "#D97706"),
        ("Optimization Engine", "Classical QUBO Matrix Energy Solver (Simulated Annealing)", "#7C3AED"),
        ("Report Generation", "ReportLab Vector PDF + Matplotlib Plotting Engine", "#0F172A")
    ]
    
    for i, (name, detail, color) in enumerate(layers):
        y = 6.0 - i * 0.8
        rect = patches.FancyBboxPatch((0.5, y - 0.3), 7.0, 0.6, boxstyle="round,pad=0.08",
                                     linewidth=1.2, edgecolor=color, facecolor="#F8FAFC")
        ax.add_patch(rect)
        ax.text(0.8, y, name, color=color, fontsize=10, fontweight="bold", va="center")
        ax.text(3.2, y, detail, color="#475569", fontsize=8.5, va="center")
        
    ax.set_title("System Technology Stack & Execution Flow", color="#0F172A", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 6.8)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'tech_stack_architecture_diagram.png'))
    plt.savefig(os.path.join(ASSETS_DIR, 'tech_stack_architecture_diagram.svg'))
    plt.close()

if __name__ == "__main__":
    print("Generating visual presentation assets...")
    create_hero_scope()
    create_workflow_diagram()
    create_qubo_diagram()
    create_tech_stack()
    print("Assets successfully generated in frontend/assets!")
