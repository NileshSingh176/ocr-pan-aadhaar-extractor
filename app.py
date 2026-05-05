from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import cv2
import mediapipe as mp
import numpy as np
import os, json, base64, math
from uuid import uuid4
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

# ──────────────────────────────────────────────────────────────
# MODULE FOLDER STRUCTURE
# Each exercise gets its own input/ and output/ sub-directory.
# Wrong-angle logs are saved as JSON inside the output folder.
# ──────────────────────────────────────────────────────────────
MODULES = [
    "pushups",
    "squat",
    "single_leg_squat",
    "broad_jump",
    "walking_lunges",
    "squat_with_stick",
    "vertical_jump",
    "speed_20m",
]

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))


INPUTS_ROOT   = os.path.join(BASE_DIR, "inputs")
OUTPUTS_ROOT  = os.path.join(BASE_DIR, "outputs")

def module_input_dir(module):
    return os.path.join(INPUTS_ROOT, module)

def module_output_dir(module):
    return os.path.join(OUTPUTS_ROOT, module)

def ensure_module_dirs():
    for m in MODULES:
        os.makedirs(module_input_dir(m),  exist_ok=True)
        os.makedirs(module_output_dir(m), exist_ok=True)

# ── Wrong-angle log helpers ────────────────────────────────────
def wrong_angle_log_path(module, session_id):
    """Return path to the JSON log file for this session."""
    return os.path.join(module_output_dir(module), f"wrong_angles_{session_id}.json")

def save_wrong_angle_log(module, session_id, source_filename, wrong_angle_events):
    """
    Persist wrong-angle events to a JSON file.

    Schema
    ------
    {
      "module":   "pushups",
      "session":  "<uuid>",
      "source":   "athlete_video.mp4",
      "recorded": "2025-05-02T14:30:00",
      "events": [
        {
          "frame":      42,
          "joint":      "left_elbow",
          "angle_deg":  78.3,
          "threshold":  "< 85 or > 165",
          "note":       "Elbow flaring — keep at 45° from torso"
        },
        ...
      ]
    }
    """
    payload = {
        "module":   module,
        "session":  session_id,
        "source":   source_filename,
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "events":   wrong_angle_events,
    }
    log_path = wrong_angle_log_path(module, session_id)
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    return log_path



# ── Session Report helpers ─────────────────────────────────────
def report_paths(module, session_id):
    """Return (json_path, txt_path, html_path) for the session report."""
    base = os.path.join(module_output_dir(module), f"report_{session_id}")
    return base + ".json", base + ".txt", base + ".html"


def _build_html_report(payload, wrong_events):
    """Render a self-contained HTML report matching the PerfLab dark UI."""
    exercise_title = payload["exercise"].upper()
    module_upper   = payload["module"].upper()
    recorded       = payload.get("recorded", "")
    source         = payload.get("source", "")
    form_score     = payload.get("form_score", "N/A")
    metrics        = payload.get("metrics", [])
    strengths      = payload.get("strengths", [])
    issues         = payload.get("issues", [])
    per_rep        = payload.get("per_rep", [])
    wrong_count    = payload.get("wrong_angle_count", 0)

    # ── metric cards ─────────────────────────────────────────────
    metric_cards_html = ""
    for m in metrics:
        metric_cards_html += f"""
        <div class="metric-card">
          <div class="metric-value">{m['value']}</div>
          <div class="metric-label">{m['label'].upper()}</div>
        </div>"""

    # ── strengths list ────────────────────────────────────────────
    strengths_html = "".join(
        f'<li><span class="dash">–</span> {s}</li>' for s in strengths
    ) or '<li><span class="dash">–</span> None recorded</li>'

    # ── issues list ───────────────────────────────────────────────
    issues_html = "".join(
        f'<li><span class="dash">–</span> {i}</li>' for i in issues
    ) or '<li><span class="dash">–</span> No major issues detected — keep up the great work!</li>'

    # ── per-rep table ─────────────────────────────────────────────
    per_rep_section = ""
    if per_rep:
        # Discover all numeric keys besides "rep"
        sample = per_rep[0]
        col_keys = [k for k in sample.keys() if k != "rep"]
        col_headers = "".join(
            f'<th>{k.replace("_", " ").upper()}</th>' for k in col_keys
        )
        rows_html = ""
        for row in per_rep:
            cells = "".join(
                f'<td>{row.get(k, ""):.1f}°</td>'
                if isinstance(row.get(k), float) else f'<td>{row.get(k, "")}</td>'
                for k in col_keys
            )
            rows_html += f"<tr><td>{row['rep']}</td>{cells}</tr>"

        per_rep_section = f"""
        <section class="section">
          <h2 class="section-title">PER-REP BREAKDOWN</h2>
          <table class="rep-table">
            <thead><tr><th>REP</th>{col_headers}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </section>"""

    # ── wrong-angle events are logged to JSON only (not shown in HTML report) ──
    wrong_section = ""

    # ── analysis summary ──────────────────────────────────────────
    score_val = form_score if isinstance(form_score, (int, float)) else 0
    if score_val >= 9:
        rating_label = "ELITE"; rating_color = "#4caf50"; rating_desc = "Outstanding movement quality — minor refinements only."
    elif score_val >= 7:
        rating_label = "GOOD"; rating_color = "#8bc34a"; rating_desc = "Solid technique with specific areas to improve."
    elif score_val >= 5:
        rating_label = "DEVELOPING"; rating_color = "#e8a020"; rating_desc = "Several form issues present — structured correction recommended."
    else:
        rating_label = "NEEDS WORK"; rating_color = "#e53935"; rating_desc = "Multiple technique faults detected — coach-guided correction advised."

    issues_count   = len([i for i in issues if "No major" not in i])
    strengths_count = len(strengths)

    summary_section = f"""
        <section class="section">
          <h2 class="section-title">ANALYSIS SUMMARY</h2>
          <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center;margin-bottom:1rem;">
            <div style="font-family:var(--mono);font-size:2.4rem;color:{rating_color};letter-spacing:0.06em;">{rating_label}</div>
            <div style="font-family:var(--mono);font-size:0.85rem;color:var(--muted);max-width:480px;">{rating_desc}</div>
          </div>
          <div style="display:flex;gap:2rem;font-family:var(--mono);font-size:0.82rem;color:var(--muted);">
            <span style="color:var(--green);">✓ {strengths_count} STRENGTH{'S' if strengths_count!=1 else ''}</span>
            <span style="color:var(--red);">✕ {issues_count} CORRECTION{'S' if issues_count!=1 else ''}</span>
            <span>FORM SCORE &nbsp;<span style="color:var(--amber);">{form_score}/10</span></span>
          </div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PerfLab — {exercise_title} Report</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #1a1e1a;
    --surface:  #222822;
    --surface2: #2a2e2a;
    --border:   #333a33;
    --amber:    #e8a020;
    --amber2:   #f0b840;
    --green:    #4caf50;
    --red:      #e53935;
    --red-bg:   #2a1a1a;
    --green-bg: #1a2a1a;
    --text:     #d4d8d0;
    --muted:    #7a8a7a;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Barlow Condensed', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ font-size: 16px; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 1rem;
    line-height: 1.5;
    min-height: 100vh;
  }}
  .header {{
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 2rem; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  .badge-complete {{
    background: transparent; border: 1.5px solid var(--green);
    color: var(--green); font-family: var(--mono); font-size: 0.7rem;
    padding: 0.2rem 0.6rem; letter-spacing: 0.08em; white-space: nowrap;
  }}
  .header-title {{
    font-family: var(--mono); font-size: 1.1rem; font-weight: 400;
    letter-spacing: 0.06em; color: var(--text); text-transform: uppercase;
  }}
  .header-meta {{
    margin-left: auto; font-family: var(--mono); font-size: 0.72rem;
    color: var(--muted); text-align: right; line-height: 1.7;
  }}
  .content {{ padding: 2rem; max-width: 1400px; margin: 0 auto; }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
  }}
  .metric-card {{
    background: var(--surface); border: 1px solid var(--border);
    padding: 1.4rem 1rem 1rem; text-align: center;
  }}
  .metric-value {{ font-family: var(--mono); font-size: 2rem; color: var(--amber); line-height: 1.1; margin-bottom: 0.5rem; }}
  .metric-label {{ font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.1em; color: var(--muted); text-transform: uppercase; }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }}
  @media (max-width: 700px) {{ .panels {{ grid-template-columns: 1fr; }} }}
  .panel {{ border: 1px solid var(--border); padding: 1.2rem 1.4rem; }}
  .panel.strengths {{ background: var(--green-bg); border-color: #2d4a2d; }}
  .panel.corrections {{ background: var(--red-bg); border-color: #4a2d2d; }}
  .panel-title {{
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.4rem;
  }}
  .panel.strengths .panel-title {{ color: var(--green); }}
  .panel.corrections .panel-title {{ color: var(--red); }}
  .panel ul {{ list-style: none; }}
  .panel ul li {{ font-size: 0.95rem; color: var(--text); margin-bottom: 0.5rem; display: flex; gap: 0.5rem; }}
  .dash {{ color: var(--muted); flex-shrink: 0; }}
  .section {{ margin-bottom: 2.5rem; }}
  .section-title {{
    font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.12em;
    color: var(--amber); text-transform: uppercase; margin-bottom: 1rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.6rem;
  }}
  .badge {{
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--muted); font-size: 0.7rem; padding: 0.1rem 0.45rem; border-radius: 2px;
  }}
  .rep-table {{ width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.85rem; }}
  .rep-table thead tr {{ border-bottom: 1px solid var(--border); }}
  .rep-table th {{
    text-align: left; font-size: 0.65rem; letter-spacing: 0.1em;
    color: var(--muted); padding: 0.5rem 1.2rem 0.5rem 0; font-weight: 400; white-space: nowrap;
  }}
  .rep-table tbody tr {{ border-bottom: 1px solid var(--border); transition: background 0.12s; }}
  .rep-table tbody tr:hover {{ background: var(--surface2); }}
  .rep-table td {{ padding: 0.75rem 1.2rem 0.75rem 0; color: var(--text); }}
  .no-issues {{ color: var(--muted); font-family: var(--mono); font-size: 0.85rem; }}
  .footer {{
    border-top: 1px solid var(--border); padding: 1rem 2rem;
    font-family: var(--mono); font-size: 0.65rem; color: var(--muted);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.4rem;
  }}
  @media print {{
    body {{ background: #fff; color: #000; }}
    .metric-value {{ color: #b06000; }}
    .panel.strengths {{ background: #efffef; }}
    .panel.corrections {{ background: #ffefef; }}
  }}
</style>
</head>
<body>

<header class="header">
  <div class="badge-complete">✓ COMPLETE</div>
  <div class="header-title">{exercise_title} REPORT</div>
  <div class="header-meta">
    <div>SOURCE &nbsp;{source}</div>
    <div>SESSION {payload['session'][:12]}…</div>
    <div>{recorded}</div>
  </div>
</header>

<main class="content">

  <div class="metrics-grid">
    {metric_cards_html}
  </div>

  {summary_section}

  <div class="panels">
    <div class="panel strengths">
      <div class="panel-title">✓ &nbsp;STRENGTHS</div>
      <ul>{strengths_html}</ul>
    </div>
    <div class="panel corrections">
      <div class="panel-title">✕ &nbsp;CORRECTIONS</div>
      <ul>{issues_html}</ul>
    </div>
  </div>

  {per_rep_section}

</main>

<footer class="footer">
  <span>PerfLab — AI Athlete Movement Analysis</span>
  <span>Module: {module_upper} &nbsp;|&nbsp; Session: {payload['session']}</span>
</footer>

</body>
</html>"""


def save_report(module, session_id, source_filename, result, wrong_events):
    """
    Save a structured summary report for the session.

    Three files are written inside  outputs/<module>/:
      • report_<session_id>.json  – machine-readable full report
      • report_<session_id>.txt   – plain-text summary
      • report_<session_id>.html  – visual HTML report matching PerfLab UI
    """
    per_rep = (
        result.get("per_rep", []) or
        result.get("rep_data", []) or
        result.get("reps", []) or []
    )

    payload = {
        "module":             module,
        "session":            session_id,
        "source":             source_filename,
        "recorded":           datetime.now().isoformat(timespec="seconds"),
        "exercise":           result.get("exercise", module),
        "metrics":            result.get("metrics", []),
        "form_score":         result.get("form_score", "N/A"),
        "issues":             result.get("issues", []),
        "strengths":          result.get("strengths", []),
        "wrong_angle_count":  result.get("wrong_angle_count", 0),
        "wrong_angle_events": wrong_events,
        "per_rep":            per_rep,
    }

    json_path, txt_path, html_path = report_paths(module, session_id)

    # ── JSON report ───────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ── Plain-text report ─────────────────────────────────────────
    sep  = "=" * 60
    sep2 = "-" * 60
    lines = [
        sep,
        f"  PERFLAB ANALYSIS REPORT",
        f"  Module  : {module.upper()}",
        f"  Exercise: {payload['exercise']}",
        f"  Source  : {source_filename}",
        f"  Session : {session_id}",
        f"  Recorded: {payload['recorded']}",
        sep,
        "",
        "METRICS",
        sep2,
    ]
    for m in payload["metrics"]:
        lines.append(f"  {m['label']:<30} {m['value']}")
    lines += [
        "",
        f"FORM SCORE: {payload['form_score']} / 10",
        "",
        "ISSUES FOUND",
        sep2,
    ]
    for i, issue in enumerate(payload["issues"], 1):
        lines.append(f"  {i}. {issue}")
    lines += [
        "",
        "STRENGTHS",
        sep2,
    ]
    for i, s in enumerate(payload["strengths"], 1):
        lines.append(f"  {i}. {s}")
    lines += [
        "",
        "ANALYSIS SUMMARY",
        sep2,
    ]
    score = payload['form_score']
    if isinstance(score, (int, float)):
        if score >= 9:
            lines.append("  RATING: ELITE — Outstanding movement quality. Minor refinements only.")
        elif score >= 7:
            lines.append("  RATING: GOOD — Solid technique with specific areas to improve.")
        elif score >= 5:
            lines.append("  RATING: DEVELOPING — Several form issues present; structured correction recommended.")
        else:
            lines.append("  RATING: NEEDS WORK — Multiple technique faults detected; coach-guided correction advised.")
    issues_count = len([i for i in payload["issues"] if "No major" not in i])
    strengths_count = len(payload["strengths"])
    lines.append(f"  Strengths identified : {strengths_count}")
    lines.append(f"  Issues to correct    : {issues_count}")
    lines += ["", sep, ""]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── HTML visual report ────────────────────────────────────────
    html = _build_html_report(payload, wrong_events)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return json_path, txt_path, html_path


ALLOWED_EXT = {'mp4', 'avi', 'mov', 'webm', 'jpg', 'jpeg', 'png'}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b; bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 1)

def get_landmark(lm, idx):
    p = lm[idx]; return [p.x, p.y, p.z]

def draw_angle_arc(img, b, angle_val, color=(255,200,0), radius=30, bad=False):
    h, w = img.shape[:2]
    cx, cy = int(b[0]*w), int(b[1]*h)
    if bad:
        cv2.circle(img, (cx, cy), radius + 6, (0,0,255), 2)
        cv2.circle(img, (cx, cy), radius, (0,0,255), 2)
        cv2.putText(img, f"{angle_val:.0f}°", (cx+5, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
    else:
        cv2.circle(img, (cx, cy), radius, color, 2)
        cv2.putText(img, f"{angle_val:.0f}°", (cx+5, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

class RollingMean:
    def __init__(self, n=5): self.n = n; self.buf = []
    def update(self, val):
        self.buf.append(val)
        if len(self.buf) > self.n: self.buf.pop(0)
        return round(float(np.mean(self.buf)), 1)

def process_video_or_image(path, is_video, frame_fn, output_path=None, snap_pcts=None):
    if snap_pcts is None:
        snap_pcts = [0.0, 0.25, 0.5, 0.75, 1.0]
    frame_snapshots = []
    snapshots_taken = set()
    writer = None
    if is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if fps <= 0 or np.isnan(fps): fps = 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: total = 1
        fc = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = frame_fn(frame, fc, total)
            if writer is not None: writer.write(frame)
            for pct in snap_pcts:
                idx = int(total * pct)
                if fc == idx and pct not in snapshots_taken:
                    frame_snapshots.append(frame_to_b64(frame))
                    snapshots_taken.add(pct)
            fc += 1
        cap.release()
        if writer is not None: writer.release()
        if fc == 0: raise ValueError("Video contains no readable frames")
    else:
        frame = cv2.imread(path)
        if frame is None: raise ValueError("Unable to open image file")
        frame = frame_fn(frame, 0, 1)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
        frame_snapshots.append(frame_to_b64(frame))
    return frame_snapshots


@app.route('/outputs/<module>/<path:filename>')
def serve_result(module, filename):
    return send_from_directory(module_output_dir(module), filename)


# ══════════════════════════════════════════════════════════════════
# EXERCISE 1 — PUSHUPS
# Wrong angles: elbow < 85° or > 165°; spine < 155°
# ══════════════════════════════════════════════════════════════════
def analyse_pushups(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    rep_count = 0; stage = None
    elbow_angles = []; spine_angles = []; frame_data = []
    smoother = RollingMean(5)
    wrong_events = []

    def pf(frame, fc, total):
        nonlocal rep_count, stage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ls=get_landmark(lm,11); le=get_landmark(lm,13); lw=get_landmark(lm,15)
            lh=get_landmark(lm,23); lk=get_landmark(lm,25)
            rs=get_landmark(lm,12); re=get_landmark(lm,14); rw=get_landmark(lm,16)
            l_e = calculate_angle(ls,le,lw); r_e = calculate_angle(rs,re,rw)
            lv=lm[13].visibility; rv=lm[14].visibility
            raw = (l_e*lv + r_e*rv)/(lv+rv+1e-8)
            avg_e = smoother.update(raw)
            elbow_angles.append(avg_e)
            spine = calculate_angle(ls,lh,lk); spine_angles.append(spine)
            if avg_e > 155:
                if stage != "up": stage = "up"
            elif avg_e < 115 and stage == "up":
                stage = "down"; rep_count += 1
                frame_data.append({"rep":rep_count,"elbow_angle":avg_e,"spine_alignment":spine})
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            left_bad  = l_e < 85 or l_e > 165
            right_bad = r_e < 85 or r_e > 165
            spine_bad = spine < 155
            # ── Log wrong angles ──
            if left_bad:
                wrong_events.append({"frame":fc,"joint":"left_elbow","angle_deg":l_e,
                    "threshold":"< 85 or > 165","note":"Elbow flaring — keep at 45° from torso"})
            if right_bad:
                wrong_events.append({"frame":fc,"joint":"right_elbow","angle_deg":r_e,
                    "threshold":"< 85 or > 165","note":"Elbow flaring — keep at 45° from torso"})
            if spine_bad:
                wrong_events.append({"frame":fc,"joint":"spine","angle_deg":spine,
                    "threshold":"< 155","note":"Spine sagging — engage core to keep body in straight line"})
        return frame

    snaps = process_video_or_image(path, is_video, pf, output_path=output_path)
    pose.close()
    if session_id:
        save_wrong_angle_log("pushups", session_id, source_filename, wrong_events)
    if is_video and len(elbow_angles) < 10:
        raise ValueError("No reliable pushup motion detected.")

    avg_e = round(np.mean(elbow_angles),1) if elbow_angles else 0
    min_e = round(np.min(elbow_angles),1)  if elbow_angles else 0
    avg_s = round(np.mean(spine_angles),1) if spine_angles else 0

    issues=[]; strengths=[]

    # ── Elbow angle: ideal 85–165° during movement ────────────────
    if avg_e < 75:
        issues.append(f"Severe elbow flare ({avg_e}°) — elbows are excessively wide, risking shoulder impingement; tuck them to 45° from torso")
    elif avg_e < 85:
        issues.append(f"Elbow angle too narrow ({avg_e}°) — elbows flaring slightly wide; aim for 45° tuck from torso for shoulder safety")
    elif avg_e > 165:
        issues.append(f"Elbows nearly locked out ({avg_e}°) — avoid hyper-extending at the top; maintain slight bend")
    else:
        strengths.append(f"Elbow angle well-controlled ({avg_e}°) — optimal 45° tuck maintained throughout")

    # ── Spine alignment: ideal ≥ 155° (straight plank line) ──────
    if avg_s < 130:
        issues.append(f"Severe spine sag ({avg_s}°) — hips are dropping significantly; brace core and glutes to restore plank line")
    elif avg_s < 155:
        issues.append(f"Spine sagging ({avg_s}°) — engage core and keep hips level; body should form a straight line head-to-heel")
    else:
        strengths.append(f"Excellent spine alignment ({avg_s}°) — straight plank position held consistently")

    # ── Pushup depth: min elbow angle ────────────────────────────
    if rep_count > 0:
        if min_e > 130:
            issues.append(f"Very shallow depth (min elbow {min_e}°) — only partial ROM achieved; lower chest to within 2–3 cm of floor")
        elif min_e > 115:
            issues.append(f"Insufficient depth (min elbow {min_e}°) — aim for 90° or below at the bottom to maximise muscle recruitment")
        elif min_e < 70:
            strengths.append(f"Exceptional depth achieved (min elbow {min_e}°) — full range of motion with chest near floor")
        else:
            strengths.append(f"Good pushup depth (min elbow {min_e}°) — chest reaching near the floor for full ROM")
    else:
        strengths.append("Ready for movement — position looks correct")

    if not issues: issues = ["No major form issues detected"]
    form_score = max(4, 10 - len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Pushups","rep_count":rep_count,"avg_elbow_angle":avg_e,
            "min_elbow_angle":min_e,"spine_alignment":avg_s,"form_score":form_score,
            "issues":issues,"strengths":strengths,"per_rep":frame_data,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Reps Counted","value":str(rep_count)},
                {"label":"Avg Elbow Angle","value":f"{avg_e}°"},
                {"label":"Min Elbow (Depth)","value":f"{min_e}°"},
                {"label":"Spine Alignment","value":f"{avg_s}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 2 — SQUAT
# Wrong angles: knee < 70° or > 165°; hip < 65°
# ══════════════════════════════════════════════════════════════════
def analyse_squat(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    rep_count=0; stage=None
    knee_angles=[]; hip_angles=[]; ankle_angles=[]; frame_data=[]
    smoother=RollingMean(5)
    wrong_events=[]

    def pf(frame, fc, total):
        nonlocal rep_count, stage
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11); lf=get_landmark(lm,31)
            rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            l_k=calculate_angle(lh,lk,la); r_k=calculate_angle(rh,rk,ra)
            lv=lm[25].visibility; rv=lm[26].visibility
            raw=(l_k*lv + r_k*rv)/(lv+rv+1e-8)
            avg_k=smoother.update(raw); knee_angles.append(avg_k)
            hip=calculate_angle(ls,lh,lk); hip_angles.append(hip)
            ank=calculate_angle(lk,la,lf); ankle_angles.append(ank)
            if avg_k > 155:
                if stage != "up": stage = "up"
            elif avg_k < 115 and stage == "up":
                stage="down"; rep_count+=1
                frame_data.append({"rep":rep_count,"knee_angle":avg_k,"hip_angle":hip})
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_k   = l_k < 70 or l_k > 165 or r_k < 70 or r_k > 165
            bad_hip = hip < 65
            if bad_k:
                wrong_events.append({"frame":fc,"joint":"knee","angle_deg":min(l_k,r_k),
                    "threshold":"< 70 or > 165","note":"Knee tracking issue — check alignment over toes"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip,
                    "threshold":"< 65","note":"Excessive forward lean — keep chest upright and core braced"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path); pose.close()
    if session_id:
        save_wrong_angle_log("squat", session_id, source_filename, wrong_events)
    avg_k=round(np.mean(knee_angles),1) if knee_angles else 0
    min_k=round(np.min(knee_angles),1)  if knee_angles else 0
    avg_h=round(np.mean(hip_angles),1)  if hip_angles else 0
    avg_a=round(np.mean(ankle_angles),1) if ankle_angles else 0

    issues=[]; strengths=[]

    # ── Squat depth: min knee angle ───────────────────────────────
    if min_k > 130:
        issues.append(f"Very shallow squat ({min_k}° min knee) — only quarter-squat depth; aim for at least parallel (≈90°)")
    elif min_k > 110:
        issues.append(f"Squat depth insufficient ({min_k}° min knee) — aim for thigh parallel to floor (≈90° knee angle)")
    elif min_k < 70:
        strengths.append(f"Deep squat achieved ({min_k}° min knee) — full depth below parallel showing excellent mobility")
    else:
        strengths.append(f"Good squat depth ({min_k}° min knee) — parallel or below achieved for full ROM")

    # ── Hip / trunk angle: ideal ≥ 65° ───────────────────────────
    if avg_h < 45:
        issues.append(f"Severe forward lean ({avg_h}° hip) — trunk is collapsing; brace core, open chest, and consider heel elevation")
    elif avg_h < 65:
        issues.append(f"Excessive forward lean ({avg_h}° hip) — keep chest upright and core braced throughout the squat")
    elif avg_h >= 80:
        strengths.append(f"Excellent upright torso ({avg_h}° hip angle) — minimal forward lean throughout")
    else:
        strengths.append(f"Good trunk position ({avg_h}° hip angle) — upright posture maintained")

    # ── Ankle dorsiflexion: ideal ≥ 55° ──────────────────────────
    if avg_a < 40:
        issues.append(f"Very restricted ankle mobility ({avg_a}°) — significantly limiting depth; daily calf and ankle dorsiflexion stretches needed")
    elif avg_a < 55:
        issues.append(f"Ankle mobility limiting depth ({avg_a}°) — elevate heels or work on daily dorsiflexion mobility exercises")
    else:
        strengths.append(f"Sufficient ankle dorsiflexion ({avg_a}°) — mobility is not limiting squat depth")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Squat","rep_count":rep_count,"avg_knee_angle":avg_k,
            "min_knee_angle":min_k,"avg_hip_angle":avg_h,"avg_ankle_angle":avg_a,
            "form_score":form_score,"issues":issues,"strengths":strengths,
            "per_rep":frame_data,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Reps Counted","value":str(rep_count)},
                {"label":"Avg Knee Angle","value":f"{avg_k}°"},
                {"label":"Min Knee (Depth)","value":f"{min_k}°"},
                {"label":"Hip Angle","value":f"{avg_h}°"},
                {"label":"Ankle Angle","value":f"{avg_a}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 3 — SINGLE LEG SQUAT
# Wrong angles: knee < 70° or > 165°; hip < 55°; pelvic drop > 5
# ══════════════════════════════════════════════════════════════════
def analyse_single_leg_squat(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    rep_count=0; stage=None
    knee_angles=[]; hip_angles=[]; pelvic_drops=[]; frame_data=[]
    smoother=RollingMean(5)
    wrong_events=[]

    def pf(frame, fc, total):
        nonlocal rep_count, stage
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11)
            raw_k=calculate_angle(lh,lk,la)
            hip=calculate_angle(ls,lh,lk)
            pelvic=round(abs(lm[23].y - lm[24].y)*100, 2)
            knee_k=smoother.update(raw_k)
            knee_angles.append(knee_k); hip_angles.append(hip); pelvic_drops.append(pelvic)
            if knee_k > 155:
                if stage != "up": stage="up"
            elif knee_k < 115 and stage == "up":
                stage="down"; rep_count+=1
                frame_data.append({"rep":rep_count,"knee_angle":knee_k,"hip_angle":hip,"pelvic_drop":pelvic})
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_k      = knee_k < 70 or knee_k > 165
            bad_hip    = hip < 55
            bad_pelvic = pelvic > 5
            if bad_k:
                wrong_events.append({"frame":fc,"joint":"knee","angle_deg":knee_k,
                    "threshold":"< 70 or > 165","note":"Knee caving — strengthen hip abductors and VMO"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip,
                    "threshold":"< 55","note":"Excessive forward trunk lean — keep chest upright"})
            if bad_pelvic:
                wrong_events.append({"frame":fc,"joint":"pelvis","angle_deg":pelvic,
                    "threshold":"> 5 (drop index)","note":"Pelvic drop — strengthen glute medius"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path); pose.close()
    if session_id:
        save_wrong_angle_log("single_leg_squat", session_id, source_filename, wrong_events)
    if is_video and len(knee_angles) < 10:
        raise ValueError("No reliable single-leg squat motion detected.")
    avg_k=round(np.mean(knee_angles),1) if knee_angles else 0
    min_k=round(np.min(knee_angles),1)  if knee_angles else 0
    avg_h=round(np.mean(hip_angles),1)  if hip_angles else 0
    avg_p=round(np.mean(pelvic_drops),2) if pelvic_drops else 0

    issues=[]; strengths=[]

    # ── Single-leg squat depth ────────────────────────────────────
    if min_k > 130:
        issues.append(f"Very shallow single-leg squat ({min_k}° min knee) — only minimal bend; aim for thigh parallel to floor (≈90°)")
    elif min_k > 110:
        issues.append(f"Insufficient depth ({min_k}° min knee) — aim for thigh parallel to floor (≈90° knee angle)")
    elif min_k < 75:
        strengths.append(f"Excellent single-leg squat depth ({min_k}°) — strong knee flexion showing good control and mobility")
    else:
        strengths.append(f"Good single-leg squat depth ({min_k}°) — adequate ROM achieved on the standing leg")

    # ── Pelvic drop: ideal ≤ 5 index units ───────────────────────
    if avg_p > 12:
        issues.append(f"Severe pelvic drop ({avg_p:.1f}) — major glute medius weakness; prioritise hip abductor strengthening exercises immediately")
    elif avg_p > 5:
        issues.append(f"Pelvic drop detected ({avg_p:.1f}) — strengthen hip abductors (glute medius, TFL) with banded clamshells and lateral walks")
    elif avg_p < 2:
        strengths.append(f"Excellent pelvic stability ({avg_p:.1f} drop index) — outstanding hip abductor control during the movement")
    else:
        strengths.append(f"Good pelvic control ({avg_p:.1f} drop index) — stable pelvis with minimal lateral shift")

    # ── Trunk lean: ideal ≥ 55° hip angle ────────────────────────
    if avg_h < 40:
        issues.append(f"Severe trunk lean on single leg ({avg_h}°) — use arms for counter-balance and focus on controlled tempo")
    elif avg_h < 55:
        issues.append(f"Excessive forward trunk lean ({avg_h}°) — keep chest upright and core engaged throughout")
    else:
        strengths.append(f"Upright torso maintained on standing leg ({avg_h}° hip angle) — good postural control")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Single Leg Squat","rep_count":rep_count,"avg_knee_angle":avg_k,
            "min_knee_angle":min_k,"avg_hip_angle":avg_h,"avg_pelvic_drop":avg_p,
            "form_score":form_score,"issues":issues,"strengths":strengths,
            "per_rep":frame_data,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Reps Counted","value":str(rep_count)},
                {"label":"Avg Knee Angle","value":f"{avg_k}°"},
                {"label":"Min Knee (Depth)","value":f"{min_k}°"},
                {"label":"Hip Angle","value":f"{avg_h}°"},
                {"label":"Pelvic Drop Index","value":f"{avg_p}"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 4 — BROAD JUMP
# Wrong angles: knee > 145° at takeoff; hip < 155°
# ══════════════════════════════════════════════════════════════════
def analyse_broad_jump(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    takeoff_angles=[]; landing_angles=[]; hip_extensions=[]
    wrong_events=[]

    def pf(frame, fc, total):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11); rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            knee_ang=(calculate_angle(lh,lk,la)+calculate_angle(rh,rk,ra))/2
            hip_ang=calculate_angle(ls,lh,lk); hip_extensions.append(hip_ang)
            pct=fc/max(total,1)
            if pct<0.25: takeoff_angles.append(knee_ang); phase="Pre-Jump"
            elif pct<0.45: takeoff_angles.append(knee_ang); phase="Takeoff"
            elif pct<0.6: phase="Flight"
            elif pct<0.8: landing_angles.append(knee_ang); phase="Landing"
            else: landing_angles.append(knee_ang); phase="Recovery"
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_k   = knee_ang > 145
            bad_hip = hip_ang < 155
            if bad_k:
                wrong_events.append({"frame":fc,"joint":"knee","angle_deg":round(knee_ang,1),
                    "threshold":"> 145","note":"Insufficient takeoff crouch — deeper bend generates more power"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip_ang,
                    "threshold":"< 155","note":"Limited hip extension — drive hips fully forward"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path,snap_pcts=[0.1,0.3,0.5,0.7,0.9])
    pose.close()
    if session_id:
        save_wrong_angle_log("broad_jump", session_id, source_filename, wrong_events)
    if is_video and len(hip_extensions) < 10:
        raise ValueError("No reliable broad jump motion detected.")
    avg_to=round(np.min(takeoff_angles),1) if takeoff_angles else 0
    avg_la=round(np.mean(landing_angles),1) if landing_angles else 0
    avg_hi=round(np.max(hip_extensions),1)  if hip_extensions else 0

    issues=[]; strengths=[]

    # ── Takeoff knee bend: ideal ≤ 120° (deep crouch) ────────────
    if avg_to > 160:
        issues.append(f"Almost no takeoff crouch ({avg_to}°) — legs nearly straight at takeoff; bend knees to ≈100–120° to load the stretch-shortening cycle")
    elif avg_to > 145:
        issues.append(f"Insufficient takeoff crouch ({avg_to}°) — deeper knee bend generates more horizontal power and distance")
    elif avg_to < 100:
        strengths.append(f"Excellent takeoff depth ({avg_to}°) — deep pre-jump crouch maximising elastic energy storage")
    else:
        strengths.append(f"Good pre-jump knee bend ({avg_to}°) — effective crouch for power generation")

    # ── Hip extension: ideal ≥ 155° ───────────────────────────────
    if avg_hi < 130:
        issues.append(f"Very limited hip extension ({avg_hi}°) — hips are not driving fully; focus on glute activation and full hip thrust at takeoff")
    elif avg_hi < 155:
        issues.append(f"Limited hip extension ({avg_hi}°) — drive hips fully forward and extend ankles for maximum distance")
    elif avg_hi >= 170:
        strengths.append(f"Powerful full hip extension ({avg_hi}°) — hips driving through complete range for maximum propulsion")
    else:
        strengths.append(f"Strong hip extension ({avg_hi}°) — good glute activation generating propulsive force")

    # ── Landing mechanics: ideal ≥ 120° ──────────────────────────
    if avg_la < 90:
        issues.append(f"Very stiff landing ({avg_la}°) — high impact load on joints; aggressively increase knee and hip flexion at ground contact")
    elif avg_la < 120:
        issues.append(f"Stiff landing ({avg_la}°) — increase knee flexion on contact to absorb impact safely and protect joints")
    else:
        strengths.append(f"Controlled, safe landing mechanics ({avg_la}°) — good shock absorption through knees and hips")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Broad Jump","avg_takeoff_angle":avg_to,"avg_landing_angle":avg_la,
            "avg_hip_extension":avg_hi,"form_score":form_score,"issues":issues,"strengths":strengths,
            "snapshots":snaps,"wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Takeoff Knee Bend","value":f"{avg_to}°"},
                {"label":"Landing Knee Angle","value":f"{avg_la}°"},
                {"label":"Peak Hip Extension","value":f"{avg_hi}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 5 — WALKING LUNGES
# Wrong angles: front_knee > 120°; trunk < 160°
# ══════════════════════════════════════════════════════════════════
def analyse_walking_lunges(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    rep_count=0; stage=None
    front_knee_angles=[]; trunk_angles=[]; frame_data=[]
    sm_l=RollingMean(5); sm_r=RollingMean(5)
    wrong_events=[]

    def pf(frame, fc, total):
        nonlocal rep_count, stage
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            ls=get_landmark(lm,11); rs=get_landmark(lm,12)
            l_k=sm_l.update(calculate_angle(lh,lk,la))
            r_k=sm_r.update(calculate_angle(rh,rk,ra))
            front_knee=min(l_k,r_k); back_knee=max(l_k,r_k)
            front_knee_angles.append(front_knee)
            mid_sh=[(ls[0]+rs[0])/2,(ls[1]+rs[1])/2,(ls[2]+rs[2])/2]
            mid_hi=[(lh[0]+rh[0])/2,(lh[1]+rh[1])/2,(lh[2]+rh[2])/2]
            trunk=calculate_angle([mid_sh[0],mid_sh[1]-0.1,mid_sh[2]], mid_sh, mid_hi)
            trunk_angles.append(trunk)
            if front_knee > 155:
                if stage != "up": stage="up"
            elif front_knee < 120 and stage == "up":
                stage="down"; rep_count+=1
                frame_data.append({"rep":rep_count,"front_knee":front_knee,"back_knee":back_knee})
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_front = front_knee > 120
            bad_trunk = trunk < 160
            if bad_front:
                wrong_events.append({"frame":fc,"joint":"front_knee","angle_deg":front_knee,
                    "threshold":"> 120","note":"Lunge depth insufficient — front knee should reach 90°"})
            if bad_trunk:
                wrong_events.append({"frame":fc,"joint":"trunk","angle_deg":trunk,
                    "threshold":"< 160","note":"Trunk leaning forward — keep chest tall, core braced"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path); pose.close()
    if session_id:
        save_wrong_angle_log("walking_lunges", session_id, source_filename, wrong_events)
    if is_video and len(front_knee_angles) < 10:
        raise ValueError("No reliable walking lunge motion detected.")
    avg_fk=round(np.mean(front_knee_angles),1) if front_knee_angles else 0
    min_fk=round(np.min(front_knee_angles),1)  if front_knee_angles else 0
    avg_tr=round(np.mean(trunk_angles),1)       if trunk_angles else 0

    issues=[]; strengths=[]

    # ── Lunge depth: min front knee angle ────────────────────────
    if min_fk > 130:
        issues.append(f"Very shallow lunge ({min_fk}° min front knee) — barely bending the front leg; aim for 90° for full ROM and muscle activation")
    elif min_fk > 100:
        issues.append(f"Lunge depth insufficient ({min_fk}° min front knee) — front knee should reach 90° for full range of motion")
    elif min_fk < 80:
        strengths.append(f"Excellent lunge depth ({min_fk}° min front knee) — deep lunge maximising quad, glute, and hip flexor recruitment")
    else:
        strengths.append(f"Good lunge depth ({min_fk}° min front knee) — front knee reaching or below 90° for full ROM")

    # ── Trunk angle: ideal ≥ 160° ──────────────────────────────
    if avg_tr < 140:
        issues.append(f"Severe trunk lean during lunges ({avg_tr}°) — significantly hunching forward; keep chest tall and shoulders back")
    elif avg_tr < 160:
        issues.append(f"Trunk leaning forward ({avg_tr}°) — keep chest tall and core braced with every lunge step")
    elif avg_tr >= 170:
        strengths.append(f"Excellent upright posture ({avg_tr}°) — very controlled trunk position throughout all lunge steps")
    else:
        strengths.append(f"Good trunk angle ({avg_tr}°) — upright posture maintained through each lunge step")

    # ── Rep count check ───────────────────────────────────────────
    if rep_count == 0:
        issues.append("No full lunge reps detected — ensure the full up/down movement is clearly visible in the recording")
    elif rep_count >= 10:
        strengths.append(f"High rep volume completed ({rep_count} reps) — good muscular endurance demonstrated")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i and "No full" not in i])*2)

    return {"exercise":"Walking Lunges","rep_count":rep_count,"avg_front_knee":avg_fk,
            "min_front_knee":min_fk,"avg_trunk_angle":avg_tr,"form_score":form_score,
            "issues":issues,"strengths":strengths,"per_rep":frame_data,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Reps Counted","value":str(rep_count)},
                {"label":"Avg Front Knee","value":f"{avg_fk}°"},
                {"label":"Min Front Knee (Depth)","value":f"{min_fk}°"},
                {"label":"Trunk Angle","value":f"{avg_tr}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 6 — OVERHEAD SQUAT WITH STICK
# Wrong angles: knee < 70° or > 165°; hip < 60°; wrist forward
# ══════════════════════════════════════════════════════════════════
def analyse_squat_with_stick(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    rep_count=0; stage=None
    knee_angles=[]; hip_angles=[]; ankle_angles=[]; wrist_heights=[]; frame_data=[]
    smoother=RollingMean(5)
    wrong_events=[]

    def pf(frame, fc, total):
        nonlocal rep_count, stage
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11); lf=get_landmark(lm,31)
            rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            l_k=calculate_angle(lh,lk,la); r_k=calculate_angle(rh,rk,ra)
            avg_k=smoother.update((l_k+r_k)/2); knee_angles.append(avg_k)
            hip=calculate_angle(ls,lh,lk); hip_angles.append(hip)
            ank=calculate_angle(lk,la,lf); ankle_angles.append(ank)
            wrist_rel=lm[15].y - lm[11].y; wrist_heights.append(wrist_rel)
            if avg_k > 155:
                if stage != "up": stage="up"
            elif avg_k < 115 and stage == "up":
                stage="down"; rep_count+=1
                frame_data.append({"rep":rep_count,"knee_angle":avg_k,"hip_angle":hip,"wrist_rel":round(wrist_rel,3)})
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_k     = l_k < 70 or l_k > 165 or r_k < 70 or r_k > 165
            bad_hip   = hip < 60
            bad_wrist = wrist_rel > -0.02
            if bad_k:
                wrong_events.append({"frame":fc,"joint":"knee","angle_deg":min(l_k,r_k),
                    "threshold":"< 70 or > 165","note":"Knee tracking off — check alignment overhead"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip,
                    "threshold":"< 60","note":"Excessive forward lean — stick should stay vertical overhead"})
            if bad_wrist:
                wrong_events.append({"frame":fc,"joint":"wrist_overhead","angle_deg":round(wrist_rel,3),
                    "threshold":"> -0.02 (relative y)","note":"Arms drifting forward — maintain strict vertical bar path"})
            overhead_ok="OK" if wrist_rel < -0.02 else "Fwd"
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path); pose.close()
    if session_id:
        save_wrong_angle_log("squat_with_stick", session_id, source_filename, wrong_events)
    if is_video and len(knee_angles) < 10:
        raise ValueError("No reliable overhead squat motion detected.")
    avg_k=round(np.mean(knee_angles),1)  if knee_angles else 0
    min_k=round(np.min(knee_angles),1)   if knee_angles else 0
    avg_h=round(np.mean(hip_angles),1)   if hip_angles else 0
    avg_a=round(np.mean(ankle_angles),1) if ankle_angles else 0
    avg_w=round(np.mean(wrist_heights),3) if wrist_heights else 0

    issues=[]; strengths=[]

    # ── Overhead squat depth: min knee angle ──────────────────────
    if min_k > 130:
        issues.append(f"Very shallow overhead squat ({min_k}° min knee) — only partial depth; aim to reach parallel with stick overhead")
    elif min_k > 110:
        issues.append(f"Squat depth insufficient ({min_k}° min knee) — aim to reach parallel or below with stick maintained overhead")
    elif min_k < 75:
        strengths.append(f"Excellent overhead squat depth ({min_k}°) — full depth achieved while maintaining overhead position")
    else:
        strengths.append(f"Good overhead squat depth ({min_k}°) — parallel or below achieved with overhead control")

    # ── Hip/trunk angle: ideal ≥ 60° ─────────────────────────────
    if avg_h < 45:
        issues.append(f"Severe forward lean ({avg_h}° hip) — stick is likely drifting forward; open the chest and work on thoracic mobility")
    elif avg_h < 60:
        issues.append(f"Excessive forward lean ({avg_h}° hip) — stick/arms drifting forward; keep torso as upright as possible overhead")
    elif avg_h >= 75:
        strengths.append(f"Excellent upright trunk ({avg_h}° hip) — minimal forward lean with stick maintained overhead")
    else:
        strengths.append(f"Good trunk angle ({avg_h}° hip) — upright position with stick held overhead")

    # ── Ankle dorsiflexion ────────────────────────────────────────
    if avg_a < 40:
        issues.append(f"Severely restricted ankle mobility ({avg_a}°) — major limitation for overhead squat depth; daily mobility work essential")
    elif avg_a < 55:
        issues.append(f"Ankle dorsiflexion limiting depth ({avg_a}°) — mobilise calves and ankles daily to improve overhead squat")
    else:
        strengths.append(f"Sufficient ankle mobility ({avg_a}°) — not limiting overhead squat depth")

    # ── Overhead wrist position ───────────────────────────────────
    if avg_w > 0.05:
        issues.append(f"Arms significantly forward of overhead ({avg_w:.3f} rel) — major overhead mobility restriction; work on shoulder flexion and thoracic extension")
    elif avg_w > -0.02:
        issues.append(f"Arms drifting slightly forward ({avg_w:.3f} rel) — maintain strict vertical bar path overhead throughout the squat")
    else:
        strengths.append(f"Overhead arm position held correctly ({avg_w:.3f} rel) — stick stays vertical throughout the movement")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Overhead Squat with Stick","rep_count":rep_count,"avg_knee_angle":avg_k,
            "min_knee_angle":min_k,"avg_hip_angle":avg_h,"avg_ankle_angle":avg_a,
            "form_score":form_score,"issues":issues,"strengths":strengths,
            "per_rep":frame_data,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Reps Counted","value":str(rep_count)},
                {"label":"Avg Knee Angle","value":f"{avg_k}°"},
                {"label":"Min Knee (Depth)","value":f"{min_k}°"},
                {"label":"Hip Angle","value":f"{avg_h}°"},
                {"label":"Ankle Angle","value":f"{avg_a}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 7 — VERTICAL JUMP
# Wrong angles: avg_k > 145° at takeoff; hip < 140°
# ══════════════════════════════════════════════════════════════════
def analyse_vertical_jump(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    hip_heights=[]; takeoff_knees=[]; landing_knees=[]; hip_angles=[]
    wrong_events=[]

    def pf(frame, fc, total):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11); rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            l_k=calculate_angle(lh,lk,la); r_k=calculate_angle(rh,rk,ra)
            avg_k=(l_k+r_k)/2
            hip=calculate_angle(ls,lh,lk); hip_angles.append(hip)
            hip_heights.append(lm[23].y)
            pct=fc/max(total,1)
            if pct<0.4: takeoff_knees.append(avg_k); phase="Takeoff"
            elif pct<0.6: phase="Peak Height"
            else: landing_knees.append(avg_k); phase="Landing"
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_k   = avg_k > 145
            bad_hip = hip < 140
            if bad_k:
                wrong_events.append({"frame":fc,"joint":"knee","angle_deg":round(avg_k,1),
                    "threshold":"> 145","note":"Insufficient takeoff knee bend — deeper crouch stores more energy"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip,
                    "threshold":"< 140","note":"Limited hip extension — fully extend hips to maximise jump height"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path,snap_pcts=[0.1,0.3,0.5,0.7,0.9])
    pose.close()
    if session_id:
        save_wrong_angle_log("vertical_jump", session_id, source_filename, wrong_events)
    if is_video and len(hip_heights) < 10:
        raise ValueError("No reliable vertical jump motion detected.")
    avg_to=round(np.min(takeoff_knees),1)  if takeoff_knees else 0
    avg_la=round(np.mean(landing_knees),1) if landing_knees else 0
    avg_h =round(np.mean(hip_angles),1)    if hip_angles else 0
    h_range=round((max(hip_heights)-min(hip_heights))*100,1) if hip_heights else 0

    issues=[]; strengths=[]

    # ── Takeoff knee bend: ideal ≤ 120° ──────────────────────────
    if avg_to > 160:
        issues.append(f"Almost no takeoff crouch ({avg_to}°) — very little knee bend; aim for 100–120° to store elastic energy in the stretch-shortening cycle")
    elif avg_to > 145:
        issues.append(f"Insufficient takeoff crouch ({avg_to}°) — deeper knee bend stores more elastic energy for greater jump height")
    elif avg_to < 90:
        strengths.append(f"Excellent pre-jump crouch ({avg_to}°) — deep knee bend maximising energy storage for vertical power")
    else:
        strengths.append(f"Good pre-jump knee bend ({avg_to}°) — effective crouch for elastic energy storage")

    # ── Hip extension: ideal ≥ 140° ───────────────────────────────
    if avg_h < 110:
        issues.append(f"Very limited hip extension ({avg_h}°) — hips not fully extending; drive hips, knees, and ankles to full extension at takeoff")
    elif avg_h < 140:
        issues.append(f"Limited hip extension ({avg_h}°) — fully extend hips and ankles to maximise jump height")
    elif avg_h >= 160:
        strengths.append(f"Powerful full hip extension ({avg_h}°) — hips driving through complete range for maximum height")
    else:
        strengths.append(f"Good hip extension contributing to height ({avg_h}°)")

    # ── Landing absorption: ideal ≥ 115° ──────────────────────────
    if avg_la < 90:
        issues.append(f"Very hard landing ({avg_la}°) — dangerously stiff impact; deeply flex knees and hips immediately on contact to protect joints")
    elif avg_la < 115:
        issues.append(f"Hard landing detected ({avg_la}°) — bend knees and hips more on ground contact to reduce joint load and injury risk")
    elif avg_la >= 140:
        strengths.append(f"Excellent soft landing ({avg_la}°) — exceptional shock absorption through knees and hips")
    else:
        strengths.append(f"Controlled, safe landing ({avg_la}°) — good shock absorption protecting joints")

    # ── Height index ──────────────────────────────────────────────
    if h_range > 20:
        strengths.append(f"Impressive height index ({h_range}) — significant hip displacement indicating good vertical power output")
    elif h_range < 8:
        issues.append(f"Low height index ({h_range}) — limited vertical displacement detected; work on explosive triple extension (hip, knee, ankle)")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Vertical Jump","avg_takeoff_angle":avg_to,"avg_landing_angle":avg_la,
            "avg_hip_angle":avg_h,"relative_height_index":h_range,"form_score":form_score,
            "issues":issues,"strengths":strengths,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Takeoff Knee Bend","value":f"{avg_to}°"},
                {"label":"Landing Knee Angle","value":f"{avg_la}°"},
                {"label":"Hip Extension","value":f"{avg_h}°"},
                {"label":"Height Index","value":f"{h_range}"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# EXERCISE 8 — SPEED 20M DASH
# Wrong angles: knee drive > 120°; hip < 145°; stride asym > 20°
# ══════════════════════════════════════════════════════════════════
def analyse_speed_20m(path, is_video, output_path=None, session_id=None, source_filename=""):
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    knee_drives=[]; hip_angles=[]; stride_asym=[]; trunk_angles=[]
    wrong_events=[]

    def pf(frame, fc, total):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            lh=get_landmark(lm,23); lk=get_landmark(lm,25); la=get_landmark(lm,27)
            ls=get_landmark(lm,11); rh=get_landmark(lm,24); rk=get_landmark(lm,26); ra=get_landmark(lm,28)
            nose=get_landmark(lm,0)
            l_k=calculate_angle(lh,lk,la); r_k=calculate_angle(rh,rk,ra)
            drive=min(l_k,r_k); knee_drives.append(drive)
            hip=calculate_angle(ls,lh,lk); hip_angles.append(hip)
            asym=abs(l_k-r_k); stride_asym.append(asym)
            trunk=calculate_angle(nose,ls,lh); trunk_angles.append(trunk)
            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            bad_drive = drive > 120
            bad_hip   = hip < 145
            bad_asym  = asym > 20
            if bad_drive:
                wrong_events.append({"frame":fc,"joint":"knee_drive","angle_deg":drive,
                    "threshold":"> 120","note":"Low knee drive — higher lift increases stride frequency"})
            if bad_hip:
                wrong_events.append({"frame":fc,"joint":"hip","angle_deg":hip,
                    "threshold":"< 145","note":"Insufficient hip extension — drive through full range"})
            if bad_asym:
                wrong_events.append({"frame":fc,"joint":"stride_asymmetry","angle_deg":round(asym,1),
                    "threshold":"> 20","note":"Stride asymmetry — address bilateral strength imbalance"})
        return frame

    snaps=process_video_or_image(path,is_video,pf,output_path=output_path,snap_pcts=[0.1,0.3,0.5,0.7,0.9])
    pose.close()
    if session_id:
        save_wrong_angle_log("speed_20m", session_id, source_filename, wrong_events)
    if is_video and len(knee_drives) < 10:
        raise ValueError("No reliable sprint motion detected.")
    avg_d=round(np.mean(knee_drives),1)   if knee_drives else 0
    avg_h=round(np.mean(hip_angles),1)    if hip_angles else 0
    avg_a=round(np.mean(stride_asym),1)   if stride_asym else 0
    avg_t=round(np.mean(trunk_angles),1)  if trunk_angles else 0

    issues=[]; strengths=[]

    # ── Knee drive: ideal avg_d ≤ 100° ───────────────────────────
    if avg_d > 140:
        issues.append(f"Very low knee drive ({avg_d}°) — knees barely lifting; significantly increase knee drive height to improve stride frequency and power")
    elif avg_d > 120:
        issues.append(f"Insufficient knee drive ({avg_d}°) — higher knee lift increases stride frequency and propulsive power output")
    elif avg_d < 80:
        strengths.append(f"Excellent knee drive ({avg_d}°) — high knee lift generating strong stride frequency and ground contact forces")
    else:
        strengths.append(f"Strong knee drive ({avg_d}°) — good knee lift contributing to sprint speed")

    # ── Hip extension: ideal avg_h ≥ 145° ────────────────────────
    if avg_h < 120:
        issues.append(f"Severely insufficient hip extension ({avg_h}°) — hips are not driving through; focus on glute activation and full hip extension drills")
    elif avg_h < 145:
        issues.append(f"Insufficient hip extension ({avg_h}°) — drive through the full hip range for more propulsion per stride")
    elif avg_h >= 165:
        strengths.append(f"Powerful hip extension ({avg_h}°) — fully driving hips through range maximising propulsive force per stride")
    else:
        strengths.append(f"Good hip extension ({avg_h}°) — generating strong propulsive force each stride")

    # ── Stride asymmetry: ideal avg_a ≤ 10° ──────────────────────
    if avg_a > 30:
        issues.append(f"Severe stride asymmetry ({avg_a}°) — major bilateral imbalance; assess for strength or flexibility deficits and consider single-leg work")
    elif avg_a > 20:
        issues.append(f"Stride asymmetry ({avg_a}°) — address bilateral strength or flexibility imbalance with unilateral exercises")
    elif avg_a < 8:
        strengths.append(f"Excellent stride symmetry ({avg_a}°) — outstanding bilateral balance in mechanics")
    else:
        strengths.append(f"Symmetrical stride pattern ({avg_a}°) — balanced bilateral mechanics")

    # ── Trunk stability: ideal avg_t ≥ 130° ──────────────────────
    if avg_t < 110:
        issues.append(f"Severe trunk instability ({avg_t}°) — significant postural breakdown at speed; brace core and use aggressive arm drive to stabilise")
    elif avg_t < 130:
        issues.append(f"Trunk instability ({avg_t}°) — brace core and drive arms aggressively to stabilise posture at sprint speed")
    elif avg_t >= 150:
        strengths.append(f"Excellent trunk stability ({avg_t}°) — very controlled forward lean with efficient sprint posture")
    else:
        strengths.append(f"Stable trunk position ({avg_t}°) — controlled forward lean with good sprint mechanics")

    if not issues: issues=["No major form issues detected"]
    form_score=max(4, 10-len([i for i in issues if "No major" not in i])*2)

    return {"exercise":"Speed 20m Dash","avg_knee_drive":avg_d,"avg_hip_angle":avg_h,
            "stride_asymmetry":avg_a,"trunk_angle":avg_t,"form_score":form_score,
            "issues":issues,"strengths":strengths,"snapshots":snaps,
            "wrong_angle_count": len(wrong_events),
            "_wrong_events": wrong_events,
            "metrics":[
                {"label":"Avg Knee Drive","value":f"{avg_d}°"},
                {"label":"Hip Extension","value":f"{avg_h}°"},
                {"label":"Stride Asymmetry","value":f"{avg_a}°"},
                {"label":"Trunk Angle","value":f"{avg_t}°"},
                {"label":"Form Score","value":f"{form_score}/10"},
                {"label":"Wrong-Angle Frames","value":str(len(wrong_events))},
            ]}


# ══════════════════════════════════════════════════════════════════
# ANALYSER REGISTRY
# ══════════════════════════════════════════════════════════════════
ANALYSERS = {
    "pushups":           analyse_pushups,
    "squat":             analyse_squat,
    "single_leg_squat":  analyse_single_leg_squat,
    "broad_jump":        analyse_broad_jump,
    "walking_lunges":    analyse_walking_lunges,
    "squat_with_stick":  analyse_squat_with_stick,
    "vertical_jump":     analyse_vertical_jump,
    "speed_20m":         analyse_speed_20m,
}


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "Uploaded file is too large. Maximum size is 200MB."}), 413

@app.route('/analyse', methods=['POST'])
def analyse():
    exercise = request.form.get('exercise')
    if exercise not in ANALYSERS:
        return jsonify({"error": "Unknown exercise"}), 400
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    session_id   = uuid4().hex
    fname        = secure_filename(file.filename)
    ext          = fname.rsplit('.', 1)[1].lower()
    is_video     = ext in {'mp4', 'avi', 'mov', 'webm'}

    # ── Save to module-specific input folder ──────────────────
    input_dir  = module_input_dir(exercise)
    input_path = os.path.join(input_dir, f"{session_id}_{fname}")
    file.save(input_path)

    # ── Build module-specific output path ─────────────────────
    output_dir  = module_output_dir(exercise)
    out_ext     = 'mp4' if is_video else ext
    output_name = f"annotated_{session_id}.{out_ext}"
    output_path = os.path.join(output_dir, output_name)
    output_url  = url_for('serve_result', module=exercise, filename=output_name)

    try:
        result = ANALYSERS[exercise](
            input_path, is_video,
            output_path=output_path,
            session_id=session_id,
            source_filename=fname,
        )
        result['output_url']  = output_url
        result['media_type']  = 'video' if is_video else 'image'
        result['session_id']  = session_id

        # Surface the wrong-angle log URL
        log_name = f"wrong_angles_{session_id}.json"
        result['wrong_angle_log_url'] = url_for(
            'serve_result', module=exercise, filename=log_name)

        # ── Save per-module session report ────────────────────────
        wrong_events = result.pop("_wrong_events", [])
        save_report(exercise, session_id, fname, result, wrong_events)

        # Expose report download URLs
        report_json_name = f"report_{session_id}.json"
        report_txt_name  = f"report_{session_id}.txt"
        report_html_name = f"report_{session_id}.html"
        result['report_json_url'] = url_for('serve_result', module=exercise, filename=report_json_name)
        result['report_txt_url']  = url_for('serve_result', module=exercise, filename=report_txt_name)
        result['report_html_url'] = url_for('serve_result', module=exercise, filename=report_html_name)

        return jsonify(result)

    except Exception as e:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({
            "error": str(e),
            "remark": "Upload a valid video or image with the movement clearly visible."
        }), 500

    finally:
        # Keep originals in the input folder — remove only on explicit cleanup
        pass


@app.route('/wrong_angles/<module>/<session_id>')
def get_wrong_angles(module, session_id):
    """Return the wrong-angle JSON log for a specific session."""
    if module not in MODULES:
        return jsonify({"error": "Unknown module"}), 400
    log_path = wrong_angle_log_path(module, session_id)
    if not os.path.exists(log_path):
        return jsonify({"error": "Log not found"}), 404
    with open(log_path) as f:
        return jsonify(json.load(f))


@app.route('/report/<module>/<session_id>')
def get_report(module, session_id):
    """Return the JSON session report for a given module + session."""
    if module not in MODULES:
        return jsonify({"error": "Unknown module"}), 400
    json_path, _ = report_paths(module, session_id)
    if not os.path.exists(json_path):
        return jsonify({"error": "Report not found"}), 404
    with open(json_path) as f:
        return jsonify(json.load(f))


@app.route('/reports/<module>')
def list_reports(module):
    """
    List all saved reports for a module.
    Returns a JSON array, newest first, with urls for JSON + TXT downloads.
    """
    if module not in MODULES:
        return jsonify({"error": "Unknown module"}), 400
    out_dir = module_output_dir(module)
    reports = []
    for fname in os.listdir(out_dir):
        if fname.startswith("report_") and fname.endswith(".json"):
            sid = fname[len("report_"):-len(".json")]
            fpath = os.path.join(out_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                reports.append({
                    "session":       sid,
                    "exercise":      data.get("exercise", module),
                    "recorded":      data.get("recorded", ""),
                    "source":        data.get("source", ""),
                    "form_score":    data.get("form_score", "N/A"),
                    "wrong_angle_count": data.get("wrong_angle_count", 0),
                    "report_json_url": url_for('serve_result', module=module, filename=fname),
                    "report_txt_url":  url_for('serve_result', module=module,
                                               filename=fname.replace(".json", ".txt")),
                    "report_html_url": url_for('serve_result', module=module,
                                               filename=fname.replace(".json", ".html")),
                })
            except Exception:
                pass
    reports.sort(key=lambda r: r["recorded"], reverse=True)
    return jsonify({"module": module, "count": len(reports), "reports": reports})


if __name__ == '__main__':
    ensure_module_dirs()   # create all inputs/outputs/<module> folders on startup
    app.run(debug=True, port=5000)