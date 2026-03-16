"""Evaluation Dashboard — ScreenApp vs ElevenLabs Transcript Comparison.

Usage:
    streamlit run dashboard.py

    Or set ASR_EVAL_DATA_DIR environment variable to specify the data directory:
    ASR_EVAL_DATA_DIR=/path/to/data streamlit run dashboard.py
"""

import difflib
import json
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Data directory - can be set via ASR_EVAL_DATA_DIR environment variable
# Defaults to FYP/Tests folder if not set
_default_data_dir = Path("/Users/sanukathamuditha/Desktop/FYP/Tests")
BASE = Path(os.environ.get("ASR_EVAL_DATA_DIR", str(_default_data_dir)))

# ── Video Registry ───────────────────────────────────────────────────

VIDEO_KEYS = [
    "troubleshooting_dimiter",
    "aws_migration",
    "followup_julien",
    "screenapp_migration_kimi",
    "gcp_security",
    "project_update",
    "compliance_discussion",
    "onboarding_andre",
    "zachary_onboarding",
    "business_discussion",
    "test_video",
]


# ── Data Loading ─────────────────────────────────────────────────────

@st.cache_data
def load_tter_dataset():
    path = BASE / "error_dataset_output" / "tter_dataset.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_tter_errors():
    path = BASE / "error_dataset_output" / "tter_errors.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_wer_data():
    """Consolidate WER data from per-video wer_comparison.json files."""
    wer = {}
    for key in VIDEO_KEYS:
        path = BASE / "comparison_output" / key / "wer_comparison.json"
        if path.exists():
            wer[key] = json.loads(path.read_text())
    return wer


@st.cache_data
def load_vocab():
    path = BASE / "tter_vocab.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_evaluation_report():
    path = BASE / "evaluation_output" / "evaluation_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_transcripts(key):
    """Load reference and hypothesis transcripts for a video."""
    ref_path = BASE / "elevenlabs_output" / key / "reference_transcript.txt"
    hyp_path = BASE / "comparison_output" / key / "screenapp_transcript.txt"
    spk_path = BASE / "elevenlabs_output" / key / "reference_by_speaker.txt"

    ref = ref_path.read_text() if ref_path.exists() else "Not available"
    hyp = hyp_path.read_text() if hyp_path.exists() else "Not available"
    spk = spk_path.read_text() if spk_path.exists() else None
    return ref, hyp, spk


def load_corrected_transcript(key):
    """Load the corrected transcript for a video."""
    path = BASE / "corrected_output" / key / "corrected_transcript.txt"
    return path.read_text() if path.exists() else None


@st.cache_data
def load_all_corrections():
    """Load corrections.json for all videos."""
    corrections = {}
    for key in VIDEO_KEYS:
        path = BASE / "corrected_output" / key / "corrections.json"
        if path.exists():
            corrections[key] = json.loads(path.read_text())
    return corrections


@st.cache_data
def load_correction_summary():
    """Load the correction_summary.json."""
    path = BASE / "corrected_output" / "correction_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── S3 Live Data Loading ────────────────────────────────────────────

def get_s3_client(bucket: str, region: str = "", endpoint_url: str = "", access_key: str = "", secret_key: str = ""):
    """Create a boto3 S3 client using env-based or sidebar credentials."""
    if not HAS_BOTO3:
        return None, None
    try:
        from botocore.config import Config as BotoConfig
        kwargs = {}
        if region:
            kwargs["region_name"] = region
        endpoint = endpoint_url or os.environ.get("AWS_S3_ENDPOINT", "")
        if endpoint:
            kwargs["endpoint_url"] = endpoint
            kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
        # Use explicit credentials if provided, otherwise fall back to env vars
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        client = boto3.client("s3", **kwargs)
        # Quick connectivity check
        client.head_bucket(Bucket=bucket)
        return client, bucket
    except (ClientError, NoCredentialsError, Exception) as e:
        st.error(f"S3 connection error: {e}")
        return None, None


@st.cache_data(ttl=300)
def list_s3_correction_reports(_bucket: str, _region: str = "", prefix: str = "transcripts/", _endpoint: str = "", _access_key: str = "", _secret_key: str = ""):
    """List all *_correction_report.json keys in the bucket."""
    client, bucket = get_s3_client(_bucket, _region, _endpoint, _access_key, _secret_key)
    if client is None:
        return []
    reports = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_correction_report.json"):
                    reports.append(key)
    except Exception:
        return []
    return reports


@st.cache_data(ttl=300)
def load_s3_json(_bucket: str, _region: str, key: str, _endpoint: str = "", _access_key: str = "", _secret_key: str = ""):
    """Download and parse a JSON file from S3."""
    client, bucket = get_s3_client(_bucket, _region, _endpoint, _access_key, _secret_key)
    if client is None:
        return None
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_s3_transcript_text(_bucket: str, _region: str, key: str, _endpoint: str = "", _access_key: str = "", _secret_key: str = ""):
    """Download a transcript JSON from S3 and extract the plain text."""
    data = load_s3_json(_bucket, _region, key, _endpoint, _access_key, _secret_key)
    if data is None:
        return None
    # OutputTranscript format has a "text" field
    if isinstance(data, dict) and "text" in data:
        return data["text"]
    return json.dumps(data, indent=2)


def diff_highlight_html(original: str, corrected: str) -> tuple:
    """Generate HTML for original and corrected texts with word-level diff highlights.

    Returns (original_html, corrected_html) with changed words wrapped in
    colored spans — red/strikethrough for removed, green/bold for added.
    """
    orig_words = original.split()
    corr_words = corrected.split()

    sm = difflib.SequenceMatcher(None, orig_words, corr_words)

    orig_parts = []
    corr_parts = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            orig_parts.append(" ".join(orig_words[i1:i2]))
            corr_parts.append(" ".join(corr_words[j1:j2]))
        elif op == "replace":
            orig_parts.append(
                '<span class="del">' + " ".join(orig_words[i1:i2]) + "</span>"
            )
            corr_parts.append(
                '<span class="add">' + " ".join(corr_words[j1:j2]) + "</span>"
            )
        elif op == "delete":
            orig_parts.append(
                '<span class="del">' + " ".join(orig_words[i1:i2]) + "</span>"
            )
        elif op == "insert":
            corr_parts.append(
                '<span class="add">' + " ".join(corr_words[j1:j2]) + "</span>"
            )

    return " ".join(orig_parts), " ".join(corr_parts)


def build_video_table(tter_data, wer_data):
    """Build unified per-video DataFrame with WER + TTER."""
    rows = []
    for key in VIDEO_KEYS:
        row = {"key": key}

        # TTER data
        if tter_data and key in tter_data.get("videos", {}):
            v = tter_data["videos"][key]
            row["video"] = v["video"]
            row["tter"] = v["overall_tter"]
            row["tter_occ"] = v["total_occurrences"]
            row["tter_err"] = v["total_errors"]
            row["tter_correct"] = v["total_correct"]
            row["terms_tracked"] = v["terms_tracked"]
        else:
            row["video"] = key.replace("_", " ").title()
            row["tter"] = None

        # WER data (handle both field naming conventions)
        if key in wer_data:
            w = wer_data[key]
            row["wer"] = w.get("wer")
            row["cer"] = w.get("cer")
            row["ref_words"] = w.get("ref_words") or w.get("reference_words")
            row["hyp_words"] = w.get("hyp_words") or w.get("hypothesis_words")
            row["substitutions"] = w.get("substitutions")
            row["deletions"] = w.get("deletions")
            row["insertions"] = w.get("insertions")
            row["hits"] = w.get("hits")
        else:
            row["wer"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ASR Evaluation Dashboard",
    page_icon="🎙️",
    layout="wide",
)

st.title("ASR Evaluation Dashboard")
st.caption("ScreenApp (hypothesis) vs ElevenLabs Scribe v2 (reference)")

# Load all data
tter_data = load_tter_dataset()
tter_errors = load_tter_errors()
wer_data = load_wer_data()
vocab_data = load_vocab()
eval_report = load_evaluation_report()
all_corrections = load_all_corrections()
correction_summary = load_correction_summary()
df = build_video_table(tter_data, wer_data)

# ── Tabs ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Transcript Comparison",
    "TTER Error Analysis",
    "WER Breakdown",
    "Correction Results",
    "ASR Fine-Tune Results",
    "Live ScreenApp Data",
])


# =====================================================================
#  TAB 1: OVERVIEW
# =====================================================================
with tab1:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Videos", len(df))
    with col2:
        avg_wer = df["wer"].mean()
        st.metric("Avg WER", f"{avg_wer:.1f}%")
    with col3:
        if tter_data:
            st.metric("Overall TTER", f"{tter_data['overall_tter']:.1f}%")
        else:
            st.metric("Overall TTER", "N/A")
    with col4:
        if tter_data:
            st.metric("Target Term Errors", f"{tter_data['total_errors']} / {tter_data['total_target_occurrences']}")
        else:
            st.metric("Target Term Errors", "N/A")

    st.divider()

    # Per-video table
    st.subheader("Per-Video Comparison")
    display_df = df[["video", "wer", "tter", "ref_words", "hyp_words",
                      "substitutions", "deletions", "insertions",
                      "tter_occ", "tter_err"]].copy()
    display_df.columns = ["Video", "WER %", "TTER %", "Ref Words", "Hyp Words",
                           "Subs", "Dels", "Ins", "TTER Occ", "TTER Errors"]
    st.dataframe(
        display_df.style.format({
            "WER %": "{:.1f}",
            "TTER %": "{:.1f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # WER vs TTER bar chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("WER vs TTER by Video")
        chart_df = df[["video", "wer", "tter"]].dropna().copy()
        chart_df = chart_df.sort_values("tter", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=chart_df["video"], x=chart_df["wer"],
            name="WER", orientation="h", marker_color="#636EFA",
        ))
        fig.add_trace(go.Bar(
            y=chart_df["video"], x=chart_df["tter"],
            name="TTER", orientation="h", marker_color="#EF553B",
        ))
        fig.update_layout(
            barmode="group", height=400, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Error Rate (%)", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("WER vs TTER Correlation")
        scatter_df = df[["video", "wer", "tter"]].dropna()
        fig2 = px.scatter(
            scatter_df, x="wer", y="tter", text="video",
            labels={"wer": "WER (%)", "tter": "TTER (%)"},
        )
        fig2.update_traces(textposition="top center", marker_size=10)
        fig2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width="stretch")


# =====================================================================
#  TAB 2: TRANSCRIPT COMPARISON
# =====================================================================
with tab2:
    # Video selector
    video_options = {row["video"]: row["key"] for _, row in df.iterrows()}
    selected_label = st.selectbox("Select Video", list(video_options.keys()), key="t2_video")
    selected_key = video_options[selected_label]

    # Stats for selected video
    row = df[df["key"] == selected_key].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("WER", f"{row['wer']:.1f}%" if pd.notna(row["wer"]) else "—")
    with c2:
        st.metric("TTER", f"{row['tter']:.1f}%" if pd.notna(row["tter"]) else "—")
    with c3:
        st.metric("Ref Words", int(row["ref_words"]) if pd.notna(row["ref_words"]) else "—")
    with c4:
        st.metric("Hyp Words", int(row["hyp_words"]) if pd.notna(row["hyp_words"]) else "—")

    st.divider()

    # Load transcripts
    ref_text, hyp_text, spk_text = load_transcripts(selected_key)

    # View mode toggle
    view_mode = st.radio("Reference view", ["Full text", "By speaker"], horizontal=True)

    # Side-by-side (key includes selected_key so widgets update on video change)
    left, right = st.columns(2)
    with left:
        st.markdown("**ElevenLabs (Reference)**")
        if view_mode == "By speaker" and spk_text:
            st.text_area("Reference (by speaker)", spk_text, height=500, key=f"ref_{selected_key}_spk", disabled=True, label_visibility="collapsed")
        else:
            st.text_area("Reference (full text)", ref_text, height=500, key=f"ref_{selected_key}_full", disabled=True, label_visibility="collapsed")
    with right:
        st.markdown("**ScreenApp (Hypothesis)**")
        st.text_area("Hypothesis", hyp_text, height=500, key=f"hyp_{selected_key}", disabled=True, label_visibility="collapsed")

    # TTER errors for this video
    if tter_data and selected_key in tter_data.get("videos", {}):
        v = tter_data["videos"][selected_key]
        errored = [t for t in v["term_results"] if t["errors"] > 0]
        if errored:
            st.divider()
            st.subheader(f"TTER Errors in {selected_label}")
            err_rows = []
            for t in sorted(errored, key=lambda x: x["tter"], reverse=True):
                found = set()
                for d in t["error_details"]:
                    if d.get("found_as"):
                        found.add(str(d["found_as"]))
                err_rows.append({
                    "Term": t["term"],
                    "Category": t["category"],
                    "Errors": f"{t['errors']}/{t['occurrences']}",
                    "TTER %": t["tter"],
                    "Found As": ", ".join(found) if found else "missing/unknown",
                })
            st.dataframe(pd.DataFrame(err_rows), width="stretch", hide_index=True)


# =====================================================================
#  TAB 3: TTER ERROR ANALYSIS
# =====================================================================
with tab3:
    if tter_data is None:
        st.warning("TTER dataset not found.")
    else:
        # Video filter
        video_filter = st.selectbox(
            "Filter by video",
            ["All Videos"] + [tter_data["videos"][k]["video"] for k in VIDEO_KEYS if k in tter_data["videos"]],
            key="t3_video",
        )

        # Build term-level DataFrame from TTER data
        term_rows = []
        for key, vdata in tter_data["videos"].items():
            for t in vdata["term_results"]:
                if t["occurrences"] == 0:
                    continue
                found_as_set = set()
                for d in t.get("error_details", []):
                    if d.get("found_as"):
                        found_as_set.add(str(d["found_as"]))

                term_rows.append({
                    "Video": vdata["video"],
                    "Term": t["term"],
                    "Category": t["category"],
                    "Occurrences": t["occurrences"],
                    "Correct": t["correct"],
                    "Errors": t["errors"],
                    "TTER %": t["tter"],
                    "Found As": ", ".join(found_as_set) if found_as_set else ("—" if t["errors"] == 0 else "missing"),
                })

        term_df = pd.DataFrame(term_rows)

        # Apply video filter
        if video_filter != "All Videos":
            term_df = term_df[term_df["Video"] == video_filter]

        # Category filter
        categories = sorted(term_df["Category"].unique())
        selected_cats = st.multiselect("Filter by category", categories, default=categories, key="t3_cats")
        term_df = term_df[term_df["Category"].isin(selected_cats)]

        # TTER threshold
        min_tter = st.slider("Minimum TTER %", 0, 100, 0, key="t3_thresh")
        term_df = term_df[term_df["TTER %"] >= min_tter]

        st.divider()

        # Category breakdown chart
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("TTER by Category")
            cat_agg = term_df.groupby("Category").agg(
                Occurrences=("Occurrences", "sum"),
                Errors=("Errors", "sum"),
            ).reset_index()
            cat_agg["TTER %"] = (cat_agg["Errors"] / cat_agg["Occurrences"] * 100).round(1)
            cat_agg = cat_agg.sort_values("TTER %", ascending=True)

            fig3 = px.bar(
                cat_agg, y="Category", x="TTER %", orientation="h",
                color="TTER %", color_continuous_scale="RdYlGn_r",
                text="TTER %",
            )
            fig3.update_layout(height=max(300, len(cat_agg) * 28), margin=dict(l=0, r=0, t=10, b=0))
            fig3.update_traces(textposition="outside")
            st.plotly_chart(fig3, width="stretch")

        with col_r:
            st.subheader("Top 20 Worst Terms")
            worst = term_df.nlargest(20, "Errors")[["Term", "Video", "Category", "Errors", "Occurrences", "TTER %", "Found As"]]
            st.dataframe(worst, width="stretch", hide_index=True)

        st.divider()

        # Full error table
        st.subheader(f"All Terms ({len(term_df)} terms)")
        st.dataframe(
            term_df.sort_values("TTER %", ascending=False),
            width="stretch",
            hide_index=True,
        )

        # Raw CSV errors with context
        if not tter_errors.empty:
            with st.expander("Raw Error Records with Context (from CSV)"):
                csv_display = tter_errors.copy()
                if video_filter != "All Videos":
                    csv_display = csv_display[csv_display["video"] == video_filter]
                if selected_cats:
                    csv_display = csv_display[csv_display["category"].isin(selected_cats)]
                st.dataframe(csv_display, width="stretch", hide_index=True)


# =====================================================================
#  TAB 4: WER BREAKDOWN
# =====================================================================
with tab4:
    if not wer_data:
        st.warning("WER data not found.")
    else:
        # Stacked bar chart: subs, dels, ins per video
        st.subheader("Error Breakdown by Video")
        wer_rows = []
        for key in VIDEO_KEYS:
            if key not in wer_data:
                continue
            w = wer_data[key]
            label = w.get("label", key)
            wer_rows.append({
                "Video": label,
                "Substitutions": w.get("substitutions", 0),
                "Deletions": w.get("deletions", 0),
                "Insertions": w.get("insertions", 0),
                "WER %": w.get("wer", 0),
                "Ref Words": w.get("ref_words", 0),
                "Hyp Words": w.get("hyp_words", 0),
                "Hits": w.get("hits", 0),
            })

        wer_df = pd.DataFrame(wer_rows)

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Substitutions", x=wer_df["Video"], y=wer_df["Substitutions"], marker_color="#636EFA"))
        fig4.add_trace(go.Bar(name="Deletions", x=wer_df["Video"], y=wer_df["Deletions"], marker_color="#EF553B"))
        fig4.add_trace(go.Bar(name="Insertions", x=wer_df["Video"], y=wer_df["Insertions"], marker_color="#00CC96"))
        fig4.update_layout(
            barmode="stack", height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Word Count",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig4, width="stretch")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Overall Error Distribution")
            total_subs = wer_df["Substitutions"].sum()
            total_dels = wer_df["Deletions"].sum()
            total_ins = wer_df["Insertions"].sum()

            fig5 = px.pie(
                values=[total_subs, total_dels, total_ins],
                names=["Substitutions", "Deletions", "Insertions"],
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
            )
            fig5.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig5, width="stretch")

        with col_b:
            st.subheader("Word Count Comparison")
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(
                name="Reference (ElevenLabs)", x=wer_df["Video"], y=wer_df["Ref Words"],
                marker_color="#636EFA",
            ))
            fig6.add_trace(go.Bar(
                name="Hypothesis (ScreenApp)", x=wer_df["Video"], y=wer_df["Hyp Words"],
                marker_color="#EF553B",
            ))
            fig6.update_layout(
                barmode="group", height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Words",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig6, width="stretch")

        st.divider()

        # Detailed WER table
        st.subheader("Detailed WER Table")
        st.dataframe(
            wer_df.style.format({"WER %": "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )


# =====================================================================
#  TAB 5: CORRECTION RESULTS
# =====================================================================
with tab5:
    if eval_report is None:
        st.warning("No evaluation report found. Run evaluate_correction.py first.")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "TTER Before",
                f"{eval_report['overall_tter_before']:.1f}%",
            )
        with c2:
            st.metric(
                "TTER After",
                f"{eval_report['overall_tter_after']:.1f}%",
                delta=f"{eval_report['overall_tter_delta']:+.1f}%",
                delta_color="inverse",
            )
        with c3:
            st.metric("Errors Fixed", eval_report["errors_fixed"])
        with c4:
            st.metric(
                "Videos Processed",
                len(eval_report.get("videos", {})),
            )

        st.divider()

        # Per-video comparison table
        st.subheader("TTER Before vs After Correction")
        corr_rows = []
        for key, v in eval_report["videos"].items():
            corr_rows.append({
                "Video": v["label"],
                "TTER Before": v["tter_before"],
                "TTER After": v["tter_after"],
                "Delta": v["tter_delta"],
                "Errors Before": v["errors_before"],
                "Errors After": v["errors_after"],
                "Fixed": v["errors_before"] - v["errors_after"],
                "OCR": "Yes" if v.get("has_ocr") else "No",
            })
        corr_df = pd.DataFrame(corr_rows)
        st.dataframe(
            corr_df.style.format({
                "TTER Before": "{:.1f}",
                "TTER After": "{:.1f}",
                "Delta": "{:+.1f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.divider()

        # Before/After bar chart
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("TTER Before vs After")
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                y=corr_df["Video"], x=corr_df["TTER Before"],
                name="Before", orientation="h", marker_color="#EF553B",
            ))
            fig_corr.add_trace(go.Bar(
                y=corr_df["Video"], x=corr_df["TTER After"],
                name="After", orientation="h", marker_color="#00CC96",
            ))
            fig_corr.update_layout(
                barmode="group", height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="TTER (%)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_corr, width="stretch")

        with col_r:
            st.subheader("Errors Fixed per Video")
            fig_fixed = px.bar(
                corr_df.sort_values("Fixed", ascending=True),
                y="Video", x="Fixed", orientation="h",
                color="OCR",
                color_discrete_map={"Yes": "#636EFA", "No": "#AB63FA"},
            )
            fig_fixed.update_layout(
                height=400, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Errors Fixed",
            )
            st.plotly_chart(fig_fixed, width="stretch")

        st.divider()

        # OCR impact
        st.subheader("OCR Impact Analysis")
        ocr_vids = {k: v for k, v in eval_report["videos"].items() if v.get("has_ocr")}
        no_ocr_vids = {k: v for k, v in eval_report["videos"].items() if not v.get("has_ocr")}

        oc1, oc2 = st.columns(2)
        with oc1:
            if ocr_vids:
                ocr_err_b = sum(v["errors_before"] for v in ocr_vids.values())
                ocr_err_a = sum(v["errors_after"] for v in ocr_vids.values())
                ocr_occ = sum(v.get("occurrences", ocr_err_b) for v in ocr_vids.values())
                st.metric(
                    f"With OCR ({len(ocr_vids)} videos)",
                    f"{ocr_err_a} errors",
                    delta=f"{ocr_err_a - ocr_err_b:+d} from {ocr_err_b}",
                    delta_color="inverse",
                )
        with oc2:
            if no_ocr_vids:
                no_err_b = sum(v["errors_before"] for v in no_ocr_vids.values())
                no_err_a = sum(v["errors_after"] for v in no_ocr_vids.values())
                st.metric(
                    f"Without OCR ({len(no_ocr_vids)} videos)",
                    f"{no_err_a} errors",
                    delta=f"{no_err_a - no_err_b:+d} from {no_err_b}",
                    delta_color="inverse",
                )

        st.divider()

        # Per-video improved/degraded terms
        st.subheader("Term-Level Changes")
        for key, v in eval_report["videos"].items():
            improved = v.get("improved_terms", [])
            degraded = v.get("degraded_terms", [])
            if not improved and not degraded:
                continue

            with st.expander(f"{v['label']} — {len(improved)} improved, {len(degraded)} degraded"):
                if improved:
                    imp_df = pd.DataFrame(improved)
                    imp_df.columns = ["Term", "Category", "Before", "After"]
                    imp_df["Change"] = imp_df["After"] - imp_df["Before"]
                    st.markdown("**Improved:**")
                    st.dataframe(imp_df, width="stretch", hide_index=True)
                if degraded:
                    deg_df = pd.DataFrame(degraded)
                    deg_df.columns = ["Term", "Category", "Before", "After"]
                    deg_df["Change"] = deg_df["After"] - deg_df["Before"]
                    st.markdown("**Degraded:**")
                    st.dataframe(deg_df, width="stretch", hide_index=True)

        st.divider()

        # ── Transcript Diff Viewer ──────────────────────────────────────
        st.subheader("Transcript Comparison (Original → Corrected)")

        diff_options = {}
        for key, v in eval_report["videos"].items():
            fixed = v["errors_before"] - v["errors_after"]
            diff_options[f"{v['label']} ({fixed:+d} errors)"] = key

        diff_label = st.selectbox(
            "Select Video", list(diff_options.keys()), key="t5_diff_video",
        )
        diff_key = diff_options[diff_label]

        sa_path = BASE / "comparison_output" / diff_key / "screenapp_transcript.txt"
        original_text = sa_path.read_text() if sa_path.exists() else None
        corrected_text = load_corrected_transcript(diff_key)

        if original_text and corrected_text:
            if original_text.strip() == corrected_text.strip():
                st.info("No changes — original and corrected transcripts are identical.")
            else:
                orig_html, corr_html = diff_highlight_html(original_text, corrected_text)

                diff_full_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
        color: #e0e0e0;
    }}
    .diff-wrapper {{
        display: flex;
        gap: 12px;
    }}
    .diff-col {{
        flex: 1;
        min-width: 0;
    }}
    .diff-col-header {{
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
    }}
    .diff-scroll {{
        max-height: 460px;
        overflow-y: auto;
    }}
    .diff-box {{
        font-family: 'Source Code Pro', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        white-space: pre-wrap;
        word-wrap: break-word;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 16px;
        color: #e0e0e0;
        background: rgba(30, 30, 30, 0.6);
    }}
    .diff-box .del {{
        background-color: rgba(255, 80, 80, 0.25);
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 3px;
        color: #ff8888;
    }}
    .diff-box .add {{
        background-color: rgba(80, 220, 100, 0.25);
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 3px;
        color: #66ff88;
    }}
    @media (prefers-color-scheme: light) {{
        body {{ color: #1a1a1a; }}
        .diff-box {{ color: #1a1a1a; background: #f8f8f8; }}
        .diff-box .del {{ background-color: #fdd; color: #c00; }}
        .diff-box .add {{ background-color: #dfd; color: #060; }}
    }}
</style>
</head>
<body>
<div class="diff-wrapper">
    <div class="diff-col">
        <div class="diff-col-header">
            Original (ScreenApp)
            <span class="del" style="font-size:12px;margin-left:6px;">removed</span>
        </div>
        <div class="diff-scroll" id="scroll-left">
            <div class="diff-box">{orig_html}</div>
        </div>
    </div>
    <div class="diff-col">
        <div class="diff-col-header">
            Corrected
            <span class="add" style="font-size:12px;margin-left:6px;">added</span>
        </div>
        <div class="diff-scroll" id="scroll-right">
            <div class="diff-box">{corr_html}</div>
        </div>
    </div>
</div>
<script>
(function() {{
    const left = document.getElementById('scroll-left');
    const right = document.getElementById('scroll-right');
    let syncing = false;
    left.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = left.scrollTop / (left.scrollHeight - left.clientHeight || 1);
        right.scrollTop = pct * (right.scrollHeight - right.clientHeight || 1);
        syncing = false;
    }});
    right.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = right.scrollTop / (right.scrollHeight - right.clientHeight || 1);
        left.scrollTop = pct * (left.scrollHeight - left.clientHeight || 1);
        syncing = false;
    }});
}})();
</script>
</body>
</html>"""
                components.html(diff_full_html, height=550, scrolling=False)

                # Corrections log
                corr_log_path = BASE / "corrected_output" / diff_key / "corrections.json"
                if corr_log_path.exists():
                    corr_log = json.loads(corr_log_path.read_text())
                    applied = [
                        c for c in corr_log.get("corrections", [])
                        if c["result"].get("confidence", 0) >= 0.7
                        and c["result"].get("changes")
                    ]
                    if applied:
                        with st.expander(f"Corrections Applied ({len(applied)})"):
                            for c in applied:
                                changes = c["result"].get("changes", [])
                                change_str = changes[0] if changes else "—"
                                conf = c["result"].get("confidence", 0)
                                st.markdown(
                                    f"- **{c['term']}** ({c['category']}): "
                                    f"`{change_str}` — confidence {conf:.2f}"
                                    + (" | OCR hints used" if c.get("ocr_hints_used") else "")
                                )
        else:
            st.warning("Transcripts not available for this video.")


# =====================================================================
#  TAB 6: ASR FINE-TUNE RESULTS
# =====================================================================
with tab6:
    if eval_report is None and not all_corrections:
        st.warning(
            "No ASR correction data found. Run the correction pipeline and "
            "evaluate_correction.py first."
        )
    else:
        # ── Section 1: Summary Metrics ──────────────────────────────────
        st.subheader("Fine-Tune Correction Summary")

        # Aggregate correction stats from per-video corrections.json
        total_attempted = 0
        total_applied = 0
        total_videos_corrected = 0
        all_confidences = []

        for key, cdata in all_corrections.items():
            total_attempted += cdata.get("corrections_attempted", 0)
            total_applied += cdata.get("corrections_applied", 0)
            total_videos_corrected += 1
            for c in cdata.get("corrections", []):
                conf = c.get("result", {}).get("confidence", 0)
                if conf > 0:
                    all_confidences.append(conf)

        avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Videos Processed", total_videos_corrected)
        with c2:
            st.metric("Corrections Attempted", total_attempted)
        with c3:
            st.metric("Corrections Applied", total_applied)
        with c4:
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        with c5:
            if eval_report:
                delta = eval_report["overall_tter_delta"]
                st.metric(
                    "TTER Improvement",
                    f"{eval_report['overall_tter_after']:.1f}%",
                    delta=f"{delta:+.1f}%",
                    delta_color="inverse",
                )
            else:
                st.metric("TTER Improvement", "N/A")

        st.divider()

        # ── Section 2: Per-Video Comparison Table ───────────────────────
        st.subheader("Per-Video Correction Comparison")

        table_rows = []
        for key in VIDEO_KEYS:
            row = {"Video": key.replace("_", " ").title()}

            # Corrections data
            if key in all_corrections:
                cd = all_corrections[key]
                row["Attempted"] = cd.get("corrections_attempted", 0)
                row["Applied"] = cd.get("corrections_applied", 0)
                confs = [
                    c["result"]["confidence"]
                    for c in cd.get("corrections", [])
                    if c.get("result", {}).get("confidence", 0) > 0
                ]
                row["Avg Conf"] = sum(confs) / len(confs) if confs else 0
            else:
                row["Attempted"] = 0
                row["Applied"] = 0
                row["Avg Conf"] = 0

            # Eval report data
            if eval_report and key in eval_report.get("videos", {}):
                ev = eval_report["videos"][key]
                row["Video"] = ev.get("label", row["Video"])
                row["TTER Before"] = ev["tter_before"]
                row["TTER After"] = ev["tter_after"]
                row["TTER Delta"] = ev["tter_delta"]
                row["WER Before"] = ev.get("wer_before", None)
                row["WER After"] = ev.get("wer_after", None)
                row["WER Delta"] = ev.get("wer_delta", None)
                row["OCR"] = "Yes" if ev.get("has_ocr") else "No"
            else:
                row["TTER Before"] = None
                row["TTER After"] = None
                row["TTER Delta"] = None
                row["WER Before"] = None
                row["WER After"] = None
                row["WER Delta"] = None
                row["OCR"] = "—"

            # Only include if we have some data
            if key in all_corrections or (
                eval_report and key in eval_report.get("videos", {})
            ):
                table_rows.append(row)

        if table_rows:
            table_df = pd.DataFrame(table_rows)
            st.dataframe(
                table_df.style.format(
                    {
                        "TTER Before": "{:.1f}",
                        "TTER After": "{:.1f}",
                        "TTER Delta": "{:+.1f}",
                        "WER Before": "{:.1f}",
                        "WER After": "{:.1f}",
                        "WER Delta": "{:+.1f}",
                        "Avg Conf": "{:.2f}",
                    },
                    na_rep="—",
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No per-video data available.")

        st.divider()

        # ── Section 3: Side-by-Side Transcript Viewer ───────────────────
        st.subheader("Original vs Enhanced Transcript")

        # Build video selector from videos that have corrected transcripts
        t6_video_options = {}
        for key in VIDEO_KEYS:
            corrected = load_corrected_transcript(key)
            if corrected is not None:
                label = key.replace("_", " ").title()
                if eval_report and key in eval_report.get("videos", {}):
                    ev = eval_report["videos"][key]
                    label = ev.get("label", label)
                    fixed = ev["errors_before"] - ev["errors_after"]
                    label = f"{label} ({fixed:+d} errors)"
                t6_video_options[label] = key

        if t6_video_options:
            t6_selected_label = st.selectbox(
                "Select Video",
                list(t6_video_options.keys()),
                key="t6_video",
            )
            t6_selected_key = t6_video_options[t6_selected_label]

            # Stats for selected video
            if eval_report and t6_selected_key in eval_report.get("videos", {}):
                ev = eval_report["videos"][t6_selected_key]
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.metric(
                        "TTER Before",
                        f"{ev['tter_before']:.1f}%",
                    )
                with mc2:
                    st.metric(
                        "TTER After",
                        f"{ev['tter_after']:.1f}%",
                        delta=f"{ev['tter_delta']:+.1f}%",
                        delta_color="inverse",
                    )
                with mc3:
                    st.metric("Errors Fixed", ev["errors_before"] - ev["errors_after"])
                with mc4:
                    if ev.get("wer_before") is not None:
                        st.metric(
                            "WER Before",
                            f"{ev['wer_before']:.1f}%",
                        )
                    else:
                        st.metric("WER Before", "—")
                with mc5:
                    if ev.get("wer_after") is not None:
                        st.metric(
                            "WER After",
                            f"{ev['wer_after']:.1f}%",
                            delta=f"{ev.get('wer_delta', 0):+.1f}%",
                            delta_color="inverse",
                        )
                    else:
                        st.metric("WER After", "—")

            # Load transcripts
            sa_path = BASE / "comparison_output" / t6_selected_key / "screenapp_transcript.txt"
            t6_original = sa_path.read_text() if sa_path.exists() else None
            t6_corrected = load_corrected_transcript(t6_selected_key)

            if t6_original and t6_corrected:
                if t6_original.strip() == t6_corrected.strip():
                    st.info("No changes — original and corrected transcripts are identical.")
                else:
                    t6_orig_html, t6_corr_html = diff_highlight_html(t6_original, t6_corrected)

                    t6_diff_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
        color: #e0e0e0;
    }}
    .diff-wrapper {{
        display: flex;
        gap: 12px;
    }}
    .diff-col {{
        flex: 1;
        min-width: 0;
    }}
    .diff-col-header {{
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
    }}
    .diff-scroll {{
        max-height: 460px;
        overflow-y: auto;
    }}
    .diff-box {{
        font-family: 'Source Code Pro', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        white-space: pre-wrap;
        word-wrap: break-word;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 16px;
        color: #e0e0e0;
        background: rgba(30, 30, 30, 0.6);
    }}
    .diff-box .del {{
        background-color: rgba(255, 80, 80, 0.25);
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 3px;
        color: #ff8888;
    }}
    .diff-box .add {{
        background-color: rgba(80, 220, 100, 0.25);
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 3px;
        color: #66ff88;
    }}
    @media (prefers-color-scheme: light) {{
        body {{ color: #1a1a1a; }}
        .diff-box {{ color: #1a1a1a; background: #f8f8f8; }}
        .diff-box .del {{ background-color: #fdd; color: #c00; }}
        .diff-box .add {{ background-color: #dfd; color: #060; }}
    }}
</style>
</head>
<body>
<div class="diff-wrapper">
    <div class="diff-col">
        <div class="diff-col-header">
            Original (ScreenApp)
            <span class="del" style="font-size:12px;margin-left:6px;">removed</span>
        </div>
        <div class="diff-scroll" id="t6-scroll-left">
            <div class="diff-box">{t6_orig_html}</div>
        </div>
    </div>
    <div class="diff-col">
        <div class="diff-col-header">
            Enhanced (Fine-Tuned)
            <span class="add" style="font-size:12px;margin-left:6px;">added</span>
        </div>
        <div class="diff-scroll" id="t6-scroll-right">
            <div class="diff-box">{t6_corr_html}</div>
        </div>
    </div>
</div>
<script>
(function() {{
    const left = document.getElementById('t6-scroll-left');
    const right = document.getElementById('t6-scroll-right');
    let syncing = false;
    left.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = left.scrollTop / (left.scrollHeight - left.clientHeight || 1);
        right.scrollTop = pct * (right.scrollHeight - right.clientHeight || 1);
        syncing = false;
    }});
    right.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = right.scrollTop / (right.scrollHeight - right.clientHeight || 1);
        left.scrollTop = pct * (left.scrollHeight - left.clientHeight || 1);
        syncing = false;
    }});
}})();
</script>
</body>
</html>"""
                    components.html(t6_diff_html, height=550, scrolling=False)
            else:
                st.warning("Transcripts not available for this video.")

            st.divider()

            # ── Section 4: Correction Details ───────────────────────────
            st.subheader("Correction Details")

            if t6_selected_key in all_corrections:
                cdata = all_corrections[t6_selected_key]
                corrections_list = cdata.get("corrections", [])

                if corrections_list:
                    # Summary row
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.metric(
                            "Total Corrections",
                            len(corrections_list),
                        )
                    with dc2:
                        applied_ct = sum(
                            1
                            for c in corrections_list
                            if c["result"].get("confidence", 0) >= 0.7
                            and c["result"].get("changes")
                        )
                        st.metric("Applied", applied_ct)
                    with dc3:
                        skipped_ct = len(corrections_list) - applied_ct
                        st.metric("Skipped", skipped_ct)

                    # Corrections table
                    corr_detail_rows = []
                    for i, c in enumerate(corrections_list):
                        conf = c["result"].get("confidence", 0)
                        changes = c["result"].get("changes", [])
                        change_str = changes[0] if changes else "—"
                        applied_flag = conf >= 0.7 and bool(changes)

                        corr_detail_rows.append({
                            "#": i + 1,
                            "Term": c.get("term", ""),
                            "Category": c.get("category", ""),
                            "Error Found": c.get("original_error", ""),
                            "Change": change_str,
                            "Confidence": conf,
                            "OCR": "Yes" if c.get("ocr_hints_used") else "No",
                            "Lip Read": "Yes" if c["result"].get("need_lip") else "No",
                            "Applied": "Yes" if applied_flag else "No",
                        })

                    corr_detail_df = pd.DataFrame(corr_detail_rows)
                    st.dataframe(
                        corr_detail_df.style.format({"Confidence": "{:.2f}"}).apply(
                            lambda row: [
                                "background-color: rgba(80, 220, 100, 0.15)"
                                if row["Applied"] == "Yes"
                                else "background-color: rgba(255, 165, 0, 0.15)"
                            ]
                            * len(row),
                            axis=1,
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    # Context expander for each correction
                    with st.expander("View Correction Contexts"):
                        for i, c in enumerate(corrections_list):
                            conf = c["result"].get("confidence", 0)
                            changes = c["result"].get("changes", [])
                            change_str = changes[0] if changes else "no change"
                            status = "Applied" if (conf >= 0.7 and changes) else "Skipped"

                            st.markdown(
                                f"**{i+1}. {c.get('term', '')}** "
                                f"({c.get('category', '')}) — "
                                f"`{change_str}` — "
                                f"conf: {conf:.2f} — "
                                f"**{status}**"
                            )
                            st.code(c.get("context", ""), language=None)
                else:
                    st.info("No corrections were attempted for this video.")
            else:
                st.info("No correction data available for this video.")

            st.divider()

            # ── Section 5: Term-Level Analysis ──────────────────────────
            st.subheader("Term-Level Changes")

            if eval_report and t6_selected_key in eval_report.get("videos", {}):
                ev = eval_report["videos"][t6_selected_key]
                improved = ev.get("improved_terms", [])
                degraded = ev.get("degraded_terms", [])

                if improved or degraded:
                    tcol_l, tcol_r = st.columns(2)

                    with tcol_l:
                        if improved:
                            st.markdown("**Improved Terms**")
                            imp_df = pd.DataFrame(improved).rename(columns={
                                "term": "Term", "category": "Category",
                                "before": "Errors Before", "after": "Errors After",
                            })
                            imp_df["Improvement"] = imp_df["Errors Before"] - imp_df["Errors After"]
                            st.dataframe(imp_df, width="stretch", hide_index=True)
                        else:
                            st.info("No terms improved.")

                    with tcol_r:
                        if degraded:
                            st.markdown("**Degraded Terms**")
                            deg_df = pd.DataFrame(degraded).rename(columns={
                                "term": "Term", "category": "Category",
                                "before": "Errors Before", "after": "Errors After",
                            })
                            deg_df["Regression"] = deg_df["Errors After"] - deg_df["Errors Before"]
                            st.dataframe(deg_df, width="stretch", hide_index=True)
                        else:
                            st.info("No terms degraded.")

                    # Bar chart: before vs after per term
                    all_terms = []
                    for t in improved:
                        all_terms.append({"Term": t["term"], "Errors Before": t["before"], "Errors After": t["after"]})
                    for t in degraded:
                        all_terms.append({"Term": t["term"], "Errors Before": t["before"], "Errors After": t["after"]})

                    if all_terms:
                        terms_chart_df = pd.DataFrame(all_terms).sort_values(
                            "Errors Before", ascending=True
                        )
                        fig_terms = go.Figure()
                        fig_terms.add_trace(go.Bar(
                            y=terms_chart_df["Term"],
                            x=terms_chart_df["Errors Before"],
                            name="Before",
                            orientation="h",
                            marker_color="#EF553B",
                        ))
                        fig_terms.add_trace(go.Bar(
                            y=terms_chart_df["Term"],
                            x=terms_chart_df["Errors After"],
                            name="After",
                            orientation="h",
                            marker_color="#00CC96",
                        ))
                        fig_terms.update_layout(
                            barmode="group",
                            height=max(300, len(all_terms) * 30),
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Error Count",
                            legend=dict(orientation="h", y=1.1),
                            title="Errors Before vs After per Term",
                        )
                        st.plotly_chart(fig_terms, width="stretch")
                else:
                    st.info("No term-level changes recorded for this video.")
            else:
                st.info("No evaluation data available for this video.")

        else:
            st.warning("No corrected transcripts found.")

        st.divider()

        # ── Section 6: Vocabulary & OCR Impact ──────────────────────────
        st.subheader("Vocabulary & OCR Impact Analysis")

        if all_corrections:
            # Aggregate OCR usage
            total_with_ocr = 0
            total_without_ocr = 0
            conf_with_ocr = []
            conf_without_ocr = []
            category_counts = {}

            for key, cdata in all_corrections.items():
                for c in cdata.get("corrections", []):
                    cat = c.get("category", "unknown")
                    category_counts[cat] = category_counts.get(cat, 0) + 1

                    conf = c.get("result", {}).get("confidence", 0)
                    if c.get("ocr_hints_used"):
                        total_with_ocr += 1
                        if conf > 0:
                            conf_with_ocr.append(conf)
                    else:
                        total_without_ocr += 1
                        if conf > 0:
                            conf_without_ocr.append(conf)

            ocr_col, cat_col = st.columns(2)

            with ocr_col:
                st.markdown("**OCR Hints Usage**")
                if total_with_ocr > 0 or total_without_ocr > 0:
                    fig_ocr = px.pie(
                        values=[total_with_ocr, total_without_ocr],
                        names=["With OCR", "Without OCR"],
                        color_discrete_sequence=["#636EFA", "#AB63FA"],
                    )
                    fig_ocr.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig_ocr, width="stretch")

                    # Confidence comparison
                    avg_ocr = (
                        sum(conf_with_ocr) / len(conf_with_ocr)
                        if conf_with_ocr
                        else 0
                    )
                    avg_no_ocr = (
                        sum(conf_without_ocr) / len(conf_without_ocr)
                        if conf_without_ocr
                        else 0
                    )
                    oc1, oc2 = st.columns(2)
                    with oc1:
                        st.metric(
                            f"Avg Conf (OCR, n={len(conf_with_ocr)})",
                            f"{avg_ocr:.2f}",
                        )
                    with oc2:
                        st.metric(
                            f"Avg Conf (No OCR, n={len(conf_without_ocr)})",
                            f"{avg_no_ocr:.2f}",
                        )
                else:
                    st.info("No OCR data available.")

            with cat_col:
                st.markdown("**Corrections by Category**")
                if category_counts:
                    cat_df = pd.DataFrame(
                        [{"Category": k, "Count": v} for k, v in category_counts.items()]
                    ).sort_values("Count", ascending=True)
                    fig_cat = px.bar(
                        cat_df,
                        y="Category",
                        x="Count",
                        orientation="h",
                        color="Count",
                        color_continuous_scale="Viridis",
                    )
                    fig_cat.update_layout(
                        height=max(300, len(cat_df) * 30),
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cat, width="stretch")
                else:
                    st.info("No category data available.")
        else:
            st.info("No correction data available for OCR analysis.")


# =====================================================================
#  TAB 7: LIVE SCREENAPP DATA (from S3)
# =====================================================================
with tab7:
    st.subheader("Live ScreenApp Correction Data")
    st.caption(
        "Load ASR correction reports saved by ScreenApp's v6 pipeline from S3. "
        "Each video processed with ASR correction saves a `_correction_report.json` artifact."
    )

    if not HAS_BOTO3:
        st.error(
            "boto3 is not installed. Install it with `pip install boto3` to enable S3 data loading."
        )
    else:
        # ── S3 Configuration (sidebar) ──────────────────────────────
        with st.sidebar:
            st.divider()
            st.subheader("S3 Configuration")

            # Initialize session state for tracking preset changes
            if "last_storage_preset" not in st.session_state:
                st.session_state.last_storage_preset = None

            # Storage provider preset
            storage_preset = st.selectbox(
                "Storage Provider",
                ["MinIO (Local)", "GCP (Google Cloud Storage)", "AWS S3", "Custom"],
                index=0,
                key="storage_preset_select",
                help="Select storage provider preset or choose Custom for manual configuration",
            )

            # Detect preset change and directly update session state values
            if st.session_state.last_storage_preset is not None and st.session_state.last_storage_preset != storage_preset:
                st.session_state.last_storage_preset = storage_preset
                # Directly set the new values in session state (more reliable than deleting keys)
                if storage_preset == "GCP (Google Cloud Storage)":
                    st.session_state.s3_endpoint = "https://storage.googleapis.com"
                    st.session_state.s3_bucket = os.environ.get("GCP_USER_RECORDINGS_BUCKET", "store-screenapp-local")
                    st.session_state.s3_region = os.environ.get("GCP_DEFAULT_REGION", "auto")
                    st.session_state.s3_access_key = os.environ.get("GCP_ACCESS_KEY_ID", "")
                    st.session_state.s3_secret_key = os.environ.get("GCP_SECRET_ACCESS_KEY", "")
                elif storage_preset == "MinIO (Local)":
                    st.session_state.s3_endpoint = "http://localhost:9000"
                    st.session_state.s3_bucket = os.environ.get("MIO_USER_RECORDINGS_BUCKET", "mio.dev.store.screenapp.io")
                    st.session_state.s3_region = "us-east-1"
                    st.session_state.s3_access_key = os.environ.get("MIO_ACCESS_KEY_ID", "minio")
                    st.session_state.s3_secret_key = os.environ.get("MIO_SECRET_ACCESS_KEY", "minio123")
                elif storage_preset == "AWS S3":
                    st.session_state.s3_endpoint = ""
                    st.session_state.s3_bucket = os.environ.get("AWS_USER_RECORDINGS_BUCKET", "")
                    st.session_state.s3_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                    st.session_state.s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
                    st.session_state.s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
                else:  # Custom
                    st.session_state.s3_endpoint = os.environ.get("AWS_S3_ENDPOINT", "")
                    st.session_state.s3_bucket = os.environ.get("GCP_USER_RECORDINGS_BUCKET", os.environ.get("AWS_USER_RECORDINGS_BUCKET", ""))
                    st.session_state.s3_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                    st.session_state.s3_access_key = ""
                    st.session_state.s3_secret_key = ""
                st.rerun()
            st.session_state.last_storage_preset = storage_preset

            # Set defaults for initial load (when keys don't exist yet)
            if storage_preset == "GCP (Google Cloud Storage)":
                default_endpoint = "https://storage.googleapis.com"
                default_bucket = os.environ.get("GCP_USER_RECORDINGS_BUCKET", "store-screenapp-local")
                default_region = os.environ.get("GCP_DEFAULT_REGION", "auto")
                default_access_key = os.environ.get("GCP_ACCESS_KEY_ID", "")
                default_secret_key = os.environ.get("GCP_SECRET_ACCESS_KEY", "")
            elif storage_preset == "MinIO (Local)":
                default_endpoint = "http://localhost:9000"
                default_bucket = os.environ.get("MIO_USER_RECORDINGS_BUCKET", "mio.dev.store.screenapp.io")
                default_region = "us-east-1"
                default_access_key = os.environ.get("MIO_ACCESS_KEY_ID", "minio")
                default_secret_key = os.environ.get("MIO_SECRET_ACCESS_KEY", "minio123")
            elif storage_preset == "AWS S3":
                default_endpoint = ""
                default_bucket = os.environ.get("AWS_USER_RECORDINGS_BUCKET", "")
                default_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                default_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
                default_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            else:  # Custom
                default_endpoint = os.environ.get("AWS_S3_ENDPOINT", "")
                default_bucket = os.environ.get("GCP_USER_RECORDINGS_BUCKET", os.environ.get("AWS_USER_RECORDINGS_BUCKET", ""))
                default_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                default_access_key = ""
                default_secret_key = ""

            s3_endpoint = st.text_input(
                "S3 Endpoint URL",
                value=default_endpoint,
                key="s3_endpoint",
                help="Custom S3 endpoint. Leave empty for AWS S3.",
            )
            s3_bucket = st.text_input(
                "S3 Bucket",
                value=default_bucket,
                key="s3_bucket",
                help="The S3 bucket where ScreenApp stores transcripts",
            )
            s3_region = st.text_input(
                "Region",
                value=default_region,
                key="s3_region",
                help="Region (e.g., us-west-2 for AWS, auto for GCP)",
            )

            # Credentials section
            with st.expander("Credentials", expanded=False):
                s3_access_key = st.text_input(
                    "Access Key ID",
                    value=default_access_key,
                    type="password",
                    key="s3_access_key",
                    help="S3/GCS HMAC Access Key ID",
                )
                s3_secret_key = st.text_input(
                    "Secret Access Key",
                    value=default_secret_key,
                    type="password",
                    key="s3_secret_key",
                    help="S3/GCS HMAC Secret Access Key",
                )

            s3_prefix = st.text_input(
                "Key Prefix",
                value="transcripts/",
                help="S3 key prefix to search under (default: transcripts/)",
            )
            if st.button("Refresh S3 Data"):
                list_s3_correction_reports.clear()
                load_s3_json.clear()
                load_s3_transcript_text.clear()
                st.rerun()

        if not s3_bucket:
            st.info(
                "Enter your S3 bucket name in the sidebar (or set `AWS_USER_RECORDINGS_BUCKET` env var) "
                "to load live correction data."
            )
        else:
            # ── Load correction reports from S3 ─────────────────────
            with st.spinner("Loading correction reports from S3..."):
                report_keys = list_s3_correction_reports(s3_bucket, s3_region, prefix=s3_prefix, _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key)

            if not report_keys:
                st.warning(
                    f"No `_correction_report.json` files found in `s3://{s3_bucket}/{s3_prefix}`. "
                    "Make sure the v6 pipeline has processed some videos."
                )
            else:
                st.success(f"Found **{len(report_keys)}** correction reports in S3.")

                # Parse file IDs from keys: transcripts/{ownerId}/{fileId}_correction_report.json
                s3_files = []
                for key in report_keys:
                    parts = key.rsplit("/", 1)
                    filename = parts[-1] if len(parts) > 1 else key
                    file_id = filename.replace("_correction_report.json", "")
                    base_prefix = parts[0] + "/" if len(parts) > 1 else ""
                    s3_files.append({
                        "file_id": file_id,
                        "report_key": key,
                        "base_key": base_prefix + file_id,
                    })

                # ── Section 1: Aggregate Summary ────────────────────
                all_s3_reports = []
                for sf in s3_files:
                    report = load_s3_json(s3_bucket, s3_region, sf["report_key"], _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key)
                    if report:
                        report["_base_key"] = sf["base_key"]
                        report["_file_id"] = sf["file_id"]
                        all_s3_reports.append(report)

                if all_s3_reports:
                    s3_total_attempted = sum(r.get("corrections_attempted", 0) for r in all_s3_reports)
                    s3_total_applied = sum(r.get("corrections_applied", 0) for r in all_s3_reports)
                    s3_all_confs = []
                    for r in all_s3_reports:
                        for c in r.get("corrections", []):
                            conf = c.get("confidence", 0)
                            if conf > 0:
                                s3_all_confs.append(conf)
                    s3_avg_conf = sum(s3_all_confs) / len(s3_all_confs) if s3_all_confs else 0
                    s3_total_time = sum(r.get("processing_time_ms", 0) for r in all_s3_reports)

                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    with sc1:
                        st.metric("Videos Processed", len(all_s3_reports))
                    with sc2:
                        st.metric("Corrections Attempted", s3_total_attempted)
                    with sc3:
                        st.metric("Corrections Applied", s3_total_applied)
                    with sc4:
                        st.metric("Avg Confidence", f"{s3_avg_conf:.2f}")
                    with sc5:
                        st.metric("Total Processing", f"{s3_total_time / 1000:.1f}s")

                    st.divider()

                    # ── Section 2: Per-Video Table ──────────────────
                    st.subheader("Per-Video Correction Summary")

                    s3_table_rows = []
                    for r in all_s3_reports:
                        confs = [c.get("confidence", 0) for c in r.get("corrections", []) if c.get("confidence", 0) > 0]
                        s3_table_rows.append({
                            "File ID": r["_file_id"],
                            "Attempted": r.get("corrections_attempted", 0),
                            "Applied": r.get("corrections_applied", 0),
                            "Avg Confidence": sum(confs) / len(confs) if confs else 0,
                            "Processing (ms)": r.get("processing_time_ms", 0),
                            "Vocab Terms": len(r.get("vocab_used", [])),
                            "Created": r.get("created_at", "—"),
                        })

                    s3_table_df = pd.DataFrame(s3_table_rows)
                    st.dataframe(
                        s3_table_df.style.format({
                            "Avg Confidence": "{:.2f}",
                            "Processing (ms)": "{:.0f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

                    st.divider()

                    # ── Section 3: Per-Video Detail Viewer ──────────
                    st.subheader("Video Correction Detail")

                    s3_video_labels = {r["_file_id"]: r for r in all_s3_reports}
                    s3_selected_id = st.selectbox(
                        "Select Video (File ID)",
                        list(s3_video_labels.keys()),
                        key="s3_video_select",
                    )
                    s3_selected = s3_video_labels[s3_selected_id]

                    # ── Re-run Enhancement Button ──────────────────
                    rerun_col, rerun_status_col = st.columns([1, 3])
                    with rerun_col:
                        rerun_clicked = st.button(
                            "Re-run Enhancement",
                            key="rerun_enhance_btn",
                            type="primary",
                        )
                    if rerun_clicked:
                        with st.spinner("Running ASR correction..."):
                            try:
                                from app.asr_correction import correct_transcript
                                from app.asr_correction.config import CorrectionConfig
                                from app.asr_correction.vocabulary import load_domain_vocab, merge_vocabularies

                                # Load original transcript from S3
                                base_key = s3_selected["_base_key"]
                                original_key = f"{base_key}_original.json"
                                fallback_key = f"{base_key}.json"

                                orig_transcript = load_s3_json(
                                    s3_bucket, s3_region, original_key,
                                    _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key,
                                )
                                if orig_transcript is None:
                                    orig_transcript = load_s3_json(
                                        s3_bucket, s3_region, fallback_key,
                                        _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key,
                                    )

                                if orig_transcript is None:
                                    st.error("Could not load transcript from S3.")
                                else:
                                    file_id = s3_selected["_file_id"]
                                    config = CorrectionConfig(dry_run=False)

                                    # Load vocab
                                    vocab_snapshot = s3_selected.get("vocab_used", [])
                                    custom_terms = [
                                        v["term"] if isinstance(v, dict) else str(v)
                                        for v in vocab_snapshot
                                    ]

                                    enhanced, report = correct_transcript(
                                        transcript=orig_transcript,
                                        file_id=file_id,
                                        custom_vocabulary=custom_terms,
                                        ocr_provider=None,
                                        config=config,
                                    )

                                    # Save results back to S3
                                    client, bucket = get_s3_client(
                                        s3_bucket, s3_region, s3_endpoint, s3_access_key, s3_secret_key,
                                    )
                                    if client:
                                        # Save enhanced transcript
                                        client.put_object(
                                            Bucket=bucket, Key=fallback_key,
                                            ContentType="application/json",
                                            Body=json.dumps(enhanced).encode("utf-8"),
                                        )
                                        # Save correction report
                                        domain_vocab = load_domain_vocab(config.domain_vocab_path)
                                        vocab_terms = merge_vocabularies(custom_terms, domain_vocab)
                                        report_data = {
                                            "file_id": report.file_id,
                                            "corrections_attempted": report.corrections_attempted,
                                            "corrections_applied": report.corrections_applied,
                                            "processing_time_ms": report.processing_time_ms,
                                            "vocab_used": vocab_terms,
                                            "corrections": [
                                                {
                                                    "term": r.candidate.term,
                                                    "category": r.candidate.category,
                                                    "error_found": r.candidate.error_found,
                                                    "context": r.candidate.context,
                                                    "confidence": r.confidence,
                                                    "changes": r.changes,
                                                    "ocr_hints_used": r.ocr_hints_used,
                                                    "need_lip": r.need_lip,
                                                    "applied": r.applied,
                                                }
                                                for r in report.results
                                            ],
                                        }
                                        report_key = f"{base_key}_correction_report.json"
                                        client.put_object(
                                            Bucket=bucket, Key=report_key,
                                            ContentType="application/json",
                                            Body=json.dumps(report_data).encode("utf-8"),
                                        )

                                        st.success(
                                            f"Done! Applied {report.corrections_applied}/{report.corrections_attempted} "
                                            f"corrections in {report.processing_time_ms:.0f}ms."
                                        )
                                        # Clear cache and rerun to show updated results
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error("Could not connect to S3 to save results.")
                            except Exception as e:
                                st.error(f"Re-run failed: {e}")

                    # Per-video metrics
                    vm1, vm2, vm3, vm4 = st.columns(4)
                    with vm1:
                        st.metric("Attempted", s3_selected.get("corrections_attempted", 0))
                    with vm2:
                        st.metric("Applied", s3_selected.get("corrections_applied", 0))
                    with vm3:
                        sel_confs = [c["confidence"] for c in s3_selected.get("corrections", []) if c.get("confidence", 0) > 0]
                        st.metric("Avg Confidence", f"{sum(sel_confs) / len(sel_confs):.2f}" if sel_confs else "—")
                    with vm4:
                        st.metric("Processing", f"{s3_selected.get('processing_time_ms', 0):.0f}ms")

                    # ── Side-by-side transcript diff ────────────────
                    st.markdown("#### Original vs Enhanced Transcript")

                    base_key = s3_selected["_base_key"]
                    original_key = f"{base_key}_original.json"
                    enhanced_key = f"{base_key}.json"

                    s3_orig_text = load_s3_transcript_text(s3_bucket, s3_region, original_key, _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key)
                    s3_enh_text = load_s3_transcript_text(s3_bucket, s3_region, enhanced_key, _endpoint=s3_endpoint, _access_key=s3_access_key, _secret_key=s3_secret_key)

                    if s3_orig_text and s3_enh_text:
                        if s3_orig_text.strip() == s3_enh_text.strip():
                            st.info("No changes — original and enhanced transcripts are identical.")
                        else:
                            s3_orig_html, s3_enh_html = diff_highlight_html(s3_orig_text, s3_enh_text)

                            s3_diff_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
        color: #e0e0e0;
    }}
    .diff-wrapper {{ display: flex; gap: 12px; }}
    .diff-col {{ flex: 1; min-width: 0; }}
    .diff-col-header {{ font-weight: bold; margin-bottom: 8px; font-size: 14px; }}
    .diff-scroll {{ max-height: 460px; overflow-y: auto; }}
    .diff-box {{
        font-family: 'Source Code Pro', 'Courier New', monospace;
        font-size: 13px; line-height: 1.8;
        white-space: pre-wrap; word-wrap: break-word;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px; padding: 16px;
        color: #e0e0e0; background: rgba(30, 30, 30, 0.6);
    }}
    .diff-box .del {{
        background-color: rgba(255, 80, 80, 0.25);
        text-decoration: line-through;
        padding: 2px 4px; border-radius: 3px; color: #ff8888;
    }}
    .diff-box .add {{
        background-color: rgba(80, 220, 100, 0.25);
        font-weight: bold;
        padding: 2px 4px; border-radius: 3px; color: #66ff88;
    }}
    @media (prefers-color-scheme: light) {{
        body {{ color: #1a1a1a; }}
        .diff-box {{ color: #1a1a1a; background: #f8f8f8; }}
        .diff-box .del {{ background-color: #fdd; color: #c00; }}
        .diff-box .add {{ background-color: #dfd; color: #060; }}
    }}
</style>
</head>
<body>
<div class="diff-wrapper">
    <div class="diff-col">
        <div class="diff-col-header">
            Original
            <span class="del" style="font-size:12px;margin-left:6px;">removed</span>
        </div>
        <div class="diff-scroll" id="s3-scroll-left">
            <div class="diff-box">{s3_orig_html}</div>
        </div>
    </div>
    <div class="diff-col">
        <div class="diff-col-header">
            Enhanced (ASR Corrected)
            <span class="add" style="font-size:12px;margin-left:6px;">added</span>
        </div>
        <div class="diff-scroll" id="s3-scroll-right">
            <div class="diff-box">{s3_enh_html}</div>
        </div>
    </div>
</div>
<script>
(function() {{
    const left = document.getElementById('s3-scroll-left');
    const right = document.getElementById('s3-scroll-right');
    let syncing = false;
    left.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = left.scrollTop / (left.scrollHeight - left.clientHeight || 1);
        right.scrollTop = pct * (right.scrollHeight - right.clientHeight || 1);
        syncing = false;
    }});
    right.addEventListener('scroll', function() {{
        if (syncing) return;
        syncing = true;
        const pct = right.scrollTop / (right.scrollHeight - right.clientHeight || 1);
        left.scrollTop = pct * (left.scrollHeight - left.clientHeight || 1);
        syncing = false;
    }});
}})();
</script>
</body>
</html>"""
                            components.html(s3_diff_html, height=550, scrolling=False)
                    elif s3_orig_text or s3_enh_text:
                        st.warning(
                            "Only one transcript available. "
                            f"Original: {'found' if s3_orig_text else 'missing'}, "
                            f"Enhanced: {'found' if s3_enh_text else 'missing'}"
                        )
                    else:
                        st.info("Transcript files not found in S3 for this video.")

                    st.divider()

                    # ── Correction Details Table ────────────────────
                    st.markdown("#### Correction Details")

                    s3_corrections = s3_selected.get("corrections", [])
                    if s3_corrections:
                        s3_corr_rows = []
                        for i, c in enumerate(s3_corrections):
                            changes = c.get("changes", [])
                            change_str = changes[0] if changes else "—"
                            s3_corr_rows.append({
                                "#": i + 1,
                                "Term": c.get("term", ""),
                                "Category": c.get("category", ""),
                                "Error Found": c.get("error_found", ""),
                                "Change": change_str,
                                "Confidence": c.get("confidence", 0),
                                "OCR Hints": ", ".join(c.get("ocr_hints_used", [])) or "None",
                                "Lip Read": "Yes" if c.get("need_lip") else "No",
                                "Applied": "Yes" if c.get("applied") else "No",
                            })

                        s3_corr_df = pd.DataFrame(s3_corr_rows)
                        st.dataframe(
                            s3_corr_df.style.format({"Confidence": "{:.2f}"}).apply(
                                lambda row: [
                                    "background-color: rgba(80, 220, 100, 0.15)"
                                    if row["Applied"] == "Yes"
                                    else "background-color: rgba(255, 165, 0, 0.15)"
                                ] * len(row),
                                axis=1,
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                        # Context expander
                        with st.expander("View Correction Contexts"):
                            for i, c in enumerate(s3_corrections):
                                changes = c.get("changes", [])
                                change_str = changes[0] if changes else "no change"
                                status = "Applied" if c.get("applied") else "Skipped"
                                st.markdown(
                                    f"**{i+1}. {c.get('term', '')}** "
                                    f"({c.get('category', '')}) — "
                                    f"`{change_str}` — "
                                    f"conf: {c.get('confidence', 0):.2f} — "
                                    f"**{status}**"
                                )
                                st.code(c.get("context", ""), language=None)
                    else:
                        st.info("No corrections were attempted for this video.")

                    st.divider()

                    # ── Vocabulary Used ─────────────────────────────
                    vocab_used = s3_selected.get("vocab_used", [])
                    if vocab_used:
                        st.markdown("#### Vocabulary Used")
                        # vocab_used items may be dicts with "term" key or plain strings
                        vocab_labels = [
                            v["term"] if isinstance(v, dict) else str(v)
                            for v in vocab_used
                        ]
                        st.write(", ".join(vocab_labels))

                    # ── Aggregate Charts (across all S3 reports) ────
                    st.divider()
                    st.subheader("Aggregate Analysis (All S3 Videos)")

                    s3_ocr_col, s3_cat_col = st.columns(2)

                    # OCR impact
                    with s3_ocr_col:
                        st.markdown("**OCR Hints Usage**")
                        s3_with_ocr = 0
                        s3_without_ocr = 0
                        s3_conf_ocr = []
                        s3_conf_no_ocr = []
                        for r in all_s3_reports:
                            for c in r.get("corrections", []):
                                conf = c.get("confidence", 0)
                                if c.get("ocr_hints_used"):
                                    s3_with_ocr += 1
                                    if conf > 0:
                                        s3_conf_ocr.append(conf)
                                else:
                                    s3_without_ocr += 1
                                    if conf > 0:
                                        s3_conf_no_ocr.append(conf)

                        if s3_with_ocr > 0 or s3_without_ocr > 0:
                            fig_s3_ocr = px.pie(
                                values=[s3_with_ocr, s3_without_ocr],
                                names=["With OCR", "Without OCR"],
                                color_discrete_sequence=["#636EFA", "#AB63FA"],
                            )
                            fig_s3_ocr.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_s3_ocr, width="stretch")

                            avg_s3_ocr = sum(s3_conf_ocr) / len(s3_conf_ocr) if s3_conf_ocr else 0
                            avg_s3_no = sum(s3_conf_no_ocr) / len(s3_conf_no_ocr) if s3_conf_no_ocr else 0
                            soc1, soc2 = st.columns(2)
                            with soc1:
                                st.metric(f"Avg Conf (OCR, n={len(s3_conf_ocr)})", f"{avg_s3_ocr:.2f}")
                            with soc2:
                                st.metric(f"Avg Conf (No OCR, n={len(s3_conf_no_ocr)})", f"{avg_s3_no:.2f}")
                        else:
                            st.info("No correction data to analyze.")

                    # Category breakdown
                    with s3_cat_col:
                        st.markdown("**Corrections by Category**")
                        s3_cats = {}
                        for r in all_s3_reports:
                            for c in r.get("corrections", []):
                                cat = c.get("category", "unknown")
                                s3_cats[cat] = s3_cats.get(cat, 0) + 1

                        if s3_cats:
                            s3_cat_df = pd.DataFrame(
                                [{"Category": k, "Count": v} for k, v in s3_cats.items()]
                            ).sort_values("Count", ascending=True)
                            fig_s3_cat = px.bar(
                                s3_cat_df, y="Category", x="Count",
                                orientation="h", color="Count",
                                color_continuous_scale="Viridis",
                            )
                            fig_s3_cat.update_layout(
                                height=max(300, len(s3_cat_df) * 30),
                                margin=dict(l=0, r=0, t=30, b=0),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_s3_cat, width="stretch")
                        else:
                            st.info("No category data available.")

                    # ── Applied vs Skipped breakdown ────────────────
                    st.divider()
                    st.markdown("#### Applied vs Skipped by Video")
                    s3_apply_rows = []
                    for r in all_s3_reports:
                        fid = r["_file_id"]
                        corrs = r.get("corrections", [])
                        applied_n = sum(1 for c in corrs if c.get("applied"))
                        skipped_n = len(corrs) - applied_n
                        s3_apply_rows.append({"File ID": fid, "Applied": applied_n, "Skipped": skipped_n})

                    if s3_apply_rows:
                        s3_apply_df = pd.DataFrame(s3_apply_rows)
                        fig_s3_apply = go.Figure()
                        fig_s3_apply.add_trace(go.Bar(
                            y=s3_apply_df["File ID"], x=s3_apply_df["Applied"],
                            name="Applied", orientation="h", marker_color="#00CC96",
                        ))
                        fig_s3_apply.add_trace(go.Bar(
                            y=s3_apply_df["File ID"], x=s3_apply_df["Skipped"],
                            name="Skipped", orientation="h", marker_color="#EF553B",
                        ))
                        fig_s3_apply.update_layout(
                            barmode="stack",
                            height=max(300, len(s3_apply_rows) * 35),
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Corrections",
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig_s3_apply, width="stretch")
                else:
                    st.warning("Could not load any correction reports from S3.")
