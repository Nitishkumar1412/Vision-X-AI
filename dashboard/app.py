import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
import subprocess
import sys


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="VisionX AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Paths
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT_DIR / "logs"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
EVENT_LOG_PATH = LOGS_DIR / "vision_events.csv"
CAMERA_SCRIPT_PATH = ROOT_DIR / "src" / "04_vision_x_ai.py"


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 20% 10%, #11151c 0%, #0a0c10 60%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(0,230,180,0.12), rgba(0,120,255,0.10));
        border: 1px solid rgba(0,230,180,0.25);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00e6b4, #4da8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        margin-top: 0.4rem;
        color: #9aa5b1;
        font-size: 0.95rem;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-right: 8px;
    }

    .pill-live {
        background: rgba(0,230,140,0.15);
        color: #00e68c;
        border: 1px solid rgba(0,230,140,0.4);
    }

    .pill-idle {
        background: rgba(255,170,0,0.15);
        color: #ffaa00;
        border: 1px solid rgba(255,170,0,0.4);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(0,230,180,0.4);
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stMetricLabel"] {
        color: #8a93a3 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #e9edf3 !important;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e9edf3;
        margin: 0.4rem 0 0.8rem 0;
        border-left: 4px solid #00e6b4;
        padding-left: 10px;
    }

    section[data-testid="stSidebar"] {
        background: #0d1016;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.07);
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Load Data
# =========================
@st.cache_data(ttl=30)
def load_events():
    if EVENT_LOG_PATH.exists():
        df = pd.read_csv(EVENT_LOG_PATH)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    return pd.DataFrame(columns=["timestamp", "event_type", "details"])


def get_screenshots():
    if SCREENSHOTS_DIR.exists():
        return sorted(list(SCREENSHOTS_DIR.glob("*.png")), reverse=True)

    return []


def extract_objects(details_series):
    all_objects = []

    for details in details_series.astype(str):
        if "objects=" in details:
            try:
                object_part = details.split("objects=")[1].split(", people_count")[0]
                object_part = object_part.replace("[", "").replace("]", "").replace("'", "")
                objects = [obj.strip() for obj in object_part.split(",") if obj.strip()]
                all_objects.extend(objects)
            except Exception:
                pass

    return all_objects


events_df = load_events()
screenshots = get_screenshots()


# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔍 Filters")

    event_types = (
        sorted(events_df["event_type"].dropna().unique().tolist())
        if not events_df.empty
        else []
    )

    selected_types = st.multiselect(
        "Event types",
        options=event_types,
        default=event_types,
    )

    date_range = None

    if (
        not events_df.empty
        and "timestamp" in events_df.columns
        and events_df["timestamp"].notna().any()
    ):
        min_d = events_df["timestamp"].min()
        max_d = events_df["timestamp"].max()

        date_range = st.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

    st.markdown("---")

    gallery_cols = st.slider(
        "Gallery columns",
        min_value=2,
        max_value=6,
        value=3,
    )

    max_shots = st.slider(
        "Max screenshots shown",
        min_value=3,
        max_value=48,
        value=12,
        step=3,
    )

    st.markdown("---")
    st.caption("AI Vision Control Center")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# =========================
# Apply Filters
# =========================
filtered_df = events_df.copy()

if not filtered_df.empty:
    if selected_types:
        filtered_df = filtered_df[filtered_df["event_type"].isin(selected_types)]

    if (
        date_range
        and isinstance(date_range, tuple)
        and len(date_range) == 2
        and "timestamp" in filtered_df.columns
    ):
        start_d, end_d = date_range

        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_d)
            & (filtered_df["timestamp"].dt.date <= end_d)
        ]


# =========================
# Hero Header
# =========================
is_live = len(events_df) > 0 and EVENT_LOG_PATH.exists()

pill_html = (
    '<span class="status-pill pill-live">● LIVE FEED</span>'
    if is_live
    else '<span class="status-pill pill-idle">● IDLE</span>'
)

st.markdown(
    f"""
    <div class="hero">
        <h1>👁️ VisionX AI</h1>
        <p>{pill_html} Real-time computer vision dashboard — object detection, human counting,
        hand gesture recognition, smart actions, screenshots, and event logging.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Metrics
# =========================
total_events = len(filtered_df)
total_screenshots = len(screenshots)

system_started_count = (
    filtered_df["event_type"].eq("system_started").sum()
    if not filtered_df.empty
    else 0
)

object_events_count = (
    filtered_df["event_type"].eq("objects_detected").sum()
    if not filtered_df.empty
    else 0
)

all_objects_full = (
    extract_objects(filtered_df["details"])
    if not filtered_df.empty
    else []
)

unique_object_types = len(set(all_objects_full))

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Events", f"{total_events:,}")
col2.metric("Screenshots", f"{total_screenshots:,}")
col3.metric("System Runs", f"{system_started_count:,}")
col4.metric("Detection Events", f"{object_events_count:,}")
col5.metric("Unique Object Types", f"{unique_object_types:,}")


# =========================
# Tabs
# =========================
st.markdown("<br>", unsafe_allow_html=True)

tab_camera, tab_overview, tab_objects, tab_gallery, tab_logs = st.tabs(
    [
        "🎥 Live Camera",
        "📊 Overview",
        "🎯 Object Detection",
        "📸 Screenshots",
        "🧾 Event Log",
    ]
)


# =========================
# Live Camera Tab
# =========================
with tab_camera:
    st.markdown(
        '<div class="section-title">Live Camera Control</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Start or stop the real-time AI camera system from the dashboard.

        The camera will open in a separate OpenCV window and will run:
        - YOLO object detection
        - People counting
        - Hand gesture recognition
        - Finger drawing mode
        - Screenshot saving
        - Event logging
        """
    )

    if "camera_process" not in st.session_state:
        st.session_state.camera_process = None

    camera_running = (
        st.session_state.camera_process is not None
        and st.session_state.camera_process.poll() is None
    )

    col_start, col_stop, col_status = st.columns(3)

    with col_start:
        if st.button("▶️ Start Camera", use_container_width=True):
            if camera_running:
                st.warning("Camera is already running.")
            else:
                if not CAMERA_SCRIPT_PATH.exists():
                    st.error(f"Camera script not found: {CAMERA_SCRIPT_PATH}")
                else:
                    st.session_state.camera_process = subprocess.Popen(
                        [sys.executable, str(CAMERA_SCRIPT_PATH)],
                        cwd=str(ROOT_DIR),
                    )
                    st.success("Camera started successfully.")

    with col_stop:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            if camera_running:
                st.session_state.camera_process.terminate()

                try:
                    st.session_state.camera_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    st.session_state.camera_process.kill()

                st.session_state.camera_process = None
                st.success("Camera stopped.")
            else:
                st.info("Camera is not running.")

    with col_status:
        if camera_running:
            st.success("Status: Running")
        else:
            st.error("Status: Stopped")

    st.markdown("---")

    st.subheader("Camera Window Controls")

    st.write(
        """
        - **Index Finger** → Draw with your finger
        - **Open Palm** → Pause drawing / Active mode
        - **Peace Sign** → Take screenshot
        - **Fist** → Exit camera after holding
        - **C** → Clear drawing
        - **Q** → Quit camera window
        """
    )

    st.info("After using the camera, click Refresh Data from the sidebar to update logs and screenshots.")


# =========================
# Overview Tab
# =========================
with tab_overview:
    st.markdown(
        '<div class="section-title">Event Type Distribution</div>',
        unsafe_allow_html=True,
    )

    if not filtered_df.empty:
        c1, c2 = st.columns([2, 1])

        event_counts = filtered_df["event_type"].value_counts().reset_index()
        event_counts.columns = ["event_type", "count"]

        with c1:
            fig_events = px.bar(
                event_counts,
                x="event_type",
                y="count",
                text="count",
                color="event_type",
                title=None,
            )

            fig_events.update_layout(
                template="plotly_dark",
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_events, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                event_counts,
                names="event_type",
                values="count",
                hole=0.55,
            )

            fig_pie.update_layout(
                template="plotly_dark",
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        if (
            "timestamp" in filtered_df.columns
            and filtered_df["timestamp"].notna().any()
        ):
            st.markdown(
                '<div class="section-title">Activity Timeline</div>',
                unsafe_allow_html=True,
            )

            ts_df = filtered_df.dropna(subset=["timestamp"]).copy()

            # Fixed: lowercase h instead of uppercase H
            ts_df["bucket"] = ts_df["timestamp"].dt.floor("h")

            timeline = (
                ts_df.groupby(["bucket", "event_type"])
                .size()
                .reset_index(name="count")
            )

            fig_timeline = px.area(
                timeline,
                x="bucket",
                y="count",
                color="event_type",
                groupnorm=None,
            )

            fig_timeline.update_layout(
                template="plotly_dark",
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25),
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

    else:
        st.warning("No events found yet. Run the camera system first.")


# =========================
# Object Detection Tab
# =========================
with tab_objects:
    st.markdown(
        '<div class="section-title">Object Detection Logs</div>',
        unsafe_allow_html=True,
    )

    object_df = filtered_df[filtered_df["event_type"] == "objects_detected"].copy()

    if not object_df.empty:
        with st.expander("📄 Recent detection records", expanded=True):
            st.dataframe(
                object_df.tail(20),
                use_container_width=True,
                height=300,
            )

        all_objects = extract_objects(object_df["details"])

        if all_objects:
            objects_count_df = pd.Series(all_objects).value_counts().reset_index()
            objects_count_df.columns = ["object", "count"]

            c1, c2 = st.columns([2, 1])

            with c1:
                fig_objects = px.bar(
                    objects_count_df.head(20),
                    x="count",
                    y="object",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Tealgrn",
                )

                fig_objects.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    template="plotly_dark",
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig_objects, use_container_width=True)

            with c2:
                st.markdown("##### 🏆 Top Detected")

                for _, row in objects_count_df.head(8).iterrows():
                    st.markdown(f"**{row['object']}** — {row['count']}")

    else:
        st.info("No object detection events logged yet.")


# =========================
# Screenshots Tab
# =========================
with tab_gallery:
    st.markdown(
        '<div class="section-title">Screenshot Gallery</div>',
        unsafe_allow_html=True,
    )

    if screenshots:
        shown = screenshots[:max_shots]
        cols = st.columns(gallery_cols)

        for idx, screenshot_path in enumerate(shown):
            with cols[idx % gallery_cols]:
                st.image(
                    str(screenshot_path),
                    caption=screenshot_path.name,
                    use_container_width=True,
                )

        st.caption(f"Showing {len(shown)} of {len(screenshots)} screenshots")

    else:
        st.info(
            "No screenshots saved yet. Use Peace Sign in the camera app to save screenshots."
        )


# =========================
# Event Log Tab
# =========================
with tab_logs:
    st.markdown(
        '<div class="section-title">Full Event Log</div>',
        unsafe_allow_html=True,
    )

    if not filtered_df.empty:
        search_term = st.text_input(
            "🔎 Search in details",
            placeholder="Type to filter rows...",
        )

        log_view = filtered_df.sort_values(by="timestamp", ascending=False)

        if search_term:
            log_view = log_view[
                log_view["details"]
                .astype(str)
                .str.contains(search_term, case=False, na=False)
            ]

        st.dataframe(
            log_view,
            use_container_width=True,
            height=450,
        )

        csv_data = log_view.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download filtered log as CSV",
            data=csv_data,
            file_name="vision_events_filtered.csv",
            mime="text/csv",
        )

    else:
        st.info("No logs available yet.")


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("VisionX AI | Python + OpenCV + YOLO + MediaPipe + Streamlit")