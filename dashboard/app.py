
import os
from pathlib import Path

import requests
import streamlit as st


API_URL = os.getenv(
    "INSPECTAI_API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")


st.set_page_config(
    page_title="InspectAI",
    page_icon="🔍",
    layout="wide",
)


def api_request(method, endpoint, **kwargs):
    """Send a request to the InspectAI FastAPI service."""
    kwargs.setdefault("timeout", 180)

    response = requests.request(
        method,
        f"{API_URL}{endpoint}",
        **kwargs,
    )

    response.raise_for_status()
    return response.json()


def show_api_error(error):
    """Display a readable API error."""
    if isinstance(error, requests.HTTPError):
        response = error.response

        if response is not None:
            try:
                detail = response.json().get("detail")
                if detail:
                    st.error(f"API error: {detail}")
                    return
            except ValueError:
                pass

    st.error(f"Request failed: {error}")


def show_decision(decision):
    """Display the inspection decision."""
    decision = str(decision).upper()

    if decision in {"PASS", "NORMAL"}:
        st.success(f"Decision: {decision}")
    elif decision in {"FAIL", "DEFECTIVE"}:
        st.error(f"Decision: {decision}")
    else:
        st.info(f"Decision: {decision}")


def show_image_inspection():
    st.header("2D Image Inspection")
    st.caption(
        "Detect visual anomalies using the PatchCore model."
    )

    uploaded_file = st.file_uploader(
        "Upload a bottle image",
        type=["png", "jpg", "jpeg"],
        key="image_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        width=450,
    )

    if st.button(
        "Inspect Image",
        type="primary",
        key="inspect_image",
    ):
        with st.spinner("Running image inspection..."):
            try:
                report = api_request(
                    "POST",
                    "/inspect",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/png",
                        )
                    },
                )
            except requests.RequestException as error:
                show_api_error(error)
                return

        st.subheader("Inspection Results")

        show_decision(
            report.get("classification", "UNKNOWN")
        )

        col1, col2, col3 = st.columns(3)

        score = report.get("anomaly_score")
        defective_pixels = report.get("defective_pixels")
        defect_area = report.get("defect_area_percentage")

        col1.metric(
            "Anomaly Score",
            f"{float(score):.4f}" if score is not None else "N/A",
        )

        col2.metric(
            "Defective Pixels",
            defective_pixels if defective_pixels is not None else "N/A",
        )

        col3.metric(
            "Defect Area",
            f"{float(defect_area):.2f}%"
            if defect_area is not None
            else "N/A",
        )

        with st.expander("Full Image Inspection Report"):
            st.json(report)


def show_pointcloud_inspection():
    st.header("3D Point Cloud Inspection")
    st.caption(
        "Inspect cylindrical pipe surfaces for inward defects."
    )

    st.info(
        "The scan must be pre-aligned with the reference pipe. "
        "Coordinates and tolerances are assumed to be in millimeters."
    )

    uploaded_file = st.file_uploader(
        "Upload a PLY point cloud",
        type=["ply"],
        key="pointcloud_upload",
    )

    tolerance = st.number_input(
        "Defect tolerance (mm)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )

    if uploaded_file is None:
        st.info("Upload a point cloud to begin.")
        return

    if st.button(
        "Inspect Point Cloud",
        type="primary",
        key="inspect_pointcloud",
    ):
        with st.spinner("Running 3D inspection..."):
            try:
                report = api_request(
                    "POST",
                    "/inspect/3d",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/octet-stream",
                        )
                    },
                    data={
                        "tolerance_mm": tolerance,
                    },
                )
            except requests.RequestException as error:
                show_api_error(error)
                return

        st.subheader("Inspection Results")

        quality = report.get("quality_control", {})
        measurements = report.get("measurement", {})

        show_decision(quality.get("decision", "UNKNOWN"))

        max_deviation = measurements.get(
            "max_inward_deviation_mm"
        )
        defective_points = quality.get(
            "defective_points"
        )
        defect_percentage = quality.get(
            "defective_percentage"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Maximum Inward Deviation",
            f"{float(max_deviation):.3f} mm"
            if max_deviation is not None
            else "N/A",
        )

        col2.metric(
            "Defective Points",
            defective_points
            if defective_points is not None
            else "N/A",
        )

        col3.metric(
            "Defective Percentage",
            f"{float(defect_percentage):.2f}%"
            if defect_percentage is not None
            else "N/A",
        )

        with st.expander("Full 3D Inspection Report"):
            st.json(report)


def show_video_inspection():
    st.header("Video Inspection")
    st.caption(
        "Sample video frames and detect visual anomalies."
    )

    uploaded_file = st.file_uploader(
        "Upload an MP4 video",
        type=["mp4"],
        key="video_upload",
    )

    sample_interval = st.number_input(
        "Sampling interval (seconds)",
        min_value=0.1,
        max_value=60.0,
        value=1.0,
        step=0.1,
    )

    if uploaded_file is None:
        st.info("Upload a video to begin.")
        return

    st.video(uploaded_file)

    if st.button(
        "Inspect Video",
        type="primary",
        key="inspect_video",
    ):
        with st.spinner(
            "Running video inspection. This may take some time..."
        ):
            try:
                report = api_request(
                    "POST",
                    "/inspect/video",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "video/mp4",
                        )
                    },
                    data={
                        "sample_interval_seconds": sample_interval,
                    },
                )
            except requests.RequestException as error:
                show_api_error(error)
                return

        st.subheader("Inspection Results")

        show_decision(report.get("decision", "UNKNOWN"))

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Inspected Frames",
            report.get("inspected_frames", "N/A"),
        )

        col2.metric(
            "Defective Frames",
            report.get("defective_frames", "N/A"),
        )

        fps = report.get("fps")

        col3.metric(
            "Video FPS",
            f"{float(fps):.2f}" if fps is not None else "N/A",
        )

        if report.get("inspection_truncated"):
            st.warning(
                "Inspection was truncated at the maximum frame limit."
            )

        predictions = report.get("predictions", [])

        if predictions:
            st.subheader("Frame-by-Frame Predictions")
            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Full Video Inspection Report"):
            st.json(report)


def show_inspection_history():
    st.header("Inspection History")
    st.caption(
        "Browse saved 2D, 3D, and video inspection reports."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        limit = st.selectbox(
            "Number of records",
            options=[10, 25, 50, 100],
            index=1,
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("Refresh History"):
            st.rerun()

    try:
        response = api_request(
            "GET",
            "/inspections",
            params={"limit": limit},
            timeout=15,
        )

        inspections = response.get("inspections", [])

    except requests.RequestException as error:
        show_api_error(error)
        return

    if not inspections:
        st.info(
            "No inspections found. Run an inspection first."
        )
        return

    st.metric("Saved Inspections Shown", len(inspections))

    inspection_types = sorted(
        {
            item.get("inspection_type", "Unknown")
            for item in inspections
        }
    )

    selected_type = st.selectbox(
        "Filter by inspection type",
        options=["All"] + inspection_types,
    )

    if selected_type == "All":
        filtered = inspections
    else:
        filtered = [
            item
            for item in inspections
            if item.get("inspection_type") == selected_type
        ]

    st.subheader("Inspection Records")

    if not filtered:
        st.info("No records match the selected filter.")
        return

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    selected_id = st.selectbox(
        "Select an inspection to view the full report",
        options=[item["id"] for item in filtered],
        format_func=lambda inspection_id: (
            f"Inspection #{inspection_id}"
        ),
    )

    if selected_id is None:
        return

    try:
        detail = api_request(
            "GET",
            f"/inspections/{selected_id}",
            timeout=15,
        )

    except requests.RequestException as error:
        show_api_error(error)
        return

    st.divider()
    st.subheader(f"Inspection #{selected_id}")

    show_decision(detail.get("decision", "UNKNOWN"))

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Type:** {detail.get('inspection_type', 'N/A')}"
        )
        st.write(
            f"**Filename:** {detail.get('filename', 'N/A')}"
        )

    with col2:
        st.write(
            f"**Timestamp (UTC):** "
            f"{detail.get('timestamp_utc', 'N/A')}"
        )

        anomaly_score = detail.get("anomaly_score")

        if anomaly_score is not None:
            st.write(
                f"**Anomaly Score:** {float(anomaly_score):.4f}"
            )

    st.subheader("Complete Inspection Report")
    st.json(detail.get("report", {}))


def main():
    st.title("InspectAI")
    st.caption(
        "Industrial Computer Vision and Quality Inspection Platform"
    )

    st.sidebar.title("InspectAI")
    st.sidebar.caption("Inspection Control Center")

    try:
        health = api_request(
            "GET",
            "/health",
            timeout=5,
        )

        if health.get("model_loaded"):
            st.sidebar.success("API Online — Model Loaded")
        else:
            st.sidebar.warning(
                "API Online — Model Not Loaded"
            )

    except requests.RequestException:
        st.sidebar.error("API Offline")

    inspection_type = st.sidebar.radio(
        "Inspection type",
        [
            "2D Image",
            "3D Point Cloud",
            "Video",
            "Inspection History",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"API: {API_URL}")

    if inspection_type == "2D Image":
        show_image_inspection()

    elif inspection_type == "3D Point Cloud":
        show_pointcloud_inspection()

    elif inspection_type == "Video":
        show_video_inspection()

    elif inspection_type == "Inspection History":
        show_inspection_history()


if __name__ == "__main__":
    main()
