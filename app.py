import os

import requests
import streamlit as st

API_URL = os.getenv("FACE_API_URL", "http://127.0.0.1:8000")


def _api_error_message(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return str(payload.get("detail", ""))
            return str(payload)
        except ValueError:
            return response.text
    return f"Could not reach the face API at {API_URL}."


def _request_analysis(image_bytes: bytes, image_type: str, filename: str) -> list[dict]:
    response = requests.post(
        f"{API_URL.rstrip('/')}/analyze",
        files={"image": (filename, image_bytes, image_type)},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["face_details"]


def _show_face_details(title: str, face_details: list[dict]) -> None:
    st.subheader(title)
    for index, face in enumerate(face_details, start=1):
        with st.expander(f"Face {index} · All analysis details", expanded=True):
            st.json(face, expanded=True)


def _show_analysis(prefix: str, title: str) -> bool:
    analysis_key = f"{prefix}_analysis"
    error_key = f"{prefix}_analysis_error"
    if st.session_state[analysis_key] is None and st.session_state[error_key] is None:
        image_bytes = st.session_state[f"{prefix}_bytes"]
        image_type = st.session_state[f"{prefix}_type"]
        try:
            with st.spinner(f"Detecting and analyzing the {title.lower()}…"):
                st.session_state[analysis_key] = _request_analysis(
                    image_bytes,
                    image_type,
                    f"{prefix}.jpg",
                )
        except requests.RequestException as exc:
            st.session_state[error_key] = _api_error_message(exc)
        except (KeyError, ValueError) as exc:
            st.session_state[error_key] = f"Unexpected API response: {exc}"

    if st.session_state[error_key]:
        st.error(st.session_state[error_key])
        if st.button("Retry face analysis", key=f"{prefix}_analysis_retry"):
            st.session_state[error_key] = None
            st.rerun()
        return False

    _show_face_details(title, st.session_state[analysis_key])
    return True


st.set_page_config(page_title="Face Match", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2.2rem; }
    .eyebrow { color: #176b5b; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">FACE VERIFICATION / DEEPFACE</div>', unsafe_allow_html=True)
st.title("Compare two faces")
st.caption("Capture two photos in sequence. For phone camera access, open this app over HTTPS and allow permission.")
st.caption("DeepFace compares the captured pair; `pyproject.toml` is project configuration, not a training-image folder.")
for key, default in {
    "reference_bytes": None,
    "reference_type": "image/jpeg",
    "reference_analysis": None,
    "reference_analysis_error": None,
    "candidate_bytes": None,
    "candidate_type": "image/jpeg",
    "candidate_analysis": None,
    "candidate_analysis_error": None,
    "reference_round": 0,
    "candidate_round": 0,
    "match_result": None,
}.items():
    st.session_state.setdefault(key, default)

if st.session_state.reference_bytes is None:
    st.subheader("Step 1 · Capture reference photo")
    reference_capture = st.camera_input(
        "Open camera for the first photo",
        key=f"reference_capture_{st.session_state.reference_round}",
    )
    if reference_capture:
        st.session_state.reference_bytes = reference_capture.getvalue()
        st.session_state.reference_type = reference_capture.type or "image/jpeg"
        st.session_state.reference_analysis = None
        st.session_state.reference_analysis_error = None
        st.rerun()
    st.info("Take the first photo. It will be kept in this session while you capture the second.")
elif st.session_state.candidate_bytes is None:
    st.subheader("Step 2 · Analyze reference and capture comparison photo")
    st.image(st.session_state.reference_bytes, caption="Saved reference photo", width="stretch")
    st.caption("Reference photo is stored in session memory only; it is not saved to disk.")

    if st.button("Retake reference photo"):
        st.session_state.reference_round += 1
        st.session_state.candidate_round += 1
        st.session_state.reference_bytes = None
        st.session_state.reference_analysis = None
        st.session_state.reference_analysis_error = None
        st.session_state.candidate_bytes = None
        st.session_state.candidate_analysis = None
        st.session_state.candidate_analysis_error = None
        st.session_state.match_result = None
        st.rerun()

    if _show_analysis("reference", "Reference face analysis"):
        st.subheader("Capture comparison photo")
        candidate_capture = st.camera_input(
            "Open camera for the second photo",
            key=f"candidate_capture_{st.session_state.candidate_round}",
        )
        if candidate_capture:
            st.session_state.candidate_bytes = candidate_capture.getvalue()
            st.session_state.candidate_type = candidate_capture.type or "image/jpeg"
            st.session_state.candidate_analysis = None
            st.session_state.candidate_analysis_error = None
            st.session_state.match_result = None
            st.rerun()
else:
    st.subheader("Step 3 · Analyze second photo and compare")
    reference_column, candidate_column = st.columns(2, gap="large")
    with reference_column:
        st.image(st.session_state.reference_bytes, caption="Reference photo", width="stretch")
    with candidate_column:
        st.image(st.session_state.candidate_bytes, caption="Second photo", width="stretch")

    detail_left, detail_right = st.columns(2, gap="large")
    with detail_left:
        _show_face_details("Reference face analysis", st.session_state.reference_analysis or [])
    with detail_right:
        candidate_analyzed = _show_analysis("candidate", "Second photo analysis")

    reset_column, retake_column, verify_column = st.columns([1, 1, 2])
    with reset_column:
        if st.button("Start over"):
            st.session_state.reference_round += 1
            st.session_state.candidate_round += 1
            st.session_state.reference_bytes = None
            st.session_state.reference_analysis = None
            st.session_state.reference_analysis_error = None
            st.session_state.candidate_bytes = None
            st.session_state.candidate_analysis = None
            st.session_state.candidate_analysis_error = None
            st.session_state.match_result = None
            st.rerun()
    with retake_column:
        if st.button("Retake second photo"):
            st.session_state.candidate_round += 1
            st.session_state.candidate_bytes = None
            st.session_state.candidate_analysis = None
            st.session_state.candidate_analysis_error = None
            st.session_state.match_result = None
            st.rerun()
    with verify_column:
        verify_clicked = st.button("Verify faces", type="primary", disabled=not candidate_analyzed)

    if verify_clicked:
        try:
            with st.spinner("Checking the images… the first request may take longer while models initialize."):
                response = requests.post(
                    f"{API_URL.rstrip('/')}/verify",
                    files={
                        "reference": (
                            "reference.jpg",
                            st.session_state.reference_bytes,
                            st.session_state.reference_type,
                        ),
                        "candidate": (
                            "candidate.jpg",
                            st.session_state.candidate_bytes,
                            st.session_state.candidate_type,
                        ),
                    },
                    timeout=300,
                )
            response.raise_for_status()
            st.session_state.match_result = response.json()
        except requests.RequestException as exc:
            st.error(_api_error_message(exc))
        except (KeyError, ValueError) as exc:
            st.error(f"The API returned an unexpected response: {exc}")

    result = st.session_state.match_result
    if result is not None:
        st.divider()
        if result["verified"]:
            st.success("Match found")
        else:
            st.warning("No match found")

        with st.expander("Verification details", expanded=True):
            st.json(result.get("verification", {}))
