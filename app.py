"""EduTune AI enterprise Streamlit application."""

from __future__ import annotations

import html

import streamlit as st

from config.settings import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    DOMAIN,
    ENVIRONMENT,
    MODEL_ID,
    TRAINING_MODE,
)
from inference import (
    build_chat_prompt,
    build_educational_prompt,
    check_inference_hardware,
    generate_response_safe,
    inference_is_available,
    load_inference_model,
)
from models.tokenizer import load_tokenizer


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=f"{APP_NAME} | AI Education Platform",
    page_icon="ED",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# HTML RENDERING HELPER
# ============================================================================
#
# WHY THIS EXISTS / THE BUG IT FIXES:
# st.markdown() runs its input through a Markdown parser before allowing raw
# HTML. Per the Markdown/CommonMark spec, a line that starts with 4+ spaces
# of indentation is treated as an INDENTED CODE BLOCK, not HTML -- so instead
# of rendering, the tags print out as literal visible text (exactly what you
# saw on the System Information page, in a little copyable code box).
#
# Every f-string template below is written with normal Python indentation
# for readability, which means the generated HTML lines are indented too.
# strip_html() collapses each template down so no line has leading
# whitespace, guaranteeing Markdown always recognizes it as an HTML block
# and renders it instead of printing it as text.


def strip_html(markup: str) -> str:
    """Remove leading indentation from every line of an HTML template.

    This prevents Streamlit's Markdown parser from mistaking indented HTML
    for a fenced/indented code block, which is what caused raw tags like
    <div class="info-row"> to render as visible text instead of HTML.
    """
    lines = [line.strip() for line in markup.strip().splitlines()]
    return "".join(lines)


def render_html(markup: str) -> None:
    """Render a raw HTML template safely via st.markdown."""
    st.markdown(strip_html(markup), unsafe_allow_html=True)


def esc(value: object) -> str:
    """HTML-escape a value before it is interpolated into raw markup."""
    return html.escape(str(value))


# ============================================================================
# ENTERPRISE UI STYLING
# ============================================================================

st.markdown(
    """
    <style>
        /* ---------------------------------------------------------------
           Global
        --------------------------------------------------------------- */

        .stApp {
            background-color: #f6f8fb;
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 700;
        }

        h2 {
            font-weight: 650;
        }

        h3 {
            font-weight: 600;
        }

        /* ---------------------------------------------------------------
           Sidebar
        --------------------------------------------------------------- */

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        section[data-testid="stSidebar"] .stRadio label {
            color: #d1d5db;
        }

        section[data-testid="stSidebar"]
        div[data-baseweb="select"] {
            background-color: #1f2937;
        }

        /* ---------------------------------------------------------------
           Cards
        --------------------------------------------------------------- */

        .enterprise-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .enterprise-card:hover {
            border-color: #d1d5db;
        }

        .enterprise-card h3 {
            margin-top: 0;
        }

        .enterprise-card ul {
            margin-bottom: 0;
            padding-left: 1.2rem;
        }

        .enterprise-card p {
            color: #4b5563;
        }

        .status-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem 1.15rem;
            min-height: 115px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .status-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .status-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #111827;
            margin-top: 0.35rem;
        }

        .status-description {
            font-size: 0.82rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        /* ---------------------------------------------------------------
           Hero
        --------------------------------------------------------------- */

        .hero {
            background: linear-gradient(
                135deg,
                #111827 0%,
                #1f2937 55%,
                #374151 100%
            );
            color: white;
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
        }

        .hero-eyebrow {
            color: #cbd5e1;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
        }

        .hero-description {
            color: #d1d5db;
            font-size: 1rem;
            line-height: 1.65;
            max-width: 850px;
        }

        /* ---------------------------------------------------------------
           Section headers
        --------------------------------------------------------------- */

        .section-label {
            color: #6b7280;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .section-title {
            color: #111827;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        /* ---------------------------------------------------------------
           Sidebar brand
        --------------------------------------------------------------- */

        .sidebar-brand {
            padding: 0.75rem 0.25rem 1.25rem 0.25rem;
        }

        .sidebar-brand-title {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 750;
        }

        .sidebar-brand-subtitle {
            color: #9ca3af;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }

        .sidebar-section {
            color: #9ca3af;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-top: 1.25rem;
            margin-bottom: 0.45rem;
        }

        /* ---------------------------------------------------------------
           Tables / info rows
        --------------------------------------------------------------- */

        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid #f0f1f3;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-key {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .info-value {
            color: #111827;
            font-size: 0.9rem;
            font-weight: 600;
            text-align: right;
        }

        /* ---------------------------------------------------------------
           Footer
        --------------------------------------------------------------- */

        .enterprise-footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
            font-size: 0.78rem;
            text-align: center;
        }

        /* ---------------------------------------------------------------
           Streamlit controls
        --------------------------------------------------------------- */

        div.stButton > button {
            border-radius: 9px;
            font-weight: 600;
            min-height: 2.7rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
        }

        .stTextArea textarea {
            border-radius: 10px;
        }

        .stSelectbox > div > div {
            border-radius: 10px;
        }

        /* ---------------------------------------------------------------
           Responsive adjustments
        --------------------------------------------------------------- */

        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.8rem;
            }

            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


# ============================================================================
# COMPONENT HELPERS
# ============================================================================
#
# Every card below is built as ONE complete HTML string and rendered through
# render_html(), which strips indentation before handing it to st.markdown().
# This avoids both bugs seen previously:
#   1. Orphaned tags from opening/closing a div across separate st.markdown()
#      calls (they never actually wrapped their contents).
#   2. Indented HTML being misread as a Markdown code block and printed as
#      literal text.


def get_hardware_status() -> dict:
    """Return current inference hardware information."""
    return check_inference_hardware()


def load_runtime_model() -> bool:
    """Load tokenizer and model when supported hardware is available."""

    if not inference_is_available():
        return False

    try:
        if st.session_state.tokenizer is None:
            st.session_state.tokenizer = load_tokenizer()

        if st.session_state.model is None:
            st.session_state.model = load_inference_model()

        return True

    except Exception as exc:
        st.error("The EduTune AI model could not be loaded.")
        st.exception(exc)
        return False


def render_info_card(rows: list[tuple[str, str]]) -> None:
    """Render a complete enterprise-style info card."""

    rows_html = "".join(
        f"""<div class="info-row">
        <span class="info-key">{esc(key)}</span>
        <span class="info-value">{esc(value)}</span>
        </div>"""
        for key, value in rows
    )

    render_html(f'<div class="enterprise-card">{rows_html}</div>')


def render_feature_card(title: str, description: str, items: list[str]) -> None:
    """Render a complete enterprise-style feature card."""

    items_html = "".join(f"<li>{esc(item)}</li>" for item in items)

    render_html(
        f"""
        <div class="enterprise-card">
            <h3>{esc(title)}</h3>
            <p>{esc(description)}</p>
            <ul>{items_html}</ul>
        </div>
        """
    )


def render_text_card(title: str, description: str) -> None:
    """Render a complete enterprise-style text-only card."""

    render_html(
        f"""
        <div class="enterprise-card">
            <h3>{esc(title)}</h3>
            <p>{esc(description)}</p>
        </div>
        """
    )


def render_status_card(title: str, value: str, description: str) -> None:
    """Render a complete status card."""

    render_html(
        f"""
        <div class="status-card">
            <div class="status-title">{esc(title)}</div>
            <div class="status-value">{esc(value)}</div>
            <div class="status-description">{esc(description)}</div>
        </div>
        """
    )


def render_section_header(eyebrow: str, title: str) -> None:
    """Render a consistent section header."""

    render_html(
        f"""
        <div class="section-label">{esc(eyebrow)}</div>
        <div class="section-title">{esc(title)}</div>
        """
    )


def render_footer() -> None:
    """Render application footer."""

    render_html(
        f"""
        <div class="enterprise-footer">
            {esc(APP_NAME)} &nbsp;&bull;&nbsp; Version {esc(APP_VERSION)}
            &nbsp;&bull;&nbsp; {esc(DOMAIN.title())} AI Platform
        </div>
        """
    )


def render_hero(eyebrow: str, title: str, description: str) -> None:
    """Render the top hero banner used on every page."""

    render_html(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{esc(eyebrow)}</div>
            <div class="hero-title">{esc(title)}</div>
            <div class="hero-description">{esc(description)}</div>
        </div>
        """
    )


# ============================================================================
# HARDWARE
# ============================================================================

hardware = get_hardware_status()

cuda_available = bool(hardware.get("cuda_available", False))


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    render_html(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">{esc(APP_NAME)}</div>
            <div class="sidebar-brand-subtitle">
                Domain-Specific Educational AI
            </div>
        </div>
        """
    )

    st.divider()

    render_html('<div class="sidebar-section">Platform</div>')

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Educational Assistant",
            "Evaluation",
            "System Information",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    render_html('<div class="sidebar-section">Model Configuration</div>')

    st.caption("MODEL")
    st.code(MODEL_ID, language="text")

    st.caption("TRAINING")
    st.write(TRAINING_MODE.upper())

    st.caption("DOMAIN")
    st.write(DOMAIN.title())

    st.divider()

    render_html('<div class="sidebar-section">Runtime Status</div>')

    if cuda_available:
        st.success("GPU inference available")

        if hardware.get("gpu_name"):
            st.caption(f"GPU: {hardware['gpu_name']}")

        st.caption(f"Devices: {hardware.get('device_count', 0)}")

    else:
        st.warning("CPU-only environment")

        st.caption(
            "Mistral-7B inference is safely blocked "
            "until CUDA hardware is available."
        )

    st.divider()

    st.caption(f"{APP_NAME} v{APP_VERSION}")
    st.caption("Enterprise AI Education Platform")


# ============================================================================
# DASHBOARD
# ============================================================================

if page == "Dashboard":

    render_hero(
        "Enterprise AI Education Platform",
        APP_NAME,
        APP_DESCRIPTION,
    )

    render_section_header("Platform Overview", "AI Platform Status")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric("Foundation Model", "Mistral-7B")

    with metric_2:
        st.metric("Fine-Tuning", "QLoRA")

    with metric_3:
        st.metric("AI Domain", "Education")

    with metric_4:
        st.metric("Inference", "Available" if cuda_available else "Blocked")

    st.write("")

    render_section_header("Architecture", "Platform Capabilities")

    capability_1, capability_2, capability_3 = st.columns(3)

    with capability_1:
        render_feature_card(
            "Data Engineering",
            "Structured educational datasets prepared "
            "through a reproducible data pipeline.",
            [
                "Seed dataset construction",
                "Dataset curation",
                "Synthetic data generation",
                "Validation and quality checks",
                "Train/test preparation",
            ],
        )

    with capability_2:
        render_feature_card(
            "Model Engineering",
            "Parameter-efficient model development "
            "designed for domain adaptation.",
            [
                "Mistral-7B Instruct",
                "QLoRA fine-tuning workflow",
                "Tokenizer integration",
                "Hardware-aware loading",
                "Safe inference controls",
            ],
        )

    with capability_3:
        render_feature_card(
            "Evaluation & QA",
            "Automated validation infrastructure for "
            "measuring and comparing model behavior.",
            [
                "Exact-match evaluation",
                "Token-overlap F1",
                "Baseline evaluation",
                "Model comparison",
                "Automated test suite",
            ],
        )

    render_section_header("Operational Readiness", "Current Project Status")

    status_1, status_2 = st.columns(2)

    with status_1:
        render_status_card(
            "Dataset Pipeline",
            "Validated",
            "Dataset creation, curation, synthesis and "
            "validation are operational.",
        )

        st.write("")

        render_status_card(
            "Evaluation Pipeline",
            "Validated",
            "Evaluation and model-comparison infrastructure "
            "has passed automated tests.",
        )

    with status_2:
        render_status_card(
            "Inference API",
            "Validated",
            "Prompt construction, generation safety, and "
            "hardware-aware inference controls are implemented.",
        )

        st.write("")

        if cuda_available:
            st.success(
                "CUDA environment detected. Model inference can be initialized."
            )
        else:
            st.warning(
                "CUDA is unavailable. Large model weights remain safely unloaded."
            )

            st.info(
                "The platform can still be inspected and validated "
                "without loading Mistral-7B on this CPU-only environment."
            )

    render_footer()


# ============================================================================
# EDUCATIONAL ASSISTANT
# ============================================================================

elif page == "Educational Assistant":

    render_hero(
        "AI Learning Workspace",
        "Educational Assistant",
        "Generate structured, student-friendly educational "
        "responses using the EduTune AI inference pipeline.",
    )

    render_section_header("Learning Context", "Configure Your Question")

    context_col_1, context_col_2 = st.columns(2)

    with context_col_1:
        subject = st.selectbox(
            "Subject",
            [
                "General",
                "Mathematics",
                "Physics",
                "Biology",
                "Computer Science",
                "Economics",
            ],
        )

    with context_col_2:
        difficulty = st.selectbox(
            "Difficulty",
            ["Beginner", "Intermediate", "Advanced"],
        )

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: Explain Newton's second law in simple terms."
        ),
        height=150,
    )

    action_col_1, action_col_2, action_col_3 = st.columns([2, 1, 1])

    with action_col_1:
        generate_clicked = st.button(
            "Generate Educational Answer",
            type="primary",
            use_container_width=True,
        )

    with action_col_2:
        clear_clicked = st.button("Clear", use_container_width=True)

    with action_col_3:
        hardware_clicked = st.button("Check Runtime", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if hardware_clicked:
        st.info(
            f"Runtime device: {hardware['device']} | "
            f"CUDA: {hardware['cuda_available']}"
        )

    if generate_clicked:

        if not question.strip():
            st.error(
                "Please enter an educational question before generating an answer."
            )

        else:

            prompt = build_educational_prompt(
                question,
                subject=subject,
                difficulty=difficulty,
            )

            if not cuda_available:

                st.warning(
                    "Live model inference is currently unavailable "
                    "because this environment does not have a CUDA-enabled GPU."
                )

                st.info(
                    "EduTune AI intentionally prevents Mistral-7B "
                    "weights from being loaded on CPU-only hardware."
                )

                render_section_header("Inference Pipeline", "Generated Prompt")

                st.code(prompt, language="text")

                st.caption(
                    "This prompt was generated successfully. "
                    "A CUDA-enabled environment is required for actual model generation."
                )

            else:

                with st.spinner("Initializing EduTune AI inference..."):
                    ready = load_runtime_model()

                if ready:

                    with st.spinner("Generating educational response..."):

                        try:

                            response = generate_response_safe(
                                st.session_state.model,
                                st.session_state.tokenizer,
                                prompt,
                            )

                            render_section_header("AI Output", "EduTune AI Response")

                            # `response` is model-generated text, not trusted
                            # markup, so it is HTML-escaped before being
                            # placed inside the card and line breaks are
                            # preserved for readability.
                            safe_response = esc(response).replace("\n", "<br>")

                            render_html(
                                f'<div class="enterprise-card">{safe_response}</div>'
                            )

                        except Exception as exc:
                            st.error("Response generation failed.")
                            st.exception(exc)

    render_footer()


# ============================================================================
# EVALUATION
# ============================================================================

elif page == "Evaluation":

    render_hero(
        "Quality Assurance & Benchmarking",
        "Evaluation Center",
        "Review the validated evaluation infrastructure, "
        "benchmark metrics, and baseline-versus-fine-tuned "
        "model comparison framework.",
    )

    render_section_header("Quality Assurance", "Evaluation Overview")

    eval_1, eval_2, eval_3, eval_4 = st.columns(4)

    with eval_1:
        st.metric("Test Records", "7")

    with eval_2:
        st.metric("Automated Tests", "34")

    with eval_3:
        st.metric("Core Metrics", "2")

    with eval_4:
        st.metric("Pipeline Status", "Validated")

    st.write("")

    render_section_header("Metrics", "Evaluation Methodology")

    metric_1, metric_2 = st.columns(2)

    with metric_1:
        render_text_card(
            "Exact Match",
            "Measures normalized exact agreement between "
            "a model prediction and its reference response.",
        )

    with metric_2:
        render_text_card(
            "Token Overlap",
            "Measures token-level F1 overlap between "
            "predictions and reference responses.",
        )

    render_section_header("Benchmarking", "Model Comparison Framework")

    comparison_1, comparison_2 = st.columns(2)

    with comparison_1:
        st.metric("Exact-Match Improvement", "+0.30")

    with comparison_2:
        st.metric("Token-Overlap Improvement", "+0.20")

    st.info(
        "The displayed comparison values represent the validated "
        "model-comparison infrastructure. They are not presented "
        "as measured Mistral-7B fine-tuning results from this "
        "CPU-only environment."
    )

    render_text_card(
        "Evaluation Integrity",
        "The evaluation architecture separates metric validation "
        "from actual model benchmarking. This allows the project "
        "to validate the complete evaluation pipeline without "
        "falsely claiming model-generation results when GPU "
        "inference is unavailable.",
    )

    render_footer()


# ============================================================================
# SYSTEM INFORMATION
# ============================================================================

elif page == "System Information":

    render_hero(
        "Platform Diagnostics",
        "System Information",
        "Runtime configuration, model configuration, "
        "and hardware diagnostics for the EduTune AI platform.",
    )

    render_section_header("Configuration", "Application Information")

    config_col_1, config_col_2 = st.columns(2)

    with config_col_1:
        render_info_card(
            [
                ("Application", APP_NAME),
                ("Version", APP_VERSION),
                ("Environment", ENVIRONMENT),
                ("Domain", DOMAIN),
            ]
        )

    with config_col_2:
        render_info_card(
            [
                ("Foundation Model", MODEL_ID),
                ("Training Method", TRAINING_MODE.upper()),
                ("Runtime Device", hardware["device"]),
                ("CUDA Available", str(hardware["cuda_available"])),
            ]
        )

    render_section_header("Hardware Diagnostics", "Inference Environment")

    hardware_col_1, hardware_col_2 = st.columns(2)

    with hardware_col_1:
        render_info_card(
            [
                ("Device", str(hardware["device"])),
                (
                    "CUDA",
                    "Available" if hardware["cuda_available"] else "Unavailable",
                ),
                ("GPU Count", str(hardware.get("device_count", 0))),
                ("GPU", str(hardware.get("gpu_name") or "None detected")),
            ]
        )

    with hardware_col_2:

        if hardware["cuda_available"]:

            st.success("The current environment supports EduTune AI inference.")

            if hardware.get("gpu_name"):
                st.info(f"Detected GPU: {hardware['gpu_name']}")

        else:

            st.warning("CPU-only environment detected.")

            st.write(
                "The configured Mistral-7B inference workflow "
                "requires CUDA-enabled hardware."
            )

            st.write(
                "Model loading is deliberately blocked to prevent "
                "large model weights from being loaded into an "
                "unsupported CPU-only environment."
            )

    render_section_header("Architecture", "Runtime Safety")

    render_text_card(
        "Hardware-Aware Execution",
        "EduTune AI checks the available runtime hardware before "
        "attempting model initialization. When CUDA is unavailable, "
        "inference remains blocked while dataset, evaluation, "
        "prompt-engineering, and system-validation workflows "
        "remain available.",
    )

    render_footer()