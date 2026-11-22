from __future__ import annotations

import asyncio

import streamlit as st
from googletrans import LANGUAGES, Translator

MAX_CHARACTERS = 15_000  # googletrans documents this web-client limit

st.set_page_config(
    page_title="Lingua — Machine Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#14213d; --muted:#64748b; --line:#e5eaf2; --blue:#3559e0; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#f7f9ff 0%,#ffffff 55%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(247,249,255,.86); }
    .block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
    .hero { padding:1rem 0 1.5rem; }
    .eyebrow { display:inline-block; color:#3559e0; background:#e9edff; border-radius:99px; padding:.38rem .72rem; font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2.2rem,5vw,3.5rem); line-height:1.08; margin:.8rem 0 .65rem; }
    .hero p { color:#64748b; font-size:1.05rem; max-width:650px; margin:0; }
    [data-testid="stTextArea"] textarea { border:1px solid var(--line); border-radius:16px; background:#fff; padding:1rem; font-size:1rem; line-height:1.6; }
    [data-testid="stTextArea"] textarea:focus { border-color:#7890ff; box-shadow:0 0 0 3px #3559e01c; }
    [data-testid="stSelectbox"] > div > div { border-radius:12px; }
    div.stButton > button, div.stDownloadButton > button { border-radius:12px; min-height:2.8rem; font-weight:700; }
    div.stButton > button[kind="primary"] { background:#3559e0; border-color:#3559e0; }
    [data-testid="stSidebar"] { background:#f2f5fb; border-right:1px solid #e5eaf2; }
    .note { background:#f0f5ff; border:1px solid #dce6ff; border-radius:14px; padding:1rem 1.1rem; color:#34456d; font-size:.9rem; line-height:1.55; }
    .metric { color:#64748b; font-size:.82rem; }
    hr { border-color:#e8edf5; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "source_language" not in st.session_state:
    st.session_state.source_language = "Auto-detect"
if "target_language" not in st.session_state:
    st.session_state.target_language = "English"
if "translation_info" not in st.session_state:
    st.session_state.translation_info = ""

# googletrans exposes language codes mapped to English language names.
LANGUAGE_NAMES = {code: name.title() for code, name in LANGUAGES.items()}
LANGUAGE_NAMES.setdefault("en", "English")
NAME_TO_CODE = {name: code for code, name in LANGUAGE_NAMES.items()}
LANGUAGE_OPTIONS = sorted(NAME_TO_CODE, key=str.casefold)


def swap_languages() -> None:
    source = st.session_state.get("source_language", "Auto-detect")
    target = st.session_state.get("target_language", "English")
    if source == "Auto-detect":
        st.session_state.swap_warning = True
        return
    st.session_state.source_language = target
    st.session_state.target_language = source
    st.session_state.swap_warning = False


async def get_translation(text: str, source_code: str, target_code: str):
    # The public Google Translate web endpoint does not require a Cloud API key.
    async with Translator(service_urls=["translate.googleapis.com"]) as translator:
        return await translator.translate(text, src=source_code, dest=target_code)


st.markdown(
    """
    <div class="hero">
      <span class="eyebrow">Translation, made simple</span>
      <h1>Say it in another language.</h1>
      <p>Translate a sentence, a message, or a longer passage. Pick a source language or let Google Translate detect it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🌐 Lingua")
    st.caption("Machine translation workspace")
    st.markdown("---")
    st.markdown("### Ready to translate")
    st.markdown(
        '<div class="note"><b>No API key needed</b><br>This classroom demo uses the unofficial <code>googletrans</code> Python library and requires an internet connection.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(" ")
    st.caption("Language choices are provided by the googletrans package.")

st.markdown("#### Choose languages")
left, swap_col, right = st.columns([5, 1, 5], vertical_alignment="bottom")
with left:
    source_choice = st.selectbox(
        "Translate from",
        ["Auto-detect", *LANGUAGE_OPTIONS],
        key="source_language",
        help="Choose Auto-detect or select the input language.",
    )
with swap_col:
    st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    st.button(
        "⇄",
        help="Swap source and target languages",
        use_container_width=True,
        on_click=swap_languages,
    )
with right:
    target_choice = st.selectbox(
        "Translate to",
        LANGUAGE_OPTIONS,
        key="target_language",
    )

if st.session_state.pop("swap_warning", False):
    st.toast("Choose a source language first to swap languages.")

st.markdown(" ")
source_col, result_col = st.columns(2, gap="large")
with source_col:
    st.markdown("#### Your text")
    source_text = st.text_area(
        "Source text",
        key="source_text",
        height=255,
        max_chars=MAX_CHARACTERS,
        placeholder="Type or paste the text you want to translate…",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<div class='metric'>{len(source_text):,} / {MAX_CHARACTERS:,} characters</div>",
        unsafe_allow_html=True,
    )
with result_col:
    st.markdown("#### Translation")
    st.text_area(
        "Translated text",
        value=st.session_state.translated_text,
        height=255,
        placeholder="Your translation will appear here…",
        disabled=True,
        label_visibility="collapsed",
    )
    if st.session_state.translation_info:
        st.markdown(
            f"<div class='metric'>{st.session_state.translation_info}</div>",
            unsafe_allow_html=True,
        )

button_col, clear_col, spacer = st.columns([2, 1, 7])
with button_col:
    translate_clicked = st.button("Translate text  →", type="primary", use_container_width=True)
with clear_col:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.source_text = ""
    st.session_state.translated_text = ""
    st.session_state.translation_info = ""
    st.rerun()

if translate_clicked:
    if not source_text.strip():
        st.warning("Enter some text first.")
    elif len(source_text) > MAX_CHARACTERS:
        st.error(f"Please keep the text under {MAX_CHARACTERS:,} characters per translation.")
    elif source_choice != "Auto-detect" and NAME_TO_CODE[source_choice] == NAME_TO_CODE[target_choice]:
        st.warning("Choose different source and target languages.")
    else:
        source_code = "auto" if source_choice == "Auto-detect" else NAME_TO_CODE[source_choice]
        target_code = NAME_TO_CODE[target_choice]
        try:
            with st.spinner("Translating…"):
                translated = asyncio.run(get_translation(source_text, source_code, target_code))
            st.session_state.translated_text = translated.text
            detected_code = getattr(translated, "src", source_code)
            detected_name = LANGUAGE_NAMES.get(str(detected_code).lower(), str(detected_code))
            st.session_state.translation_info = (
                f"Detected source: {detected_name} · Target: {target_choice}"
                if source_choice == "Auto-detect"
                else f"Source: {source_choice} · Target: {target_choice}"
            )
            st.rerun()
        except Exception as exc:
            st.error(
                "Translation failed. Check your internet connection and try again. "
                "The unofficial Google Translate web interface may also temporarily block requests."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

if st.session_state.translated_text:
    st.download_button(
        "Download translation as .txt",
        data=st.session_state.translated_text,
        file_name="translation.txt",
        mime="text/plain; charset=utf-8",
    )

st.markdown("---")
st.caption("Classroom/demo use · Powered by the unofficial googletrans package · Internet connection required")
