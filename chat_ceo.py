import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

import file_parser
import embed_and_store
from answer_with_rag import answer  # adaptive + keyword + fallback
from onedrive_reader import sync_onedrive_folder

# Must be the first Streamlit call
st.set_page_config(page_title="AI CEO Assistant", page_icon="🧠", layout="wide")

# ────────────────────────────────────────
# Login System
# ────────────────────────────────────────
USERNAME = "admin123"
PASSWORD = "BestOrg123@#"

def login():
    st.title("Login to AI CEO Assistant")
    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username_input == USERNAME and password_input == PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ────────────────────────────────────────
# Constants & Setup
# ────────────────────────────────────────
HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")
UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)
EMBED_INDEX = Path("embeddings/faiss.index")
EMBED_META = Path("embeddings/metadata.pkl")

# ────────────────────────────────────────
# Helpers
# ────────────────────────────────────────
def load_history():
    return json.loads(HIST_PATH.read_text(encoding="utf-8")) if HIST_PATH.exists() else []

def save_history(history):
    HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def reset_chat():
    if HIST_PATH.exists():
        HIST_PATH.unlink()

def save_refresh_time():
    REFRESH_PATH.write_text(datetime.now().strftime("%b-%d-%Y %I:%M %p"))

def load_refresh_time():
    return REFRESH_PATH.read_text() if REFRESH_PATH.exists() else "Never"

def export_history_to_csv(history: list) -> bytes:
    return pd.DataFrame(history).to_csv(index=False).encode("utf-8")

def embeddings_exist() -> bool:
    return EMBED_INDEX.exists() and EMBED_META.exists()

def _secrets_diag():
    KEYS = [
        "OPENAI_API_KEY",
        "MS_CLIENT_ID",
        "MS_CLIENT_SECRET",
        "MS_TENANT_ID",
        "ONEDRIVE_FOLDER_NAME",
        "USER_EMAIL",
    ]
    st.subheader("Secrets Diagnostic")
    try:
        all_keys = []
        try:
            all_keys = list(st.secrets.keys())  # type: ignore[attr-defined]
        except Exception:
            pass
        st.write("Visible keys at runtime:", all_keys if all_keys else "(none)")
        for k in KEYS:
            try:
                v = st.secrets.get(k)  # type: ignore[attr-defined]
            except Exception:
                v = None
            ok = isinstance(v, str) and len(v.strip()) > 0
            shown = (v[:4] + "…" + v[-4:]) if isinstance(v, str) and len(v) >= 8 else ("(set)" if ok else "(missing)")
            st.write(f"- {k}: {'OK' if ok else 'MISSING'} → {shown}")
    except Exception as e:
        st.error(f"Secrets diagnostic failed: {e}")

# ────────────────────────────────────────
# Sidebar Navigation
# ────────────────────────────────────────
st.sidebar.title("AI CEO Panel")
st.sidebar.markdown(f"Logged in as: `{USERNAME}`")
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

# strict vs fallback toggle (default OFF => always answer)
strict_mode = st.sidebar.checkbox("Strict answers only (no fallback)", value=False)

mode = st.sidebar.radio("Navigation", ["💬 New Chat", "📜 View History", "🔁 Refresh Data"])

# ────────────────────────────────────────
# Mode: Refresh Data (Single Button Flow)
# ────────────────────────────────────────
if mode == "🔁 Refresh Data":
    st.title("Refresh AI Knowledge Base")
    st.caption("Sync files, parse documents, and rebuild embeddings.")
    st.markdown(f"**Last Refreshed:** {load_refresh_time()}")

    with st.expander("Secrets diagnostic (click to expand)"):
        _secrets_diag()

    run_all = st.button("Run Full Refresh (Sync → Parse → Embed)")

    if run_all:
        try:
            with st.spinner("Step 1/3: Syncing OneDrive → downloaded_files/..."):
                sync_onedrive_folder()
            st.success("Step 1/3 complete: OneDrive sync finished.")

            with st.spinner("Step 2/3: Parsing downloaded_files → parsed_data..."):
                file_parser.main()
            st.success("Step 2/3 complete: Parsing finished.")

            with st.spinner("Step 3/3: Building embeddings from parsed_data → embeddings/..."):
                embed_and_store.main()
            st.success("Step 3/3 complete: Embeddings built.")

            # Clear cached FAISS handle so the next QA uses the new files
            st.cache_resource.clear()

            save_refresh_time()
            st.info(f"Full refresh completed. **Last Refreshed:** {load_refresh_time()}")

        except Exception as e:
            st.error(f"Full refresh failed: {e}")

    if embeddings_exist():
        st.info("Embeddings detected.")
    else:
        st.warning("Embeddings not found. Click the button above to run the full refresh.")

# ────────────────────────────────────────
# Mode: View History
# ────────────────────────────────────────
elif mode == "📜 View History":
    st.title("Chat History")
    history = load_history()

    if not history:
        st.info("No chat history found.")
    else:
        for turn in history:
            role = "You" if turn.get("role") == "user" else "Assistant"
            timestamp = turn.get("timestamp", "N/A")
            with st.expander(f"{role} | [{timestamp}]"):
                st.markdown(turn.get("content", ""))

        st.markdown("---")
        st.download_button(
            label="Download Chat History as CSV",
            data=export_history_to_csv(history),
            file_name="chat_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Chat History"):
            reset_chat()
            st.success("History cleared.")

# ────────────────────────────────────────
# Mode: New Chat
# ────────────────────────────────────────
elif mode == "💬 New Chat":
    st.title("AI CEO Assistant")
    st.caption("Ask about meetings, projects, hiring, finances, and research. Answers cite your documents.")
    st.markdown(f"**Last Refreshed:** {load_refresh_time()}")

    if not embeddings_exist():
        st.warning("Embeddings not found. Go to 'Refresh Data' and run the Full Refresh.")
    history = load_history()

    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(f"**[{turn.get('timestamp', 'N/A')}]**  \n{turn.get('content', '')}")

    user_msg = st.chat_input("Type your question…")
    if user_msg:
        now = datetime.now().strftime("%b-%d-%Y %I:%M%p")
        history.append({"role": "user", "content": user_msg, "timestamp": now})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = answer(user_msg, k=7, chat_history=history, strict_mode=strict_mode)
                except Exception as e:
                    reply = f"Error: {e}"
            st.markdown(f"**[{datetime.now().strftime('%b-%d-%Y %I:%M%p')}]**  \n{reply}")

        history.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%b-%d-%Y %I:%M%p")})
        save_history(history)
