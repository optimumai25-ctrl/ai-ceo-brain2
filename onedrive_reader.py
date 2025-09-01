# onedrive_reader.py
import os
import requests
from pathlib import Path
import streamlit as st

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
DOWNLOAD_DIR = "downloaded_files"
SCOPES = ["https://graph.microsoft.com/.default"]

def _get_secret(name: str, default: str = "") -> str:
    try:
        return (st.secrets.get(name) if hasattr(st, "secrets") else None) or os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)

def _secrets():
    client_id = _get_secret("MS_CLIENT_ID")
    client_secret = _get_secret("MS_CLIENT_SECRET")
    tenant_id = _get_secret("MS_TENANT_ID")
    folder_name = _get_secret("ONEDRIVE_FOLDER_NAME")
    user_email = _get_secret("USER_EMAIL")
    missing = [k for k, v in {
        "MS_CLIENT_ID": client_id,
        "MS_CLIENT_SECRET": client_secret,
        "MS_TENANT_ID": tenant_id,
        "ONEDRIVE_FOLDER_NAME": folder_name,
        "USER_EMAIL": user_email,
    }.items() if not v]
    if missing:
        raise KeyError(f"Missing secrets: {', '.join(missing)}")
    return client_id, client_secret, tenant_id, folder_name, user_email

def get_token():
    client_id, client_secret, tenant_id, *_ = _secrets()
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": " ".join(SCOPES),
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_drive_id(token: str) -> str:
    *_, user_email = _secrets()
    url = f"{GRAPH_BASE}/users/{user_email}/drive"
    headers = {"Authorization": f"Bearer {token}"}
    drive_resp = requests.get(url, headers=headers)
    drive_resp.raise_for_status()
    return drive_resp.json()["id"]

def get_folder_id(token: str, drive_id: str, folder_name: str) -> str:
    url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{folder_name}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["id"]

def download_files_from_folder(token: str, drive_id: str, folder_id: str, folder_path: str = ""):
    url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/children"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    items = response.json().get("value", [])
    for item in items:
        name = item["name"]
        is_folder = "folder" in item
        rel_path = os.path.join(folder_path, name)

        if is_folder:
            download_files_from_folder(token, drive_id, item["id"], folder_path=rel_path)
        else:
            file_path = os.path.join(DOWNLOAD_DIR, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            download_url = item["@microsoft.graph.downloadUrl"]
            r = requests.get(download_url)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)

def sync_onedrive_folder():
    client_id, client_secret, tenant_id, folder_name, user_email = _secrets()  # validate early
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    token = get_token()
    drive_id = get_drive_id(token)
    root_folder_id = get_folder_id(token, drive_id, folder_name)
    download_files_from_folder(token, drive_id, root_folder_id)
