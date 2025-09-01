# onedrive_reader.py
import os
import requests
from pathlib import Path

import streamlit as st

CLIENT_ID = st.secrets["MS_CLIENT_ID"]
CLIENT_SECRET = st.secrets["MS_CLIENT_SECRET"]
TENANT_ID = st.secrets["MS_TENANT_ID"]
FOLDER_NAME = st.secrets["ONEDRIVE_FOLDER_NAME"]
USER_EMAIL = st.secrets["USER_EMAIL"]

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
DOWNLOAD_DIR = "downloaded_files"
SCOPES = ["https://graph.microsoft.com/.default"]

def get_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": " ".join(SCOPES),
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_drive_id(token: str) -> str:
    url = f"{GRAPH_BASE}/users/{USER_EMAIL}/drive"
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
            with open(file_path, "wb") as f:
                f.write(r.content)

def sync_onedrive_folder():
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    token = get_token()
    drive_id = get_drive_id(token)
    root_folder_id = get_folder_id(token, drive_id, FOLDER_NAME)
    download_files_from_folder(token, drive_id, root_folder_id)
