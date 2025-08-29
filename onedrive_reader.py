import os
import requests
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

import streamlit as st

client_id = st.secrets["MS_CLIENT_ID"]
client_secret = st.secrets["MS_CLIENT_SECRET"]
tenant_id = st.secrets["MS_TENANT_ID"]
folder_name = st.secrets["ONEDRIVE_FOLDER_NAME"]
user_email = st.secrets["USER_EMAIL"]


GRAPH_BASE = "https://graph.microsoft.com/v1.0"
DOWNLOAD_DIR = "downloaded_files"
SCOPES = ["https://graph.microsoft.com/.default"]


def get_token():
    print("🔑 Authenticating with Microsoft Graph...")
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


def get_drive_id(token):
    print("📦 Getting drive ID...")
    url = f"{GRAPH_BASE}/users/{USER_EMAIL}/drive"
    headers = {"Authorization": f"Bearer {token}"}
    drive_resp = requests.get(url, headers=headers)
    drive_resp.raise_for_status()
    return drive_resp.json()["id"]


def get_folder_id(token, drive_id, folder_name):
    print(f"📂 Locating folder: {folder_name}")
    url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{folder_name}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["id"]
    else:
        raise Exception(f"❌ Folder '{folder_name}' not found. Details: {response.text}")


def download_files_from_folder(token, drive_id, folder_id, folder_path=""):
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
            print(f"📁 Entering folder: {rel_path}")
            download_files_from_folder(token, drive_id, item["id"], folder_path=rel_path)
        else:
            file_path = os.path.join(DOWNLOAD_DIR, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            download_url = item["@microsoft.graph.downloadUrl"]
            print(f"📄 Downloading: {rel_path}")
            r = requests.get(download_url)
            with open(file_path, "wb") as f:
                f.write(r.content)


if __name__ == "__main__":
    token = get_token()
    drive_id = get_drive_id(token)
    root_folder_id = get_folder_id(token, drive_id, FOLDER_NAME)
    print(f"📥 Downloading files to: {DOWNLOAD_DIR}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_files_from_folder(token, drive_id, root_folder_id)
    print("✅ Done.")
