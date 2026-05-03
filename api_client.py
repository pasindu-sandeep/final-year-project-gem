import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/authenticate"


def run_ai_request(uploaded_files):
    """
    For manual Streamlit file uploads.
    """
    try:
        files_payload = []

        for f in uploaded_files:
            file_name = getattr(f, "name", "image.jpg")
            file_type = getattr(f, "type", "image/jpeg") or "image/jpeg"
            file_bytes = f.getvalue()

            files_payload.append(
                ("files", (file_name, file_bytes, file_type))
            )

        response = requests.post(API_URL, files=files_payload, timeout=120)

        if response.status_code != 200:
            return False, f"API Error {response.status_code}: {response.text}"

        return True, response.json()

    except Exception as e:
        return False, str(e)


def run_ai_request_from_paths(image_paths):
    """
    For auto captured images saved on disk.
    """
    opened_files = []

    try:
        files_payload = []

        for path in image_paths:
            path = Path(path)
            f = open(path, "rb")
            opened_files.append(f)

            files_payload.append(
                ("files", (path.name, f, "image/jpeg"))
            )

        response = requests.post(API_URL, files=files_payload, timeout=120)

        if response.status_code != 200:
            return False, f"API Error {response.status_code}: {response.text}"

        return True, response.json()

    except Exception as e:
        return False, str(e)

    finally:
        for f in opened_files:
            try:
                f.close()
            except Exception:
                pass