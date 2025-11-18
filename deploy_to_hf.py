import os
from huggingface_hub import upload_folder

REPO_ID = "asmaegr50/Heart_Disease_Classification"
TOKEN = os.environ.get("HF")

if TOKEN is None:
    raise RuntimeError(
        "Missing HF token in environment variable 'HF'. "
        "Make sure the GitHub secret HF is defined and passed in cd.yml."
    )

def upload(path: str, dest: str, message: str) -> None:
    print(f"📤 Uploading '{path}' -> '{dest}' on {REPO_ID}")
    upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=path,
        path_in_repo=dest,
        token=TOKEN,
        commit_message=message,
    )

def main():
    # App Gradio
    upload("App", ".", "Sync App files")

    # Modèle
    upload("Model", "Model", "Sync Model")

    # Résultats / métriques
    upload("Results", "Metrics", "Sync Metrics")

if __name__ == "__main__":
    main()
