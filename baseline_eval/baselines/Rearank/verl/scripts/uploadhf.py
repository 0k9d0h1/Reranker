from huggingface_hub import HfApi
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
parser.add_argument("-hf","--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
args = parser.parse_args()

api = HfApi()
api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
api.upload_folder(folder_path=args.local_dir, repo_id=args.hf_upload_path, repo_type="model")