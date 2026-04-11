"""S3 bucket configuration for data transfer between GPU and CPU instances."""
import subprocess
import sys

BUCKET = "s3://electricdroplet/electricdroplet"

# S3 paths
CLUSTERS_S3 = f"{BUCKET}/clusters"
DFT_RESULTS_S3 = f"{BUCKET}/dft_results"
MODELS_S3 = f"{BUCKET}/models"
CHECKPOINTS_S3 = f"{BUCKET}/checkpoints"


def sync_up(local_dir, s3_path, delete=False):
    """Upload local directory to S3."""
    cmd = ["aws", "s3", "sync", str(local_dir), s3_path]
    if delete:
        cmd.append("--delete")
    print(f"  Uploading {local_dir} -> {s3_path}")
    subprocess.run(cmd, check=True)


def sync_down(s3_path, local_dir):
    """Download S3 path to local directory."""
    cmd = ["aws", "s3", "sync", s3_path, str(local_dir)]
    print(f"  Downloading {s3_path} -> {local_dir}")
    subprocess.run(cmd, check=True)


def upload_file(local_path, s3_path):
    """Upload a single file to S3."""
    cmd = ["aws", "s3", "cp", str(local_path), s3_path]
    print(f"  Uploading {local_path} -> {s3_path}")
    subprocess.run(cmd, check=True)


def download_file(s3_path, local_path):
    """Download a single file from S3."""
    cmd = ["aws", "s3", "cp", s3_path, str(local_path)]
    print(f"  Downloading {s3_path} -> {local_path}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    print(f"Bucket: {BUCKET}")
    subprocess.run(["aws", "s3", "ls", BUCKET + "/"], check=True)
    print("S3 access OK.")
