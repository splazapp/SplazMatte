"""Upload session files to Cloudflare R2."""

import logging
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from config import (
    R2_ACCESS_KEY,
    R2_BUCKET,
    R2_CDN_DOMAIN,
    R2_ENDPOINT,
    R2_PREFIX,
    R2_SECRET_KEY,
)

log = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _build_client():
    """Create a boto3 S3 client configured for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )


def upload_session_to_r2(
    session_id: str, files: list[Path]
) -> dict[str, str]:
    """Upload a list of files to R2 under ``{R2_PREFIX}/{session_id}/``.

    Args:
        session_id: Unique session identifier used as the R2 sub-prefix.
        files: Local file paths to upload.

    Returns:
        Mapping of ``{filename: cdn_url}`` for every successfully uploaded
        file.  Returns empty dict when R2 is not configured.
    """
    if not R2_ENDPOINT:
        log.warning("R2 not configured (SPLAZMATTE_R2_ENDPOINT is empty), skipping upload.")
        return {}

    client = _build_client()
    cdn_urls: dict[str, str] = {}

    for filepath in files:
        if not filepath.exists():
            log.warning("Skipping non-existent file: %s", filepath)
            continue

        key = f"{R2_PREFIX}/{session_id}/{filepath.name}"
        content_type = CONTENT_TYPES.get(filepath.suffix.lower(), "application/octet-stream")

        log.info("Uploading %s → s3://%s/%s", filepath.name, R2_BUCKET, key)
        try:
            client.upload_file(
                str(filepath),
                R2_BUCKET,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            cdn_urls[filepath.name] = f"https://{R2_CDN_DOMAIN}/{key}"
        except Exception:
            log.exception("Failed to upload %s", filepath.name)

    return cdn_urls
