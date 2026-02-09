"""Configurable cloud storage upload (Cloudflare R2 / Aliyun OSS)."""

import logging
from pathlib import Path

from config import STORAGE_BACKEND

log = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def upload_session(
    session_id: str, files: list[Path]
) -> dict[str, str]:
    """Upload session files to the configured cloud storage backend.

    Args:
        session_id: Unique session identifier used as the object key prefix.
        files: Local file paths to upload.

    Returns:
        Mapping of ``{filename: cdn_url}`` for every successfully uploaded
        file.  Returns empty dict when storage is disabled.
    """
    if STORAGE_BACKEND == "r2":
        return _upload_to_r2(session_id, files)
    if STORAGE_BACKEND == "oss":
        return _upload_to_oss(session_id, files)

    log.warning(
        "STORAGE_BACKEND is empty or unrecognised (%r), skipping upload.",
        STORAGE_BACKEND,
    )
    return {}


def _upload_to_r2(
    session_id: str, files: list[Path]
) -> dict[str, str]:
    """Upload files to Cloudflare R2 via boto3."""
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

    if not R2_ENDPOINT:
        log.warning("R2 not configured (SPLAZMATTE_R2_ENDPOINT is empty).")
        return {}

    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )

    cdn_urls: dict[str, str] = {}
    for filepath in files:
        if not filepath.exists():
            log.warning("Skipping non-existent file: %s", filepath)
            continue

        key = f"{R2_PREFIX}/{session_id}/{filepath.name}"
        content_type = CONTENT_TYPES.get(
            filepath.suffix.lower(), "application/octet-stream"
        )

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


def _upload_to_oss(
    session_id: str, files: list[Path]
) -> dict[str, str]:
    """Upload files to Aliyun OSS via oss2."""
    import oss2

    from config import (
        OSS_ACCESS_KEY_ID,
        OSS_ACCESS_KEY_SECRET,
        OSS_BUCKET,
        OSS_CDN_DOMAIN,
        OSS_ENDPOINT,
        OSS_PREFIX,
    )

    if not OSS_ENDPOINT:
        log.warning("OSS not configured (SPLAZMATTE_OSS_ENDPOINT is empty).")
        return {}

    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

    cdn_urls: dict[str, str] = {}
    for filepath in files:
        if not filepath.exists():
            log.warning("Skipping non-existent file: %s", filepath)
            continue

        key = f"{OSS_PREFIX}/{session_id}/{filepath.name}"
        content_type = CONTENT_TYPES.get(
            filepath.suffix.lower(), "application/octet-stream"
        )

        log.info("Uploading %s → oss://%s/%s", filepath.name, OSS_BUCKET, key)
        try:
            headers = {"Content-Type": content_type}
            bucket.put_object_from_file(key, str(filepath), headers=headers)
            if OSS_CDN_DOMAIN:
                cdn_urls[filepath.name] = f"https://{OSS_CDN_DOMAIN}/{key}"
            else:
                cdn_urls[filepath.name] = (
                    f"{OSS_ENDPOINT}/{key}"
                )
        except Exception:
            log.exception("Failed to upload %s", filepath.name)

    return cdn_urls
