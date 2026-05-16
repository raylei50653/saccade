from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_RTSP_HOST = "127.0.0.1"
DEFAULT_RTSP_PORT = 8554
DEFAULT_RTSP_READ_USER = "reader"
DEFAULT_RTSP_READ_PASSWORD = "readpass123"
DEFAULT_RTSP_PUBLISH_USER = "publisher"
DEFAULT_RTSP_PUBLISH_PASSWORD = "pubpass123"
DEFAULT_RTSP_STREAM_PREFIX = "stream_"
DEFAULT_RTSP_SINGLE_STREAM_PATH = "live"
DEFAULT_RTSP_OUTPUT_PATH = "detected"


def normalize_rtsp_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if not normalized:
        raise ValueError("RTSP path must not be empty")
    return normalized


def build_stream_path(
    stream_id: int | str, prefix: str = DEFAULT_RTSP_STREAM_PREFIX
) -> str:
    return f"{prefix}{stream_id}"


def build_rtsp_url(
    path: str,
    *,
    host: str = DEFAULT_RTSP_HOST,
    port: int = DEFAULT_RTSP_PORT,
    username: str | None = None,
    password: str | None = None,
) -> str:
    normalized_path = normalize_rtsp_path(path)
    authority = host
    if port != DEFAULT_RTSP_PORT:
        authority = f"{authority}:{port}"
    elif ":" not in host:
        authority = f"{authority}:{port}"

    if username:
        credentials = username
        if password is not None:
            credentials = f"{credentials}:{password}"
        authority = f"{credentials}@{authority}"

    return f"rtsp://{authority}/{normalized_path}"


def build_reader_url(
    path: str,
    *,
    host: str = DEFAULT_RTSP_HOST,
    port: int = DEFAULT_RTSP_PORT,
    username: str = DEFAULT_RTSP_READ_USER,
    password: str = DEFAULT_RTSP_READ_PASSWORD,
) -> str:
    return build_rtsp_url(
        path,
        host=host,
        port=port,
        username=username,
        password=password,
    )


def build_publisher_url(
    path: str,
    *,
    host: str = DEFAULT_RTSP_HOST,
    port: int = DEFAULT_RTSP_PORT,
    username: str = DEFAULT_RTSP_PUBLISH_USER,
    password: str = DEFAULT_RTSP_PUBLISH_PASSWORD,
) -> str:
    return build_rtsp_url(
        path,
        host=host,
        port=port,
        username=username,
        password=password,
    )


@dataclass(frozen=True)
class RTSPEndpoint:
    host: str
    port: int
    path: str
    username: str | None = None
    password: str | None = None

    def url(self) -> str:
        return build_rtsp_url(
            self.path,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )

    @classmethod
    def from_url(cls, url: str) -> "RTSPEndpoint":
        parsed = urlparse(url)
        if parsed.scheme != "rtsp":
            raise ValueError(f"Unsupported RTSP URL scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise ValueError(f"RTSP URL missing host: {url}")
        return cls(
            host=parsed.hostname,
            port=parsed.port or DEFAULT_RTSP_PORT,
            path=normalize_rtsp_path(parsed.path),
            username=parsed.username,
            password=parsed.password,
        )
