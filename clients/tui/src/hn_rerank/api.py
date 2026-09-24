from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from platformdirs import user_config_path

from .models import Feed


@dataclass(frozen=True)
class Impression:
    event_id: str
    client_session_id: str
    story_id: int
    dashboard_version: int
    position: int
    sort_mode: str
    age_filter: str
    occurred_at: float
    event_type: str = "impression"
    source_filter: str = "mixed"
    ranker_arm: str = "tui_observed"


class APIError(Exception):
    """Safe, credential-free error for display."""


class InvalidProfile(APIError):
    pass


@dataclass(frozen=True)
class Summary:
    """Summary text plus whether the server marked it provisional.

    Provisional responses (a stale fallback or a retryable partial) are shown
    but must not be cached: a later attempt may produce a complete summary.
    Empty responses mean hydration found nothing summarizable; clients skip
    them instead of rendering the placeholder sentence.
    """

    text: str
    provisional: bool = False
    empty: bool = False


def normalize_server(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Invalid server URL.")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Enter an http(s) server URL without credentials, query or fragment."
        )
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Use HTTPS for a remote server to protect your profile.")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ValueError("Enter an http(s) server URL with a valid port.") from exc
    path = parsed.path.rstrip("/") + "/"
    if any(part in {".", ".."} for part in unquote(path).split("/")):
        raise ValueError("Invalid server path.")
    normalized = urlunsplit((parsed.scheme, parsed.netloc.lower(), path, "", ""))
    # The validator must accept exactly what the HTTP layer can request.
    # httpx raises InvalidURL (not RequestError) for unprintable characters
    # or hosts that cannot be IDNA-encoded; without this check such a URL
    # would escape every API caller as an uncaught exception.
    try:
        httpx.URL(normalized)
    except httpx.InvalidURL as exc:
        raise ValueError("Enter a server URL the client can request.") from exc
    return normalized


@dataclass(frozen=True)
class Profile:
    server: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "server", normalize_server(self.server))
        if not isinstance(self.token, str):
            raise TypeError("Invalid profile token.")
        if (
            not self.token
            or not self.token.isascii()
            or not all(c.isalnum() or c in "_-" for c in self.token)
        ):
            raise ValueError("Invalid profile token.")

    @classmethod
    def from_link(cls, link: str) -> Profile:
        parsed = urlsplit(link.strip())
        prefix, marker, token = parsed.path.rstrip("/").rpartition("/u/")
        if not marker or parsed.query or parsed.fragment:
            raise ValueError("Use the complete profile link ending in /u/TOKEN.")
        return cls(
            urlunsplit((parsed.scheme, parsed.netloc, prefix + "/", "", "")), token
        )


def profile_path() -> Path:
    return user_config_path("hn-rerank", appauthor=False) / "profile.json"


def load_profile(path: Path) -> Profile | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Profile(data["server"], data["token"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InvalidProfile(
            "Saved profile is unreadable. Import your profile again."
        ) from exc


def restrict_permissions(path: Path, *, directory: bool = False) -> None:
    if os.name != "nt":
        path.chmod(0o700 if directory else 0o600)
        return
    # chmod on Windows does not set a private DACL. Replace it with an
    # owner-only ACL through the OS-provided PowerShell, without shell interpolation.
    script = """
$ErrorActionPreference = 'Stop'
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($env:HN_RERANK_DIRECTORY -eq '1') {
    $acl = New-Object System.Security.AccessControl.DirectorySecurity
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
} else {
    $acl = New-Object System.Security.AccessControl.FileSecurity
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($sid, 'FullControl', 'Allow')
}
$acl.SetOwner($sid)
$acl.SetAccessRuleProtection($true, $false)
$acl.AddAccessRule($rule)
Set-Acl -LiteralPath $env:HN_RERANK_CONFIG_PATH -AclObject $acl
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "HN_RERANK_CONFIG_PATH": str(path),
            "HN_RERANK_DIRECTORY": "1" if directory else "0",
        },
    )
    if result.returncode:
        raise OSError("Could not restrict profile permissions.")


def save_profile(profile: Profile, path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    restrict_permissions(path.parent, directory=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".profile-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"server": profile.server, "token": profile.token}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        restrict_permissions(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class API:
    def __init__(
        self,
        server: str,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.server = normalize_server(server)
        self._token = token
        self.client = httpx.AsyncClient(
            timeout=45, follow_redirects=False, transport=transport
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def request(
        self, method: str, path: str, json: dict[str, object] | None = None
    ) -> httpx.Response:
        # All paths are internal constants; reject absolute/escaping inputs anyway.
        if path.startswith("/") or ":" in path or ".." in path:
            raise ValueError("Invalid API path")
        headers = {"Cookie": f"hn_token={self._token}"} if self._token else {}
        try:
            response = await self.client.request(
                method, self.server + path, headers=headers, json=json
            )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise APIError(
                "Connection failed. Press r to refresh; votes are not retried."
            ) from exc
        if response.status_code in {401, 403}:
            raise InvalidProfile(
                "Profile rejected or server access denied. Import a valid profile."
            )
        if response.status_code == 429:
            delay = response.headers.get("Retry-After", "a few")
            delay = delay if delay.isdigit() else "a few"
            raise APIError(f"Rate limited. Try again in {delay} seconds.")
        if response.is_redirect:
            raise APIError(
                "Server redirected the request. Check the server URL and deployment prefix."
            )
        if response.is_error:
            raise APIError(
                f"Server returned {response.status_code}. Press r to retry reading."
            )
        return response

    async def validate(self) -> Profile:
        response = await self.request("GET", "api/user")
        try:
            data = response.json()
            profile = Profile(self.server, data["token"])
            if self._token is not None and profile.token != self._token:
                raise ValueError("Profile mismatch")
            self._token = profile.token
            return profile
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidProfile("Server returned an invalid profile.") from exc

    async def create(self) -> Profile:
        await self.request("GET", "")
        return await self.validate()

    async def feed(self) -> Feed:
        response = await self.request("GET", "api/feed")
        try:
            return Feed.parse(response.json())
        except (TypeError, ValueError) as exc:
            raise APIError(
                "Invalid feed response. Check the server API version."
            ) from exc

    async def cached_summary(self, story_id: int) -> Summary | None:
        """Read existing summaries only; old servers fail without generation."""
        response = await self.request("GET", f"api/tldr-cache/{story_id}")
        if response.status_code == 204:
            return None
        try:
            data = response.json()
            value = data["tldr"]
            if not isinstance(value, str):
                raise TypeError("Invalid summary")
            return Summary(
                value,
                provisional=data.get("stale") is True or data.get("retryable") is True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError("Invalid cached summary.") from exc

    async def summary(self, story_id: int, *, force_refresh: bool = False) -> Summary:
        payload: dict[str, object] = {"story_id": story_id}
        if force_refresh:
            payload["force_refresh"] = True
        response = await self.request("POST", "api/tldr-detail", json=payload)
        try:
            data = response.json()
            value = data["tldr"]
            if not isinstance(value, str):
                raise TypeError("Invalid summary")
            return Summary(
                value,
                provisional=data.get("stale") is True or data.get("retryable") is True,
                empty=data.get("empty") is True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError(
                "Summary unavailable. Select another story or refresh."
            ) from exc

    async def impression(self, event: Impression) -> None:
        await self.request("POST", "api/interaction", json={"events": [asdict(event)]})

    async def vote(self, story_id: int, action: str) -> int:
        response = await self.request(
            "POST", "api/feedback", json={"story_id": story_id, "action": action}
        )
        try:
            data = response.json()
            target = data["target_version"]
            if data.get("ok") is not True or type(target) is not int or target < 0:
                raise ValueError("Invalid vote acknowledgement")
            return target
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError(
                "Vote result is uncertain; it will not be retried automatically."
            ) from exc

    async def ready(self, target: int) -> tuple[bool, int]:
        response = await self.request(
            "GET", f"api/ranking-ready?min_version={target}&target_version={target}"
        )
        try:
            data = response.json()
            current = data["current_version"]
            if (
                type(current) is not int
                or current < 0
                or type(data["ready"]) is not bool
            ):
                raise ValueError("Invalid readiness")
            return data["ready"], current
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError("Invalid ranking status. Press r to refresh.") from exc
