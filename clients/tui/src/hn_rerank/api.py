from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from platformdirs import user_config_path

from .models import Feed


class APIError(Exception):
    """Safe, credential-free error for display."""


class InvalidProfile(APIError):
    pass


def normalize_server(value: str) -> str:
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
    path = parsed.path.rstrip("/") + "/"
    if any(part in {".", ".."} for part in unquote(path).split("/")):
        raise ValueError("Invalid server path.")
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), path, "", ""))


@dataclass(frozen=True)
class Profile:
    server: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "server", normalize_server(self.server))
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


def save_profile(profile: Profile, path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".profile-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"server": profile.server, "token": profile.token}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
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
        except httpx.RequestError as exc:
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
        except ValueError as exc:
            raise APIError(
                "Invalid feed response. Check the server API version."
            ) from exc

    async def summary(self, story_id: int) -> str:
        response = await self.request(
            "POST", "api/tldr-detail", json={"story_id": story_id}
        )
        try:
            value = response.json()["tldr"]
            if not isinstance(value, str):
                raise TypeError("Invalid summary")
            return value
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError(
                "Summary unavailable. Select another story or refresh."
            ) from exc

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
