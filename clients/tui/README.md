# HN Rerank terminal reader

Run `uvx hn-rerank` (Python 3.12+). Choose an existing profile link or create a
profile with your server URL, for example `https://your-host/hn/`.
`--server URL` selects a deployment; credentials from another server are never reused.
The server retains its existing access policy: installing the client does not grant access.

Keys: j/k or arrows navigate/scroll; Tab changes focus; Enter reads; Escape returns;
1/2/3 vote positive/neutral/negative; u undoes the latest successful vote;
o/c open article/comments; r refreshes; ? shows help; q quits.
Use the sort and age selectors to change filters. Below 100 columns only one pane shows.

Configuration lives in platformdirs' user config directory (`hn-rerank/profile.json`),
outside uv's cache. The profile token is a credential: keep that file and profile links private.
Requests do not follow redirects or retry ambiguous votes. Refresh after a connection failure;
undo can clear the latest confirmed vote. Reading requires an online server; no local ML.

Local build: `uv build --project clients/tui`.
Local launch: `uvx --from /absolute/path/to/hn_rerank-0.1.0-py3-none-any.whl hn-rerank`.
If the package name cannot be registered, publish as `hn-rerank-tui` and run
`uvx --from hn-rerank-tui hn-rerank`.
