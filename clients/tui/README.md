# HN Rerank terminal reader

Run `uvx hn-rerank` (Python 3.12+). Choose an existing profile link or create a
profile with your server URL, for example `https://your-host/hn/`.
`--server URL` selects a deployment; credentials from another server are never reused.
The server retains its existing access policy: installing the client does not grant access.

Keys: j/k or arrows navigate/scroll; Tab changes focus; Enter reads (again to return);
Escape returns; 1/2/3 vote positive/neutral/negative; u undoes the latest successful vote;
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

## Reader appearance

The charcoal and ivory reader uses orange for the brand, active filters and
focus, with restrained blue/sage/amber accents in metadata and vote counts.
Bold key terms in a summary take the same orange, and single-marker emphasis
(italic) does too, so emphasis reads without extra chrome. A `>` marker
identifies the selected headline without relying on color. At 100 columns and
above, filter tabs sit over a one-third headline/two-thirds summary layout.
Smaller terminals use dropdowns; Enter opens reading (again to leave) and
Escape returns to headlines.
Resize and focus changes preserve reading position. Run the offline preview from
the repository root with `uv run python -m clients.tui.tests.preview --headless`.

Unselected headlines are dimmed, so the selected row (ivory on a lifted band,
plus the `>` marker) reads first. One docked footer bar carries the current
filter's counts on the left (`2 shown · +0 ~0 −0`) and context-sensitive keys on
the right; failures appear there with a `✗` prefix and replace the counts until
the next successful refresh. Scrollbars follow the theme, and stories without a
usable timestamp simply omit the age.

Empty filters and feed failures get headed notices in the reading pane rather
than a bare line; the setup form is a bordered card on a dimmed scrim; the
reading pane caps at 100 columns with a hairline under the headline; votes
confirm in the status line.
