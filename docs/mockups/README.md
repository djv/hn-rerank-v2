# Bloomberg-inspired HN reader concepts

`bloomberg-reader.html` is a standalone, interactive HTML/CSS/JavaScript mockup.
It uses illustrative stories, summaries, conversations, and activity counts.
No packages, remote assets, production API calls, or database access are needed.

From the repository root, serve the mockup with:

```sh
uv run python -m http.server 8766 --bind 127.0.0.1 --directory docs/mockups
```

Then open <http://127.0.0.1:8766/bloomberg-reader.html>. Opening the HTML file
directly also works; browser storage may be unavailable in that mode.

## Ideas to review

- **Commands:** click the search box or press Ctrl/Command+K. Search for
  `Popular week`, `discussion`, or `sync`; use arrows and Enter to choose.
- **Saved views:** Daily, Weekend, and Deep dive apply different sort/window/layout
  combinations. Save up to three named custom views; these persist locally.
- **Discussion changes:** What changed shows a sample return-visit briefing.
  Open a story's discussion and mark its new comments as read to clear its delta.
- **Linked research:** Article, Discussion, and Related share one selected story.
  Follow related reading, then use Back to your story to restore the earlier tab
  and reading position.
- **Desktop scan:** Scan keeps the story list beside the reader; Focus expands
  the selected story. The scan layout stacks on smaller screens.

Demo votes advance the queue and support Undo. Votes and reviewed-comment state
are session-local; only custom views and the light/dark choice use localStorage.
Reset demo restores the sample queue, unread changes, and built-in views.
The `?` button lists keyboard controls and identifies the illustrative content.

The earlier review preview used port 8766 in the transient user service
`hn-reader-mockup.service`; this does not establish that it is currently running.
If active, stop it after review with
`systemctl --user stop hn-reader-mockup.service`. It is not enabled at boot.
`preview.jpg` shows the desktop concept at a 1440×900 CSS viewport.

The design follows the current TUI's charcoal, ivory, restrained orange,
aligned counts, and keyboard controls. Bloomberg references:

- [Navigation and autocomplete guide](https://data.bloomberglp.com/professional/sites/10/Getting-Started-Guide-for-Students-English.pdf)
- [Launchpad linked workspaces](https://bloomberg.com/company/stories/innovating-a-modern-icon-how-bloomberg-keeps-the-terminal-cutting-edge?trk=article-ssr-frontend-pulse_little-text-block)
- [News activity, velocity, and briefings](https://professional.bloomberg.com/products/bloomberg-terminal/news/)
