# Manual browser testing guide for `hn-rewrite`

**Server:** `http://127.0.0.1:8766` (systemd: `hn_rewrite.service`).
**Browser:** any modern browser. Chrome DevTools open (F12 → Network + Console tabs) for first-pass observation.

## 0. First-time setup (one time per browser profile)

1. Open Chrome (or any browser). Open DevTools (F12).
2. Visit `http://127.0.0.1:8766/`.
3. You'll be redirected to `/u/<8-hex-chars>` and a `hn_token` cookie is set. URL bar returns to `/`.
4. **Expected on first hit:** a skeleton page (3-line CSS-only stub with a `meta http-equiv="refresh" content="3"` tag). After ~3 s the browser auto-reloads and the full dashboard appears. From this point onward you're a known user.

> **Use a separate browser profile (or Incognito) per "user" you want to test.** Each cookie jar = one user. Profile 1 = upvoter, Profile 2 = downvoter, Profile 3 = no-feedback baseline. (In Chrome: top-right avatar → Add → Continue without account → Name "u1". Repeat for u2, u3.)

## 1. Dashboard load (cold path)

What to check:
- Skeleton appears within ~200 ms (no blocking server work).
- `meta http-equiv="refresh" content="3"` is in the HTML.
- After 3 s, browser auto-reloads. Real dashboard is now present.
- **No `meta http-equiv` in the reloaded HTML** (you should be cache-hit from then on).
- Story cards have buttons: ▲ (upvote), ▼ (downvote), ✕ (clear), and a "TLDR" / "↗" link.
- Top of page has a sort toggle (default / popular / explore / date). Clicking each should re-render client-side.
- Right side has a queue-status panel and mode selector.

DevTools:
- Network tab: `/` request returns 200, ~2-50 KB. `/api/tldr-detail` only fires when you click "TLDR" (or pre-warmed cards).
- Console: should be silent. Warnings about fetch failures or `top_comments` are normal for fresh stories.

## 2. The feedback loop (the heart of the app)

1. Pick a story you find interesting → click **▲** (upvote).
2. The card should immediately change state (e.g. button gets `data-voted="up"`, color change). No full page reload.
3. Open DevTools → Network → POST `/api/feedback`. Response: `{"ok": true, "ranking_refresh_queued": true}`.
4. Click **▼** on the same story → state changes to downvoted. The card may not visibly reorder immediately.
5. Click **✕** on it → vote cleared.
6. Upvote 5–10 different stories. Then refresh the page (F5).
7. The previously upvoted stories should be ranked higher than before.

What to check:
- `POST /api/feedback` returns `ranking_refresh_queued: true` for new votes, `false` for re-votes on the same `(user, story)` with the same `action`.
- After ~1 s of voting, the next dashboard request is a stale-while-revalidate hit (no skeleton, just the old dashboard). Within a few seconds the warm finishes and the new ranking shows up.
- Vote-clear (`action: "clear"`) should remove the card's `data-voted` attribute and free that story to be ranked neutrally.

## 3. Multiple users (multi-profile test)

| Profile | Action | What you should observe |
|---|---|---|
| **u1 (Incognito 1)** | Upvote 3 stories | Those 3 float to the top in u1's view |
| **u2 (Incognito 2)** | Downvote the same 3 stories | Those 3 sink in u2's view |
| **u3 (no votes)** | Just browse | See neutral ranking (similar to baseline popularity) |

Critical: each profile is fully isolated. Upvoting in u1 should have **zero** effect on u2's ranking of the same story.

## 4. TLDR detail

1. Click the "TLDR" or expand button on any story.
2. A panel/modal opens with the TLDR (article summary + top discussion points).
3. First click on a story: ~2-5 s (Mistral LLM call). Spinner visible.
4. Click the same TLDR again: should be **instant** (cached).
5. Different story: ~2-5 s again.

DevTools:
- Network: `POST /api/tldr-detail` with `{"story_id": N}`. First call: `~3-5 s`. Second call (same story): `~50 ms` with `"cached": true` in the JSON response.

## 5. SWR (stale-while-revalidate) — visible behavior

1. Load dashboard once. Wait for full render.
2. Open DevTools → Network → throttle to **Slow 3G**.
3. Click ▲ on a story.
4. **Expected:** the button updates instantly (no waiting). Next page load returns the old dashboard (not skeleton). Within ~5-10 s under Slow 3G, the warm completes and ranking updates.
5. If you instead see a skeleton, the SWR is broken (this was a real bug — see WORKLOG 2026-06-27).

## 6. Caching and limits

- Click ▲ on 20 different stories, then F5. First load: cached render. No skeleton.
- Vote on 50+ stories, then F5 repeatedly. The 20-most-recently-rendered dashboards stay cached. Older ones may produce a skeleton (acceptable; the next warm fills them).

## 7. Regen loop (background, no user action)

Every ~3 h the regen thread refreshes the candidate pool. Visible signals:
- `journalctl --user -u hn_rewrite.service` shows `candidate_fetch` lines.
- Your dashboard auto-refreshes to pick up new stories; you may notice new cards appear.

You don't need to test this in a normal session — it's a long-running background job.

## 8. Failure-mode checklist

| Symptom | Likely cause | What to look for |
|---|---|---|
| Skeleton never goes away | Warm thread errored | `journalctl --user -u hn_rewrite.service -e` |
| Card shows but TLDR spins forever | Mistral API issue | `journalctl` for "Mistral"/"tldr" lines |
| Buttons click but nothing changes | JS console error | DevTools → Console |
| Vote submitted, but ranking doesn't update on F5 | `expected_version` mismatch | `journalctl` for `result=stale_hit` vs `result=cache_hit` |
| 500 on dashboard | Pipeline exception | `journalctl` for stack trace |
| Skeleton on every F5 | Cache not warming | Check `_wait_for_cache`-style polling in tests for hints |

## 9. Useful DevTools tricks

- **Application → Cookies** → see `hn_token` (8 hex chars). Delete it, F5 → you become a new user.
- **Network → Disable cache** is **off**. If you toggle it, the server still caches, but the browser re-requests the dashboard HTML each time → you'll see skeleton occasionally.
- **Console** can run `fetch('/api/user').then(r => r.json()).then(console.log)` to confirm session.
