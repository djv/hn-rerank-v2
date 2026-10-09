The agent recommends **three P2 fixes before deployment**, plus a P3 wording correction. The prior review’s blanket “four P2 blockers” conclusion overstates the profile-state issue and does not distinguish the preexisting vote race.

Assessment covered `4e946e5` → `5129813`, excluding AGENTS-only `7706816`. Source/diff and relevant tests were assessed before reading the prior findings. Evidence is **source-level tracing, not executed reproduction**. The supplied passing gates were accepted without rerunning; no database, provider, network or live-service access occurred. Git remained clean.

1. **Confirmed — malformed strict hydration; P2, mandatory.**
   [pipeline/enrichment.py:157](/home/d/code/hn-rerank/pipeline/enrichment.py:157) validates only truthiness and `type == "story"`. Sequence: stored article body → forced refresh → HTTP 200 `{"type":"story"}` → defaults replace title/URL/score → preserved article body makes composed text nonempty → upsert and returned `Story` → generation. The server’s failure guard at [server.py:3240](/home/d/code/hn-rerank/server.py:3240) therefore accepts it.

   Counterevidence: HTTP failures, wrong types and exceptions correctly return `None`; those cases have regression coverage. DB merging preserves richer content and comment markers ([database.py:665](/home/d/code/hn-rerank/database.py:665)), but does **not** preserve title/URL/score against these defaults. Validation weakness predates the commit; forced hydration of zero-comment/article-only stories newly exposes it. Validate identity and required fields before writing or treating input as fresh.

2. **Confirmed mechanism; “permanent” overstated — orphaned forced intent; P2, mandatory.**
   Sequence: first forced task running → first stats check finishes → second `r` declares a new intent → delayed load joins that task at [app.py:1502](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1502), bypassing intent consumption at line 1514 → second stats observes growth → completed summary remains behind → deferred comparison reaches the still-current intent at [app.py:2083](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:2083) and defers again.

   Counterevidence: task completion removes `_forced_tasks`, and navigation changes the selection serial, allowing recovery. The guard is therefore persistent **at that selection**, not permanent across navigation. The repeated-`r` test at [test_sort_refresh.py:416](/home/d/code/hn-rerank/clients/tui/tests/test_sort_refresh.py:416) does not ensure the first forced task has started before the second press. Consume intent when joining, while retaining one paid request.

3. **Conditional — acknowledged vote target overwritten; P2, mandatory for this scope, preexisting race.**
   Required sequence: feed response is constructed at V/ready → its delivery is delayed → vote/undo acknowledges V+1 → [app.py:2209](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:2209) marks the current feed pending → older response arrives → [app.py:1737](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1737) passes identity/serial checks → replacement loses the target; line 1754 clears restored rows.

   Counterevidence: merely holding a GET before the server constructs its payload is insufficient. `rated` still protects optimistic votes, stored feedback is unaffected, and later polling can recover ([app.py:1625](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1625)); polling pauses while reading. Baseline already had the replacement/clear behavior, but sort refresh adds exposure. Preserve an acknowledged pending target and undo restoration until the response covers it.

4. **Confirmed — profile-scoped stats state survives reconnect; downgrade to P3, optional.**
   [app.py:1059](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1059) resets summaries and vote state but leaves stats requests/offered counts. Sequence: A declines 25 → B connects with the same story → B’s 25 observation is suppressed. A held request can also make B’s check return early at [app.py:1978](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1978).

   Counterevidence: A’s eventual reply is rejected by API identity at line 2017; it cannot apply A’s counts to B. Requests are bounded, and later checks recover. This newly introduced omission can suppress a check/prompt, but is not cross-profile reply application or automatic spending. Reset/cancel the stats state on reconnect.

5. **Confirmed — warm-flight join omits known snapshot; P3, optional.**
   [server.py:1706](/home/d/code/hn-rerank/server.py:1706) caches text with the captured snapshot, but line 1713 publishes a metadata-free reply. An ordinary joiner receives that reply; [app.py:1570](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1570) then records unknown provenance. [test_tldr_single_flight.py:458](/home/d/code/hn-rerank/tests/test_tldr_single_flight.py:458) explicitly expects this payload.

   Counterevidence: atomic DB provenance remains correct, subsequent cache reads return it, and forced callers obtain their fresh follow-up. Unknown is conservative, not fabricated coverage. Include the captured snapshot in complete successful warm replies to avoid this feature gap.

6. **Confirmed — coverage wording; P3, required copy correction.**
   [app.py:645](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:645) says “Summary covers N comments,” although [pipeline/ranking.py:518](/home/d/code/hn-rerank/pipeline/ranking.py:518) selects comments and line 599 bounds their text. Counterevidence: the stored number intentionally represents discussion size, so the **storage design is defensible**. Change the wording to “Discussion had N comments when summarized; M now.”

One additional **P3 optional copy fix**: [app.py:1433](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1433) still promises “r restores hidden stories,” while the new handler preserves hidden rows and returns immediately without a selection. Remove that obsolete instruction.