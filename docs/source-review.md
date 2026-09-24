# Source review — 2026-09-23

Local and VPS config.toml SHA256 match. Review uses configured subscriptions
and a read-only 30-day story inventory, not personal vote attribution or a
feed availability test. Stored volume is not exposure or preference evidence.
No subscriptions were changed.

## Keep

Strong complementary core: Simon Willison, Interconnects, Import AI, Eugene
Yan, Chip Huyen; Dan Luu, Julia Evans, Cloudflare, LWN, Jane Street, Tweag,
Well-Typed; Construction Physics, Bits about Money, Pedestrian Observations,
Human Transit, Quanta, Project Zero and Trail of Bits. Low posting frequency
alone is not a reason to drop a thoughtful author.

## Change before adding more

- Improve content availability for existing high-value sources. Of stored
  30-day stories, 0/24 Hugging Face, 16/123 Lobsters, 3/27 OCaml.org,
  2/11 DeepMind, 3/22 Aeon and 0/4 Asterisk have >=300 characters of self
  text/article body or any comments. This is a coarse content-presence check,
  not proof feeds are broken or a recommendation to remove these sources.
- Reddit is heavily represented. Prefer reducing repetitive/high-volume
  categories before adding more subreddits: AI_Agents/ChatGPTCoding,
  CreditCards/awardtravel, fatFIRE/financialindependence/Bogleheads.
  Keep distinct purposes (e.g. ExpatFIRE's cross-border angle).
- Keep both Works in Progress feeds until overlap is measured; magazine
  and newsletter can carry distinct material.

## First removal/pause candidates (editorial judgement, not vote-derived)

- Slashdot: 256 stored stories in 30 days; likely lowest marginal novelty
  alongside HN, Lobsters and programming Reddit.
- GitHub Trending weekly/all: overlaps the language-specific feeds and is
  a discovery list rather than sustained analysis. Keep Python/Haskell if
  they are useful. All three share a source label in current ingestion.
- r/AI_Agents: trial pause to reduce product pitches/agent chatter; retain
  LocalLLaMA and ChatGPTCoding for now. Recent content quality not sampled.
- r/CreditCards and r/fatFIRE only if US card optimization and high-net-worth
  lifestyle discussions are not active interests; these are conditional cuts.

## Add candidates

- Martin Kleppmann: distributed systems and local-first software; adds depth
  missing from the AI-heavy mix.
- Rachel by the Bay: production debugging and systems operations.
- Stripe engineering: concrete infrastructure/payments implementation, a
  complement to Bits about Money rather than another general tech aggregator.

Suggested first trial: pause Slashdot and GitHub Trending/all, add Kleppmann
and Rachel by the Bay. Candidate feed URLs/availability have not been verified.
