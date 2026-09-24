# HN Rerank status

## Objective
Make TUI reading and navigation responsive: TLDR zoom, voting hints, and
aggressive prefetch across sorting, voting and j/k navigation.

## Verified result
- Zoom hides headlines at all widths; Enter/Escape restores list focus.
  Wide zoom is centered and capped at 100 columns.
- Footer says `1 up · 2 neutral · 3 down → next story`. Status sits above
  shortcuts at all widths, preserving counts and error space.
- Prefetch defaults: 20 forward cache targets, previous three and first three
  in other sorts; generate missing TLDRs for next three and navigation neighbors.
  Four background requests maximum; selected stories reuse in-flight work.
  `--prefetch-generate 0` opts out of speculative generation.
- Commit review fixed footer clipping for three-line errors in narrow zoom.
  TUI relaunched in `work:3.1` with the fix; 44 stories rendered.
- TUI tests: 121 passed, 1 skipped. Backend: 821 passed using
  `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model`.
  Ruff, touched-file formatting and ty passed.
- Relaunched in `work:3.1`: 44 stories loaded. VPS journal confirms cache
  prefetch hits and a generated TLDR returning HTTP 200 (about 9.3 seconds).
- Previous committed review fixes remain in `c8651e5` and `23ebcb5`.

## Blocker / limits
- TUI review is complete; user authorized commit and push.
  Server/VPS code unchanged.
- Prefetch can still be outrun by rapid navigation or slow/rate-limited generation.
  API errors pause new background work for 60 seconds.
- Generation now uses provider capacity ahead of selection, as authorized.

## Next step
Check remote CI after pushing `Improve TUI zoom and navigation prefetch`.
