#!/bin/bash
# Silver labels from Muse (OpenCode), 4 chunks of 60 stories; output silver-muse-K.txt
set -u
cd /tmp/tagging
split -l 60 -d -a 1 silver-cards.jsonl chunk-
for k in 0 1 2 3; do
  [ -s silver-muse-$k.txt ] && continue
  prompt="Label Hacker News stories with 12 yes/no tags. Judge each tag independently from the title, domain, source and excerpt; use general knowledge of the domain where needed (e.g. whether a site is usually paywalled).

Tags (name: question, YES and NO descriptions):
$(cat tag-criteria.txt)

Stories, one JSON object per line:
$(cat chunk-$k)

Reply with ONLY JSON lines, one per story in the same order, no other text, exactly this shape:
{\"id\": <id>, \"show_hn\": 0|1, \"ask_hn\": 0|1, \"research\": 0|1, \"tutorial\": 0|1, \"opinion\": 0|1, \"news_event\": 0|1, \"business\": 0|1, \"politics_policy\": 0|1, \"ai_ml\": 0|1, \"product_pr\": 0|1, \"sensational\": 0|1, \"paywalled\": 0|1}"
  timeout 900 ~/.opencode/bin/opencode run -m opencode-go/muse-spark-1.3-contributor "$prompt" > silver-muse-$k.txt 2> silver-muse-$k.err
  echo "chunk $k exit $? lines $(grep -c '"id"' silver-muse-$k.txt)"
done
echo "muse labels done"
