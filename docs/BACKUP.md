# Backup & restore — hn_rewrite.db

Moved from AGENTS.md 2026-09-14 (runbook, not working rules). Unclosed code fence fixed in transit.


The HN database is backed up daily to Google Drive via a systemd user timer.

- Script: `scripts/backup_hn_db.sh`
- Service: `~/.config/systemd/user/hn-rewrite-backup.service`
- Timer: `~/.config/systemd/user/hn-rewrite-backup.timer` (active)
- Target: `drive:hn-rewrite/backups/<YYYYMMDDTHHMMSSZ>/hn_rewrite.db`
- Retention: 30 most recent snapshots (env: `HN_KEEP_N=30`)
- Logs: `journalctl --user -u hn-rewrite-backup.service`

### Manual backup

```bash
./scripts/backup_hn_db.sh                          # default config
HN_DB_PATH=/path/to/other.db ./scripts/backup_hn_db.sh
HN_KEEP_N=7 ./scripts/backup_hn_db.sh             # keep 7
```

### Restore

```bash
LATEST=$(rclone lsf --dirs-only drive:hn-rewrite/backups/ | sort -r | head -1)
rclone copy drive:hn-rewrite/backups/$LATEST/hn_rewrite.db ./hn_rewrite.db
sqlite3 hn_rewrite.db "PRAGMA integrity_check;"
```
