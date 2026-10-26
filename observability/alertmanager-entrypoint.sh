#!/bin/sh
# Builds the Alertmanager config: Telegram delivery only if both secrets are present.
# The token stays in a file mounted read-only (bot_token_file): it never enters the
# config, the environment, the logs or the repository.
set -eu
TOKEN=/etc/alertmanager/secrets/telegram_bot_token
OUT=/tmp/alertmanager.yml
# Only a numeric port may enter sed and the HTML link.
GRAFANA_PORT=${GRAFANA_PORT:-3000}
case "$GRAFANA_PORT" in
  *[!0-9]*|'') echo "alertmanager: invalid GRAFANA_PORT" >&2; exit 1 ;;
esac
sed "s/__GRAFANA_PORT__/$GRAFANA_PORT/g" /etc/alertmanager/telegram.tmpl > /tmp/telegram.tmpl
if [ -s "$TOKEN" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  sed "s|    # TELEGRAM_RECEIVER|    telegram_configs:\n      - bot_token_file: $TOKEN\n        chat_id: $TELEGRAM_CHAT_ID\n        send_resolved: true\n        parse_mode: HTML\n        message: '{{ template \"telegram.message\" . }}'|" \
    /etc/alertmanager/alertmanager.yml > "$OUT"
  echo "alertmanager: Telegram delivery on"
else
  cp /etc/alertmanager/alertmanager.yml "$OUT"
  echo "alertmanager: no Telegram secrets, alerts only in the UI"
fi
# Without external-url the Source link points at an unreachable container id.
exec /bin/alertmanager --config.file="$OUT" --storage.path=/alertmanager --web.external-url=http://localhost:9093
