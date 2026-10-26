"""Alert delivery must show actionable text instead of the default label dump."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

OBS = Path(__file__).resolve().parents[1] / "observability"


def test_alert_annotations():
    rules = yaml.safe_load((OBS / "alerts.yml").read_text(encoding="utf-8"))
    for group in rules["groups"]:
        for rule in group["rules"]:
            for key in ("title", "summary", "impact", "description"):
                assert rule["annotations"].get(key), (rule["alert"], key)
                # Regression: pages must stay English and contain no emoji.
                assert rule["annotations"][key].isascii()
            # The description goes to Telegram unescaped (safeHtml), so no HTML specials in it.
            assert not set("<>&") & set(rule["annotations"]["description"]), rule["alert"]


def test_telegram_template_wiring():
    template = (OBS / "telegram.tmpl").read_text(encoding="utf-8")
    assert '{{ define "telegram.message"' in template
    assert template.isascii()
    for field in ("summary:", "impact:", "action:", "started:", "duration:", "links:"):
        assert field in template
    assert "[FIRING]" in template and "[RESOLVED]" in template
    assert re.findall(r"\.Labels\.(\w+)", template) == ["alertname", "severity"]
    assert ".Alerts.Firing" in template and ".Alerts.Resolved" in template
    # Regression: with parse_mode HTML Alertmanager renders with html/template, which escapes
    # values itself and rejects an explicit `html` call ("predefined escaper disallowed"):
    # every Telegram notification failed. Only description is trusted HTML (for <code>).
    assert "html ." not in template
    for key in ("title", "summary", "impact"):
        assert f"{{{{ .Annotations.{key} }}}}" in template
    assert "| safeHtml }}" in template
    config = yaml.safe_load((OBS / "alertmanager.yml").read_text(encoding="utf-8"))
    assert config["templates"] == ["/tmp/telegram.tmpl"]
    compose = yaml.safe_load((OBS / "docker-compose.yml").read_text(encoding="utf-8"))
    am = compose["services"]["alertmanager"]
    assert "./telegram.tmpl:/etc/alertmanager/telegram.tmpl:ro" in am["volumes"]
    assert am["environment"]["GRAFANA_PORT"] == "${GRAFANA_PORT:-3000}"
    assert (
        "--web.external-url=http://localhost:${PROMETHEUS_PORT:-9090}" in compose["services"]["prometheus"]["command"]
    )


@pytest.mark.skipif(os.name == "nt", reason="The container entrypoint uses a POSIX shell")
@pytest.mark.parametrize("enabled", [False, True])
def test_entrypoint_rendered_config(tmp_path, enabled):
    # Regression: the receiver used Alertmanager's default labels and broken Source URL.
    # Redirect every container path; only a synthetic token fixture is ever inspected.
    source = (OBS / "alertmanager-entrypoint.sh").read_text(encoding="utf-8")
    token = tmp_path / "synthetic-token"
    if enabled:
        token.write_text("synthetic-test-token", encoding="utf-8")
    # /tmp/ first: on Linux tmp_path itself is under /tmp/, and replacing it last mangled the
    # token path inserted before it (CI on GitHub failed, the Mac's tmp is /var/folders).
    source = source.replace("/tmp/", tmp_path.as_posix() + "/")
    source = source.replace("/etc/alertmanager/secrets/telegram_bot_token", token.as_posix())
    source = source.replace("/etc/alertmanager/", OBS.as_posix() + "/")
    source = source.replace("exec /bin/alertmanager", "printf '%s\\n'")
    script = tmp_path / "entrypoint.sh"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        ["sh", str(script)],
        env={**os.environ, "GRAFANA_PORT": "3300", "TELEGRAM_CHAT_ID": "123"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    config = yaml.safe_load((tmp_path / "alertmanager.yml").read_text(encoding="utf-8"))
    receiver = config["receivers"][0]
    if enabled:
        telegram = receiver["telegram_configs"][0]
        assert telegram["message"] == '{{ template "telegram.message" . }}'
        assert telegram["bot_token_file"] == token.as_posix()
        assert telegram["parse_mode"] == "HTML"
        assert telegram["send_resolved"] is True
    else:
        assert "telegram_configs" not in receiver
    assert config["templates"] == ["/tmp/telegram.tmpl"]
    template = (tmp_path / "telegram.tmpl").read_text(encoding="utf-8")
    assert "http://localhost:3300/d/devcopilot" in template
    assert "__GRAFANA_PORT__" not in template
    assert "--web.external-url=http://localhost:9093" in result.stdout
