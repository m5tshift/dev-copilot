# Dev Copilot

Мультиагентный ассистент инженерной команды: по вопросу на естественном языке расследует падения CI, находит причину в коде и связывает ее с задачами трекера, а изменения (задачи, комментарии, ветки с исправлением) вносит только после одобрения человека. Все модели работают локально, внешние API не используются.

Стек: Python 3.12, LangGraph, MCP, Ollama (`qwen3.5:9b` по умолчанию), uv. Observability: OpenTelemetry, Phoenix, Prometheus, Grafana, Loki, Alertmanager.

## Требования

- Python 3.12 и [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com) с моделями: `ollama pull qwen3.5:9b qwen3.5:4b gemma4:e4b qwen3-embedding:0.6b`.
  Замеры в проекте сделаны на GPU с 12 ГБ VRAM; модели загружаются по очереди.
- Необязательно: [ollaya](https://github.com/ollaya-dev/ollaya) с моделью решений `winnow:e4b` для gate
  (без нее gate решает LLM), Docker для стека observability.
- Тесты не требуют ни LLM, ни сети: `uv run pytest -q`.

## Быстрый старт

```bash
uv sync                      # зависимости
uv run devcopilot seed       # создать «мир»: git-репозиторий, трекер, CI
uv run devcopilot ask "Почему падает CI на main и связано ли это с жалобами в трекере?"
uv run devcopilot ask "Исправь баг: создай ветку с фиксом и проверь её тестами"
uv run devcopilot ask --team "Почему падает CI на main?"   # supervisor + специалисты
```

Любое изменение (задачи, комментарии, ветки) агент делает только после подтверждения: CLI покажет вызов и спросит `[a]pprove / [r]eject / [e]dit`.

Другие команды:

```bash
uv run devcopilot tools      # инструменты MCP-серверов, их аннотации и политика подтверждений
uv run devcopilot tools --schemas   # плюс JSON-схемы в том виде, в каком их получает модель
uv run devcopilot graph [--team]   # граф агента или команды в формате Mermaid
uv run devcopilot bench      # одни и те же вопросы через одиночного агента и команду (этап 2)
uv run pytest                # тесты (LLM не нужна)
DEVCOPILOT_REASONING=true uv run devcopilot ask "..."   # режим thinking
```

## Роутинг моделей, валидатор, память

```bash
# каждой роли своя модель; по умолчанию gate=qwen3.5:4b (резерв модели решений),
# validator=gemma4:e4b — другого семейства, проверяет финальный ответ
uv run devcopilot ask --team "Почему падает CI на main?"
# опция: LLM-проверка каждого отчёта специалиста (детерминированная есть всегда), +~90 с на отчёт
DEVCOPILOT_VALIDATE_REPORTS=true uv run devcopilot ask --team "Почему падает CI на main?"

# три слоя памяти: SQLite-checkpoint'ы, эпизоды с эмбеддингами, темпоральные факты
DEVCOPILOT_MEMORY=true uv run devcopilot ask "Кто занимается задачей #3?"
DEVCOPILOT_MEMORY=true uv run devcopilot ask --thread <id> "А что с #2?"   # продолжение диалога
uv run devcopilot memory facts "#3" --history status    # версии факта во времени
uv run devcopilot memory recall "почему красный CI"     # что агент вспомнит
```

Промпты собираются из `persona/*.md` (IDENTITY, SOUL, USER, AGENTS), навыки лежат в
`skills/*/SKILL.md` и подгружаются инструментом `load_skill`. Песочница: [sandbox/README.md](sandbox/README.md).

## Evals

```bash
uv run devcopilot eval tasks                                  # датасет: 45 задач, dev и holdout
uv run devcopilot eval run --split dev --repeat 3 --tag base  # прогон -> results/evals/*.jsonl
uv run devcopilot eval judge results/evals/<file>.jsonl       # вердикты LLM-судьи по рубрике
uv run devcopilot eval report results/evals/<file>.jsonl      # pass@1 с интервалом, pass@k, pass^k, проверки
uv run devcopilot eval show results/evals/<file>.jsonl --task ci_root_cause   # ответы и вердикты по задаче
uv run devcopilot eval compare <base>.jsonl <new>.jsonl       # регрессионный гейт (код 1 при FAIL)
uv run devcopilot eval label results/evals/<file>.jsonl --annotator <имя>     # ручная разметка
uv run devcopilot eval calibrate results/evals/<file>.jsonl --annotator <имя> # согласие судьи с разметкой
uv run devcopilot eval components --repeat 3                  # gate и supervisor отдельно
uv run devcopilot eval online --sample 10                     # проверка реальных трейсов из Phoenix
```

Конфигурации задаются как `режим/модель[+think][+jev=компоненты][+<роль>=<модель>][+validate]`: `--config single/qwen3.5:4b --config team/qwen3.5:9b+jev=gate,review,stop`. Конфигурация evals явная: `team/qwen3.5:9b` — без модели решений и без маршрутов из `config.py`, продукт — `team/qwen3.5:9b+jev=gate+gate=qwen3.5:4b+validator=gemma4:e4b`.

## Модель решений на gate (Jev-like)

Gate решает один бит — просит ли пользователь что-то изменить, — поэтому его решает не LLM, а модель решений: она не генерирует текст, а отвечает вероятностью «да». Клиент говорит по протоколу `/v1/systemone` с локальными серверами моделей решений — [ollaya](https://github.com/ollaya-dev/ollaya) и laya; облачные сервисы не используются, всё работает локально. **По умолчанию gate решает `winnow:e4b` из ollaya на `localhost:11435` (порог 0,5)**; если сервер недоступен или отвечает мусором, решает LLM маршрута gate (по умолчанию `qwen3.5:4b`) с тем же входом и срабатывает алерт `DecisionModelUnavailable`.

```bash
ollaya pull winnow:e4b && ollaya serve             # локальный сервер моделей решений
uv run devcopilot ask --team "Закрой задачу #4"    # gate: модель решений
DEVCOPILOT_JEV_FOR= uv run devcopilot ask --team "..."   # без модели решений: gate на LLM
uv run devcopilot eval components --jev-gate --gate-only   # gate: модель решений против LLM
uv run devcopilot eval judge <file>.jsonl --model jev      # как судья (не оправдано: κ 0,40)
```

Замеры: gate на `winnow:e4b` — 45 из 45 задач, 0,92 с через SSH-туннель; как судья она проигрывает LLM. Код: `src/devcopilot/decide.py`.

## Модель на другом компьютере

`DEVCOPILOT_OLLAMA_BASE_URL` указывает на любую Ollama, например на ПК с GPU через SSH-туннель:

```bash
ssh -N -L 11444:127.0.0.1:11434 user@<IP ПК>      # 127.0.0.1, а не localhost: на Windows это IPv6
export DEVCOPILOT_OLLAMA_BASE_URL=http://localhost:11444
```

Прогоны evals на другом хосте в отчётах идут отдельной строкой: их время несравнимо.

## Observability

```bash
docker compose -f observability/docker-compose.yml up -d
export DEVCOPILOT_OTEL_ENDPOINT=http://localhost:4318
uv run devcopilot ask "..."
```

Трейсы и траектории — Phoenix (http://localhost:6006), дашборд и логи — Grafana (http://localhost:3000), алерты — Alertmanager (http://localhost:9093). Без `DEVCOPILOT_OTEL_ENDPOINT` телеметрия выключена.

Настройки задаются через переменные окружения или файл `.env` (пример в `.env.example`, все поля в `src/devcopilot/config.py`).

## «Мир», с которым работает агент

`devcopilot seed` создаёт `workspace/` с тремя источниками данных:

| Источник | Что внутри |
|---|---|
| `repo/` | git-репозиторий сервиса `shortlink` (сокращатель ссылок), 4 коммита от alice, bob и carol |
| `tracker.db` | SQLite-трекер: 6 задач с комментариями (в #6 спрятана prompt injection) |
| `ci/` | 5 прогонов CI с логами |

В коммите `329311c` спрятан настоящий баг. Он ломает CI, и на него же жалуются пользователи в задаче #3, хотя сама жалоба об этом не говорит. Хеши коммитов детерминированы: они одинаковы после каждого `seed`.

Исходные данные мира лежат в `src/devcopilot/world/data/`, генератор — в `src/devcopilot/world/seed.py`.

## Структура

```
src/devcopilot/
  config.py          настройки
  llm.py             фабрика модели и роутинг по ролям (единственное место, которое знает про Ollama)
  persona.py         персона из persona/*.md и skills из skills/*/SKILL.md
  memory.py          память: checkpoint'ы, эпизоды, темпоральные факты
  agents/single.py   агент: цикл на LangGraph, подтверждение записей, пометка недоверенных данных
  agents/team.py     команда: gate, supervisor, специалисты, синтез и ревью
  agents/prompts.py  все промпты
  grounding.py       проверка: упомянутое агентом встречалось в выводе инструментов
  runner.py          запуск вопроса, подтверждения, корневой спан и метрики прогона
  telemetry.py       OpenTelemetry: трейсы, метрики, логи (выключено без DEVCOPILOT_OTEL_ENDPOINT)
  usage.py           токены и GenAI-метрики по каждому вызову LLM
  decide.py          типизированные решения «да/нет»: модель решений или LLM за одним интерфейсом
  bench.py           бенчмарк одиночный агент против команды (этап 2)
  trajectory.py      траектория прогона: вызовы инструментов, gate, раунды supervisor'а
  evals/             датасет, проверки, судья, статистика, отчёты, разметка, онлайн-оценки
  tools/             реализация инструментов: обычные функции (repo, tracker, ci)
  servers/           MCP-серверы, которые публикуют эти функции
  mcp_client.py      запуск серверов и загрузка их инструментов
  policy.py          что требует подтверждения, какие аргументы добавляет клиент
  world/             симулированный мир и его генератор
  cli.py             CLI
observability/       docker compose: Collector, Phoenix, Prometheus, Alertmanager, Loki, Grafana
evals/               датасеты агента, базовых LLM и памяти, ручная разметка (labels/)
persona/             IDENTITY, SOUL, USER, AGENTS.md
skills/              навыки: SKILL.md с front matter
sandbox/             образ, установка, проверки и evals в песочнице (Docker Sandboxes, gVisor)
scripts/             синхронизация Mac → ПК, Windows-обвязка: интерактивная сессия, запрет сна, запуск сервисов
tests/               тесты инструментов, агента, команды, телеметрии и evals
```

