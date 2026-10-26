# Песочница

Агент читает недоверенные данные (задачи, коммиты, логи CI) и исполняет чужой код
(`run_tests` запускает pytest над репозиторием). Поэтому всё, что исполняет решения
модели, работает в изоляции с отдельным ядром.

| Что | Где | Почему |
|---|---|---|
| CLI, граф агентов, память | песочница | принимает решения по недоверенным данным |
| MCP-серверы repo, tracker, ci | песочница (подпроцессы CLI) | читают и пишут workspace |
| workspace: git, tracker.db, CI | песочница | данные, которые агент может испортить |
| `run_tests` | песочница | исполняет код |
| Ollama | хост, GPU | нужны GPU и только HTTP; файлы не нужны |
| ollaya (`winnow:e4b`, модель решений для gate) | отдельный хост, на GPU-хост — через обратный SSH-туннель | только HTTP; при недоступности gate решает LLM |
| Observability | Docker Desktop на хосте | телеметрия переживает песочницу |

Исходящая сеть из песочницы: только модель (`:11434`), OTLP (`:4318`) и модель решений
для gate (`:11435`, туннель к ollaya).

## Основной вариант: Docker Sandboxes (`sbx`), microVM

Установка на Windows (PowerShell от администратора, затем обычный пользователь):

```powershell
# 0. Виртуализация включена (SVM/VT-x в BIOS): должно быть True
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled
# 1. Windows Hypervisor Platform и перезагрузка
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All
Restart-Computer
# 2. CLI и вход
winget install -h Docker.sbx
sbx login
# 3. Сеть: модель, OTLP и модель решений на хосте; остальное закрыто
sbx policy allow network localhost:11434
sbx policy allow network localhost:4318
sbx policy allow network localhost:11435
# 4. Песочница с каталогом проекта (путь на диске Windows, не \\wsl.localhost)
cd C:\src\devcopilot
sbx run --name devcopilot shell
```

Установка зависимостей требует временно разрешить `pypi.org`, `files.pythonhosted.org` и сайт uv;
после установки правила снимаются. Внутри песочницы проект запускается так же, как на хосте:

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venv
export DEVCOPILOT_OLLAMA_BASE_URL=http://host.docker.internal:11434
export DEVCOPILOT_OTEL_ENDPOINT=http://host.docker.internal:4318
export DEVCOPILOT_JEV_BASE_URL=http://host.docker.internal:11435
uv sync --frozen --no-dev && uv run devcopilot seed
uv run devcopilot ask --team "Почему падает CI на main?"
```

Скрипты: `setup.sh` — установка внутри песочницы (с временными правилами сети),
`probe.sh` — проверка изоляции изнутри, `evals.sh` — агентные evals команды.
Проверка, что изоляция настоящая:

```bash
sbx exec devcopilot bash /c/src/devcopilot/sandbox/probe.sh
```

## Запасной вариант: gVisor (runsc) в отдельном дистрибутиве WSL2

Образ системы — [Dockerfile](Dockerfile): python-slim, git, uv, непривилегированный
пользователь, код только для чтения, запись только в `/data`.

```bash
docker build -f sandbox/Dockerfile -t devcopilot .
docker run --rm -it --runtime=runsc \
  -v devcopilot-data:/data \
  --add-host host.docker.internal:host-gateway \
  -e DEVCOPILOT_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  devcopilot seed
```

Сеть здесь закрывается средствами Docker. Это слабее, чем политика `sbx`: ядро WSL2
общее для всех дистрибутивов, а gVisor защищает только от системных вызовов.
