# AI Challenge — CLI-чат с LLM

День 1 (`main.py`): консольное приложение отправляет запрос в OpenCode Go (OpenAI-совместимый API)
через голый `httpx` и печатает ответ, модель, токены и время.
День 2 (`day2_format.py`): сравнение способов управлять форматом ответа — см. [ниже](#день-2-формат-ответа).

## Установка (PowerShell, Python 3.11+)

```powershell
cd C:\Users\user\Desktop\AI_Projects\AI_Challenge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env   # впишите OPENCODE_API_KEY
```

Если `Activate.ps1` запрещён политикой: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Запуск

```powershell
python main.py "Привет, кто ты?"
python main.py                              # спросит запрос в консоли
python main.py --model <id> "Привет"        # другая модель (id — из списка ниже)
```

Пример вывода:

```
Привет! Я — ИИ-ассистент...

deepseek-v4.1-flash | токены: 31/24 | 1.42 с
```

При ошибке печатается понятное сообщение, код выхода — 1.

## Проверка доступа без кода

```powershell
$key = ((Get-Content .env) -match '^OPENCODE_API_KEY=')[0].Split('=', 2)[1]
$h = @{ Authorization = "Bearer $key"; "User-Agent" = "ai-challenge-cli/0.1"
        "x-opencode-session" = [guid]::NewGuid().ToString("N") }   # без него OpenCode Go отвечает 400

# список моделей
(Invoke-RestMethod https://opencode.ai/inference/go/v1/models -Headers $h).data.id

# тестовый запрос
$body = '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"ping"}]}'
(Invoke-RestMethod https://opencode.ai/inference/go/openai/v1/chat/completions `
  -Method Post -Headers $h -ContentType "application/json" -Body $body).choices[0].message.content
```

## Структура

| Файл | Что делает |
|---|---|
| `main.py` | CLI: аргументы, ввод, вывод |
| `llm_client.py` | `LLMClient.chat(messages, **params) -> ChatResult` (текст, `finish_reason`, рассуждение, usage), HTTP и разбор ошибок |
| `config.py` | чтение и проверка `.env` |
| `day2_format.py` | День 2: один запрос с разным уровнем контроля формата |
| `results/` | сырые ответы и отчёты прогонов дня 2, выводы — `day2_conclusions.md` |
| `tasks/` | тексты заданий челленджа |

## День 2: формат ответа

`day2_format.py` отправляет один и тот же запрос с разным уровнем контроля
и проверяет, может ли код распарсить ответ.

```powershell
python day2_format.py                        # все 7 вариантов × 3 прогона
python day2_format.py --runs 5 --max-tokens 40
python day2_format.py --variants format,schema --runs 2
python day2_format.py --timeout 120 --temperature 0.7 --model <id>
```

| Параметр | По умолчанию | Что делает |
|---|---|---|
| `--runs` | 3 | прогонов на каждый вариант |
| `--variants` | все | список через запятую из таблицы ниже |
| `--max-tokens` | 60 | лимит для `max_tokens` и `schema_max_tokens` |
| `--timeout` | `LLM_TIMEOUT` из `.env` | таймаут одного запроса, с |
| `--temperature` | как у модели | температура для всех вариантов |
| `--model` | `LLM_MODEL` из `.env` | id модели |

| Вариант | Что меняется в запросе |
|---|---|
| `plain` | только вопрос, без системного промта |
| `format` | формат JSON описан словами в системном промте |
| `max_tokens` | `format` + `max_tokens` (по умолчанию 60) |
| `stop` | `format` + просьба закончить маркером `<END>` + `stop=["<END>"]` |
| `json_mode` | `format` + `response_format={"type": "json_object"}` |
| `schema` | `format` + `response_format` со строгой JSON Schema (`strict: true`) |
| `schema_max_tokens` | строгая схема + `max_tokens` |

Каждый ответ проверяется тремя способами: `json.loads` как есть, JSON после чистки
(срезать ```-обёртку и текст вокруг `{...}`) и соответствие схеме (поля, типы, ровно 3 языка).
Для каждого прогона печатаются токены (сколько из них ушло на скрытое рассуждение),
время и `finish_reason`, а для сломанных ответов — причина: например, «лимит max_tokens
съело рассуждение» или «обрезан по лимиту». В конце — итоговая таблица по вариантам.
В `results/` пишутся сырые ответы вместе с рассуждением (`.json`) и отчёт (`.md`),
где для пустого ответа показан хвост рассуждения.
Если провайдер не поддерживает `response_format` или запрос упал по таймауту,
вариант покажет ошибку API, а не уронит скрипт.

Выводы по прогонам на `deepseek-v4.1-flash` — в [`results/day2_conclusions.md`](results/day2_conclusions.md).
