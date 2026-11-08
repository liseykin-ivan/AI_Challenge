# AI Challenge — CLI-чат с LLM

День 1: консольное приложение отправляет запрос в OpenCode Go (OpenAI-совместимый API)
через голый `httpx` и печатает ответ, модель, токены и время.

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
$h = @{ Authorization = "Bearer $key"; "User-Agent" = "ai-challenge-cli/0.1" }

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
| `llm_client.py` | `LLMClient.chat(messages) -> ChatResult`, HTTP и разбор ошибок |
| `config.py` | чтение и проверка `.env` |
