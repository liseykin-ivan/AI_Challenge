# Создаёт файл задания tasks\day_<N>.md из текста в буфере обмена.
# N — сквозной номер дня челленджа, без номера недели.
# Использование (из корня проекта):
#   powershell -ExecutionPolicy Bypass -File tasks\new_task.ps1            # следующий по счёту день
#   powershell -ExecutionPolicy Bypass -File tasks\new_task.ps1 -Day 9
param(
    [int]$Day = 0
)

$dir = $PSScriptRoot

# Номер следующего дня: после самого большого номера среди day_<N>.md
if ($Day -eq 0) {
    $last = Get-ChildItem -Path $dir -Filter 'day_*.md' |
        ForEach-Object { if ($_.Name -match '^day_(\d+)\.md$') { [int]$Matches[1] } } |
        Sort-Object | Select-Object -Last 1
    $Day = if ($last) { $last + 1 } else { 1 }
}

$path = Join-Path $dir ("day_{0}.md" -f $Day)
if (Test-Path $path) {
    Write-Host "Файл уже существует: $path" -ForegroundColor Red
    exit 1
}

$text = Get-Clipboard -Raw
if ([string]::IsNullOrWhiteSpace($text)) {
    Write-Host "Буфер обмена пуст — скопируйте текст задания." -ForegroundColor Red
    exit 1
}
$text = $text.Trim()

# Заголовок: первая непустая строка без эмодзи и префикса "День N."
$first = ($text -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -First 1)
$title = ($first -replace '^[^\p{L}\p{N}]+', '') -replace '^День\s*\d+\s*[.:]?\s*', ''

$content = "# День $Day — $title`r`n`r`n## Постановка задания`r`n`r`n$text`r`n"
[IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Создан: $path" -ForegroundColor Green
