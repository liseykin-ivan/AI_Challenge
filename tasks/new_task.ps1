# Создаёт файл задания из текста в буфере обмена.
# Использование (из корня проекта):
#   powershell -ExecutionPolicy Bypass -File tasks\new_task.ps1            # следующий по счёту день
#   powershell -ExecutionPolicy Bypass -File tasks\new_task.ps1 -Week 2 -Day 3
param(
    [int]$Week = 0,
    [int]$Day = 0,
    [int]$DaysPerWeek = 5
)

$dir = $PSScriptRoot

# Номер следующего дня: после последнего существующего файла
if ($Week -eq 0 -or $Day -eq 0) {
    $last = Get-ChildItem -Path $dir -Filter 'week_*_day_*.md' |
        ForEach-Object {
            if ($_.Name -match '^week_(\d+)_day_(\d+)\.md$') {
                [pscustomobject]@{ W = [int]$Matches[1]; D = [int]$Matches[2] }
            }
        } |
        Sort-Object W, D | Select-Object -Last 1

    if (-not $last)                      { $Week = 1;           $Day = 1 }
    elseif ($last.D -lt $DaysPerWeek)    { $Week = $last.W;     $Day = $last.D + 1 }
    else                                 { $Week = $last.W + 1; $Day = 1 }
}

$path = Join-Path $dir ("week_{0}_day_{1}.md" -f $Week, $Day)
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

$content = "# Неделя $Week · День $Day — $title`r`n`r`n## Постановка задания`r`n`r`n$text`r`n"
[IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Создан: $path" -ForegroundColor Green
