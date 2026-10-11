<#
  build.ps1 —— 一键构建 + 日志体检

  做三件事：
    1. 用 latexmk 走完整的 xelatex -> biber -> xelatex 循环；
    2. 从 build/main.log 里数出错误、未定义引用、未定义引文、溢出盒；
    3. 把结论（而不是"我觉得没问题"）打出来。

  退出码：0 = 干净构建；1 = 有错误或未定义引用。
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Quiet
)

#  注意：不要设 $ErrorActionPreference='Stop'。latexmk / xelatex / biber
#  都会往 stderr 写正常进度信息，在 Stop 模式下会被 PowerShell 当成致命
#  错误抛出（NativeCommandError）。这里改为按退出码判断成败。
$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    $log  = Join-Path $here 'build\main.log'
    $pdf  = Join-Path $here 'build\main.pdf'

    if ($Clean) {
        Write-Host '== 清理 ==' -ForegroundColor Cyan
        & latexmk -C 2>&1 | Out-Null
        if (Test-Path (Join-Path $here 'build')) {
            Remove-Item (Join-Path $here 'build') -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host '已清理。'
        return
    }

    Write-Host '== 构建 ==' -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    #  stderr 也要收进来（latexmk 的进度与告警都走 stderr），
    #  并把 ErrorRecord 统一转成字符串，免得混进对象把输出弄乱。
    $out = & latexmk -pdfxe -silent 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $sw.Stop()

    if (-not $Quiet -and $out) { $out | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray } }

    if (-not (Test-Path $log)) {
        Write-Host '构建失败：没有生成 build/main.log' -ForegroundColor Red
        exit 1
    }

    # ---- 日志体检 ----------------------------------------------------
    $text = Get-Content $log -Raw -Encoding UTF8

    function Count-Matches([string]$pattern) {
        return ([regex]::Matches($text, $pattern)).Count
    }

    $errors     = Count-Matches '(?m)^! '
    $undefRef   = Count-Matches 'Reference `[^'']+'' on page'
    $undefCite  = Count-Matches 'Citation `[^'']+'' on page'
    $multiply   = Count-Matches 'There were multiply-defined labels'
    $overfull   = Count-Matches 'Overfull \\hbox'
    $underfull  = Count-Matches 'Underfull \\hbox'
    $missing    = Count-Matches 'LaTeX Warning: File `[^'']+'' not found'
    $fwdLabels  = Count-Matches '\[huntian-doc\] forward-label'

    # 页数
    $pages = '?'
    if ($text -match 'Output written on .*\((\d+) pages') { $pages = $Matches[1] }
    elseif ($text -match 'Output written on .*\((\d+) page') { $pages = 1 }

    Write-Host ''
    Write-Host '== 日志体检 ==' -ForegroundColor Cyan
    Write-Host ("  页数                : {0}" -f $pages)
    Write-Host ("  未成文章节（前向标签）: {0}" -f $fwdLabels)
    Write-Host ("  LaTeX 错误          : {0}" -f $errors)
    Write-Host ("  未定义引用          : {0}" -f $undefRef)
    Write-Host ("  未定义引文          : {0}" -f $undefCite)
    Write-Host ("  重复标签警告        : {0}" -f $multiply)
    Write-Host ("  缺失文件            : {0}" -f $missing)
    Write-Host ("  溢出盒 (Overfull)   : {0}" -f $overfull)
    Write-Host ("  欠满盒 (Underfull)  : {0}" -f $underfull)
    Write-Host ("  构建耗时            : {0:N1} s" -f $sw.Elapsed.TotalSeconds)

    if (Test-Path $pdf) {
        $kb = [math]::Round((Get-Item $pdf).Length / 1KB, 1)
        Write-Host ("  PDF                 : build/main.pdf ({0} KB)" -f $kb)
    }

    # ---- 未定义引用的明细（便于定位）--------------------------------
    if ($undefRef -gt 0 -or $undefCite -gt 0) {
        Write-Host ''
        Write-Host '  未定义引用/引文明细：' -ForegroundColor Yellow
        Select-String -Path $log -Pattern 'Reference `|Citation `' -Encoding UTF8 |
            Select-Object -First 20 |
            ForEach-Object { Write-Host ("    " + $_.Line.Trim()) -ForegroundColor Yellow }
    }

    Write-Host ''
    if ($code -ne 0 -or $errors -gt 0 -or $undefRef -gt 0 -or $undefCite -gt 0 -or $multiply -gt 0) {
        Write-Host '结论：构建不干净。' -ForegroundColor Red
        exit 1
    }
    Write-Host '结论：构建干净。' -ForegroundColor Green
    exit 0
}
finally {
    Pop-Location
}
