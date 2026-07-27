[CmdletBinding()]
param(
    [ValidateSet("qa", "all")]
    [string]$Mode = "qa",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path -Path $PSScriptRoot -ChildPath "..")
)
$streamlitExe = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
$envFile = Join-Path $projectRoot ".env"


function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)]
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        return $null
    }

    $escapedName = [regex]::Escape($Name)
    $matchedLine = Get-Content -LiteralPath $envFile -Encoding UTF8 |
        Where-Object { $_ -match "^\s*$escapedName\s*=" } |
        Select-Object -Last 1
    if ($null -eq $matchedLine) {
        return $null
    }

    return ($matchedLine -replace "^\s*$escapedName\s*=\s*", "").Trim()
}


function Test-PortListening {
    param(
        [Parameter(Mandatory)]
        [int]$Port
    )

    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().
        GetActiveTcpListeners()
    return $null -ne ($listeners | Where-Object { $_.Port -eq $Port } |
        Select-Object -First 1)
}


function Start-RagServiceWindow {
    param(
        [Parameter(Mandatory)]
        [string]$Title,

        [Parameter(Mandatory)]
        [string]$AppFile,

        [Parameter(Mandatory)]
        [int]$Port
    )

    if (Test-PortListening -Port $Port) {
        Write-Host "端口 $Port 已有服务监听，跳过重复启动。" -ForegroundColor Yellow
        return
    }

    $appPath = Join-Path $projectRoot $AppFile
    if (-not (Test-Path -LiteralPath $appPath -PathType Leaf)) {
        throw "缺少应用文件：$appPath"
    }

    if ($DryRun) {
        Write-Host "[检查模式] 将启动：$AppFile，端口：$Port"
        return
    }

    $escapedTitle = $Title.Replace("'", "''")
    $escapedRoot = $projectRoot.Replace("'", "''")
    $escapedStreamlit = $streamlitExe.Replace("'", "''")
    $escapedApp = $appPath.Replace("'", "''")
    $childScript = @"
`$Host.UI.RawUI.WindowTitle = '$escapedTitle'
Set-Location -LiteralPath '$escapedRoot'
& '$escapedStreamlit' run '$escapedApp' --server.port $Port --server.headless true
if (`$LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '服务异常退出，请查看上方日志。' -ForegroundColor Red
}
"@
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($childScript)
    )

    Start-Process `
        -FilePath "powershell.exe" `
        -WorkingDirectory $projectRoot `
        -ArgumentList @(
            "-NoLogo",
            "-NoProfile",
            "-NoExit",
            "-EncodedCommand",
            $encodedCommand
        ) | Out-Null
}


function Wait-ServiceReady {
    param(
        [Parameter(Mandatory)]
        [string]$Url,

        [int]$TimeoutSeconds = 20
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $null = Invoke-WebRequest `
                -Uri $Url `
                -UseBasicParsing `
                -TimeoutSec 1
            return $true
        }
        catch {
            Start-Sleep -Milliseconds 300
        }
    }
    return $false
}


try {
    if (-not (Test-Path -LiteralPath $streamlitExe -PathType Leaf)) {
        throw (
            "未找到项目虚拟环境。请先运行：" +
            ".\.venv\Scripts\python.exe -m pip install -r requirements.lock"
        )
    }

    $apiKey = $env:DASHSCOPE_API_KEY
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        $apiKey = Get-DotEnvValue -Name "DASHSCOPE_API_KEY"
    }
    if (
        [string]::IsNullOrWhiteSpace($apiKey) -or
        $apiKey -like "replace-with-*"
    ) {
        throw "DASHSCOPE_API_KEY 尚未配置，请先编辑项目根目录下的 .env。"
    }

    $targets = @(
        @{
            Title = "RAG 问答服务"
            App = "app_qa.py"
            Port = 8501
            Url = "http://localhost:8501"
        }
    )
    if ($Mode -eq "all") {
        $targets += @{
            Title = "RAG 知识库管理"
            App = "app_file_uploader.py"
            Port = 8502
            Url = "http://localhost:8502"
        }

        $adminToken = $env:RAG_ADMIN_TOKEN
        if ([string]::IsNullOrWhiteSpace($adminToken)) {
            $adminToken = Get-DotEnvValue -Name "RAG_ADMIN_TOKEN"
        }
        if ([string]::IsNullOrWhiteSpace($adminToken) -or $adminToken.Length -lt 24) {
            Write-Host (
                "警告：RAG_ADMIN_TOKEN 未配置或不足 24 个字符，" +
                "知识库管理页将保持锁定。"
            ) -ForegroundColor Yellow
        }
    }

    foreach ($target in $targets) {
        Start-RagServiceWindow `
            -Title $target.Title `
            -AppFile $target.App `
            -Port $target.Port
    }

    if ($DryRun) {
        Write-Host "启动器检查通过。" -ForegroundColor Green
        exit 0
    }

    foreach ($target in $targets) {
        if (Wait-ServiceReady -Url $target.Url) {
            Start-Process $target.Url
        }
        else {
            Write-Host (
                "服务尚未在 20 秒内就绪：$($target.Url)。" +
                "请查看对应的 PowerShell 服务窗口。"
            ) -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host ""
    Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
