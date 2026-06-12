# ============================================================
# push_to_github.ps1 — 一键推送 people-daily-economy-daily
# ============================================================
# 用法:
#   .\push_to_github.ps1
#   .\push_to_github.ps1 -GitHubUser "yourname" -RepoName "people-daily-economy-daily"
# 前置:
#   1. git 已安装并在 PATH
#   2. 已在 GitHub 创建同名空仓库
#   3. 已配置认证 (PAT 或 SSH)
# ============================================================

[CmdletBinding()]
param(
    [string]$GitHubUser = "",
    [string]$RepoName = "people-daily-economy-daily",
    [string]$CommitMsg = "init: 人民日报经济新闻每日热点报告系统"
)

# --- 颜色函数 ---
function Cyan($s)  { Write-Host $s -ForegroundColor Cyan }
function Green($s) { Write-Host $s -ForegroundColor Green }
function Yellow($s){ Write-Host $s -ForegroundColor Yellow }
function Red($s)   { Write-Host $s -ForegroundColor Red }

Cyan "`n=== people-daily-economy-daily 一键推送 ===`n"

# --- 1. 检查 git ---
Cyan "[1/8] 检查 git..."
$git = (Get-Command git -ErrorAction SilentlyContinue)
if (-not $git) {
    Red "[X] 未检测到 git,请先安装 Git for Windows: https://git-scm.com/download/win"
    exit 1
}
Green "[OK] git 版本: $(git --version)"

# --- 2. 询问 GitHub 用户名 ---
if (-not $GitHubUser) {
    Cyan "[2/8] 请输入你的 GitHub 用户名:"
    $GitHubUser = Read-Host "GitHub Username"
}
if ([string]::IsNullOrWhiteSpace($GitHubUser)) {
    Red "[X] 用户名不能为空"
    exit 1
}
Green "[OK] GitHub 用户: $GitHubUser"

# --- 3. cd 到脚本所在目录 ---
Cyan "[3/8] 切换到项目目录..."
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Green "[OK] 当前目录: $(Get-Location)"

# --- 4. git init ---
Cyan "[4/8] 初始化 git 仓库..."
if (-not (Test-Path ".git")) {
    git init | Out-Null
    git branch -M main | Out-Null
    Green "[OK] 仓库已初始化 (branch=main)"
} else {
    Yellow "[..] .git 已存在,跳过 init"
}

# --- 5. 配置 user.name / user.email ---
Cyan "[5/8] 检查 git 用户配置..."
$userName = (git config user.name 2>$null)
$userEmail = (git config user.email 2>$null)
if (-not $userName) {
    Yellow "[!] 未配置 user.name,请输入(用于提交签名):"
    $inputName = Read-Host "Your Name"
    if ($inputName) { git config user.name $inputName; Green "[OK] user.name 已设置" }
}
if (-not $userEmail) {
    Yellow "[!] 未配置 user.email,请输入:"
    $inputEmail = Read-Host "your@email.com"
    if ($inputEmail) { git config user.email $inputEmail; Green "[OK] user.email 已设置" }
}
if ($userName -and $userEmail) {
    Green "[OK] 用户: $userName <$userEmail>"
}

# --- 6. .gitignore 与空目录占位 ---
Cyan "[6/8] 确保 .gitignore 与空目录占位..."
if (-not (Test-Path ".gitignore")) {
    Yellow "[!] 未发现 .gitignore,创建基础版本"
    @'
__pycache__/
*.py[cod]
.venv/
.env
.ipynb_checkpoints/
'@ | Out-File -Encoding utf8 ".gitignore"
}
if (-not (Test-Path "images\.gitkeep")) { New-Item -ItemType File -Path "images\.gitkeep" | Out-Null }
if (-not (Test-Path "data\raw\.gitkeep")) { New-Item -ItemType File -Path "data\raw\.gitkeep" | Out-Null }
if (-not (Test-Path "data\processed\.gitkeep")) { New-Item -ItemType File -Path "data\processed\.gitkeep" | Out-Null }
Green "[OK] 占位文件已就绪"

# --- 7. add + commit ---
Cyan "[7/8] 添加并提交..."
git add .
$status = git status --short
if ([string]::IsNullOrWhiteSpace($status)) {
    Yellow "[..] 无变更需要提交"
} else {
    git commit -m $CommitMsg 2>&1 | Out-Null
    Green "[OK] 已提交: $CommitMsg"
}

# --- 8. 配置 remote + push ---
Cyan "[8/8] 配置 remote 并推送..."
$remoteUrl = "https://github.com/$GitHubUser/$RepoName.git"
$existingRemote = (git remote get-url origin 2>$null)
if ($existingRemote) {
    Yellow "[..] origin 已存在: $existingRemote"
} else {
    git remote add origin $remoteUrl
    Green "[OK] 已添加 remote: $remoteUrl"
}

Write-Host ""
Cyan "准备推送到: $remoteUrl"
Cyan "请确认已在 GitHub 创建空仓库 https://github.com/$GitHubUser/$RepoName"
$confirm = Read-Host "继续推送? (Y/n)"
if ($confirm -eq "n" -or $confirm -eq "N") {
    Yellow "[!] 已取消推送,稍后可手动执行: git push -u origin main"
    exit 0
}

try {
    git push -u origin main
    Green "`n[OK] 推送成功!访问: https://github.com/$GitHubUser/$RepoName`n"
} catch {
    Red "`n[X] 推送失败。常见原因:"
    Yellow "  1. 仓库未创建或名称不匹配"
    Yellow "  2. 认证失败 — 推荐使用 Personal Access Token"
    Yellow "     GitHub → Settings → Developer settings → PAT (classic) → 勾选 repo"
    Yellow "  3. 网络问题 — 可换用 SSH: git remote set-url origin git@github.com:$GitHubUser/$RepoName.git"
    Write-Host ""
}

Cyan "=== 完成 ===`n"