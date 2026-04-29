param(
    [string]$MainFile = "main.tex",
    [switch]$Clean,
    [switch]$Open,
    [switch]$ForcePdflatex
)

$paperBuildScript = Join-Path $PSScriptRoot "tmm_paper\build.ps1"
if (-not (Test-Path $paperBuildScript)) {
    throw "Paper build script not found: $paperBuildScript"
}

& $paperBuildScript -MainFile $MainFile -Clean:$Clean -Open:$Open -ForcePdflatex:$ForcePdflatex
exit $LASTEXITCODE
