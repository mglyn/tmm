param(
    [string]$MainFile = "main.tex",
    [switch]$Clean,
    [switch]$Open,
    [switch]$ForcePdflatex
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[build] $Message" -ForegroundColor Cyan
}

function Remove-BuildArtifacts {
    param(
        [string]$BaseName
    )

    $extensions = @(
        "aux", "bbl", "bcf", "blg", "fdb_latexmk", "fls", "lof", "log",
        "lot", "nav", "out", "run.xml", "snm", "synctex.gz", "toc", "xdv"
    )

    foreach ($ext in $extensions) {
        $path = Join-Path $PSScriptRoot "$BaseName.$ext"
        if (Test-Path $path) {
            Remove-Item $path -Force
        }
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    Write-Step ("Running: {0} {1}" -f $FilePath, ($ArgumentList -join " "))
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2}" -f $LASTEXITCODE, $FilePath, ($ArgumentList -join " "))
    }
}

$mainPath = Join-Path $PSScriptRoot $MainFile
if (-not (Test-Path $mainPath)) {
    throw "Main TeX file not found: $mainPath"
}

$baseName = [System.IO.Path]::GetFileNameWithoutExtension($mainPath)
$pdfPath = Join-Path $PSScriptRoot "$baseName.pdf"
$texName = [System.IO.Path]::GetFileName($mainPath)

Push-Location $PSScriptRoot
try {
    if ($Clean) {
        Write-Step "Cleaning LaTeX build artifacts"
        Remove-BuildArtifacts -BaseName $baseName
    }

    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue

    if ($latexmk -and -not $ForcePdflatex) {
        Write-Step "Using latexmk"
        Invoke-CheckedCommand -FilePath $latexmk.Source -ArgumentList @(
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            $texName
        )
    }
    else {
        Write-Step "latexmk not found or disabled, falling back to pdflatex + bibtex"

        $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
        if (-not $pdflatex) {
            throw "pdflatex not found in PATH. Please install a TeX distribution or use latexmk."
        }

        Invoke-CheckedCommand -FilePath $pdflatex.Source -ArgumentList @(
            "-interaction=nonstopmode",
            "-halt-on-error",
            $texName
        )

        $auxPath = Join-Path $PSScriptRoot "$baseName.aux"
        if (Test-Path $auxPath) {
            $auxContent = Get-Content $auxPath -Raw
            if ($auxContent -match "\\bibdata" -or (Test-Path (Join-Path $PSScriptRoot "references.bib"))) {
                $bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
                if (-not $bibtex) {
                    throw "bibtex not found in PATH, but bibliography is required."
                }

                Invoke-CheckedCommand -FilePath $bibtex.Source -ArgumentList @($baseName)
            }
        }

        Invoke-CheckedCommand -FilePath $pdflatex.Source -ArgumentList @(
            "-interaction=nonstopmode",
            "-halt-on-error",
            $texName
        )
        Invoke-CheckedCommand -FilePath $pdflatex.Source -ArgumentList @(
            "-interaction=nonstopmode",
            "-halt-on-error",
            $texName
        )
    }

    if (-not (Test-Path $pdfPath)) {
        throw "Build finished without generating PDF: $pdfPath"
    }

    Write-Step "Build succeeded: $pdfPath"

    if ($Open) {
        Write-Step "Opening PDF"
        Start-Process $pdfPath
    }
}
finally {
    Pop-Location
}
