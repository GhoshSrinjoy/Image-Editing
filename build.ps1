<#
.SYNOPSIS
  Builds imgedit on Windows: the command-line tool, the WebAssembly module for
  the web page, and the tests.

.DESCRIPTION
  Uses Zig as the C compiler. If zig is not on PATH, a portable copy is
  downloaded into .tools\ the first time.

.EXAMPLE
  ./build.ps1                # CLI + wasm + tests
  ./build.ps1 cli            # just build\imgedit.exe
  ./build.ps1 -Runtime       # also fetch ONNX Runtime (DirectML: GPU on any DX12 card) into build\
#>
param(
  [ValidateSet('all', 'cli', 'wasm', 'test')][string]$Target = 'all',
  [switch]$Runtime
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$zigVersion = '0.17.0'
$ortVersion = '1.24.4'   # last release of the DirectML package
$dmlVersion = '1.15.4'

function Get-Zig {
  $cmd = Get-Command zig -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $dir = Join-Path $root ".tools\zig-x86_64-windows-$zigVersion"
  $exe = Join-Path $dir 'zig.exe'
  if (-not (Test-Path $exe)) {
    Write-Host "Downloading Zig $zigVersion (portable, ~100 MB) into .tools\ ..."
    New-Item -ItemType Directory -Force (Join-Path $root '.tools') | Out-Null
    $zip = Join-Path $root ".tools\zig.zip"
    Invoke-WebRequest "https://ziglang.org/download/$zigVersion/zig-x86_64-windows-$zigVersion.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath (Join-Path $root '.tools') -Force
    Remove-Item $zip
  }
  return $exe
}

function Invoke-Zig([string[]]$Arguments) {
  & $script:zig @Arguments
  if ($LASTEXITCODE -ne 0) { throw "zig $($Arguments -join ' ') failed" }
}

$zig = Get-Zig
Push-Location $root
try {
  New-Item -ItemType Directory -Force build | Out-Null
  $lib = 'parallel', 'image', 'png', 'color', 'geometry', 'filter', 'matte' | ForEach-Object { "src/$_.c" }
  $warn = '-Wall', '-Wextra', '-Wno-unused-function'

  if ($Target -in 'all', 'cli') {
    Invoke-Zig (@('cc', '-std=c11', '-O3') + $warn + $lib + @('src/nn.c', 'src/platform.c', 'src/cli.c', '-lshell32', '-o', 'build/imgedit.exe'))
    Write-Host 'built build\imgedit.exe'
  }
  if ($Target -in 'all', 'wasm') {
    Invoke-Zig (@('cc', '-target', 'wasm32-wasi', '-std=c11', '-O3', '-msimd128', '-mbulk-memory', '-mexec-model=reactor',
        '-DIE_NO_FILES', '-s', '-Wl,--max-memory=4294901760', '-Wl,-z,stack-size=1048576') + $warn + $lib + @('src/wasm.c', '-o', 'web/imgedit.wasm'))
    Write-Host 'built web\imgedit.wasm'
  }
  if ($Target -in 'all', 'test') {
    Invoke-Zig (@('cc', '-std=c11', '-O2') + $warn + $lib + @('tests/test_core.c', '-o', 'build/test_core.exe'))
    & build/test_core.exe
    if ($LASTEXITCODE -ne 0) { throw 'core tests failed' }
    if ((Test-Path web/imgedit.wasm) -and (Get-Command node -ErrorAction SilentlyContinue)) {
      node tests/wasm.test.mjs
      if ($LASTEXITCODE -ne 0) { throw 'wasm tests failed' }
    }
  }
  if ($Runtime) {
    $tmp = Join-Path $env:TEMP "imgedit-ort-$ortVersion"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    Write-Host "Downloading ONNX Runtime $ortVersion (DirectML) and DirectML $dmlVersion ..."
    Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ortVersion" -OutFile "$tmp\ort.zip"
    Invoke-WebRequest "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$dmlVersion" -OutFile "$tmp\dml.zip"
    Expand-Archive "$tmp\ort.zip" "$tmp\ort" -Force
    Expand-Archive "$tmp\dml.zip" "$tmp\dml" -Force
    Copy-Item "$tmp\ort\runtimes\win-x64\native\onnxruntime.dll", "$tmp\ort\runtimes\win-x64\native\onnxruntime_providers_shared.dll", "$tmp\dml\bin\x64-win\DirectML.dll" build\
    Write-Host 'ONNX Runtime copied next to build\imgedit.exe'
  }
}
finally {
  Pop-Location
}
