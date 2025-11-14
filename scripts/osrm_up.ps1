# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\osrm_up.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------------------
# Paths & constants
# -------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RoadnetDir  = Resolve-Path (Join-Path $ProjectRoot "roadnet")
$PbfPath     = Join-Path $RoadnetDir "beijing.osm.pbf"

$ContainerName = "osrm-beijing"
$Image         = "osrm/osrm-backend"
$ListenPort    = 5000

# Geofabrik Beijing PBF and MD5
$PbfUrl    = "https://download.geofabrik.de/asia/china/beijing-latest.osm.pbf"
$PbfMd5Url = "https://download.geofabrik.de/asia/china/beijing-latest.osm.pbf.md5"

# -------------------------------
# Helpers (ASCII only to avoid encoding issues)
# -------------------------------
function Log-Step($msg){ Write-Host "[step] $msg" -ForegroundColor Cyan }
function Log-Ok($msg){ Write-Host "[ok]   $msg" -ForegroundColor Green }
function Log-Warn($msg){ Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Log-Err($msg){ Write-Host "[err]  $msg" -ForegroundColor Red }

function Ensure-Docker {
  try {
    docker version | Out-Null
  } catch {
    Log-Err "Docker is not available. Please install/start Docker Desktop."
    exit 1
  }
}

function Download-PBF {
  param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$Md5Url,
    [Parameter(Mandatory=$true)][string]$OutFile
  )
  Log-Step "Downloading Beijing OSM PBF..."
  Invoke-WebRequest -Uri $Url -OutFile $OutFile

  try {
    Log-Step "Verifying MD5..."
    $remoteMd5 = (Invoke-WebRequest -Uri $Md5Url).Content.Trim().Split(" ")[0].ToLower()
    $localMd5  = (Get-FileHash -Algorithm MD5 -Path $OutFile).Hash.ToLower()
    if ($localMd5 -ne $remoteMd5) {
      Remove-Item -Force $OutFile -ErrorAction SilentlyContinue
      throw "MD5 mismatch (local=$localMd5, remote=$remoteMd5). File removed. Please rerun."
    }
    Log-Ok "MD5 OK: $localMd5"
  } catch {
    Log-Warn "MD5 check failed or remote MD5 unavailable: $($_.Exception.Message)"
    Log-Warn "Will continue with the downloaded file. If extract fails, rerun to redownload."
  }
}

function Run-Docker {
  param([string[]]$Args)
  $cmd = "docker " + ($Args -join " ")
  Write-Host $cmd -ForegroundColor DarkGray
  & docker @Args
}

# -------------------------------
# Main
# -------------------------------
Write-Host "[root] $ProjectRoot"
Write-Host "[data] $RoadnetDir"
Ensure-Docker

# 0) Cleanup old container (if any)
Log-Step "Cleanup existing container (if any)"
docker stop $ContainerName 2>$null | Out-Null
docker rm   $ContainerName 2>$null | Out-Null

# 1) Ensure PBF exists
if (-not (Test-Path $PbfPath)) {
  Log-Step "PBF not found. Downloading to: $PbfPath"
  Download-PBF -Url $PbfUrl -Md5Url $PbfMd5Url -OutFile $PbfPath
} else {
  Log-Ok "PBF exists: $PbfPath"
}

# 2) Resolve absolute host path for volume mount
$HostDir = (Resolve-Path $RoadnetDir).Path
Write-Host ("[mount] " + "${HostDir}:`/data")  # note: escaping for preview only

# 3) osrm-extract
Log-Step "osrm-extract (this may take a while)"
Run-Docker @("run","--rm","-v","${HostDir}:/data",$Image,
  "osrm-extract","-p","/opt/car.lua","/data/beijing.osm.pbf")

# 4) osrm-partition
Log-Step "osrm-partition"
Run-Docker @("run","--rm","-v","${HostDir}:/data",$Image,
  "osrm-partition","/data/beijing.osrm")

# 5) osrm-customize
Log-Step "osrm-customize"
Run-Docker @("run","--rm","-v","${HostDir}:/data",$Image,
  "osrm-customize","/data/beijing.osrm")

# 6) osrm-routed (service)
Log-Step "Starting osrm-routed at http://localhost:$ListenPort"
Run-Docker @("run","--name",$ContainerName,"-p","${ListenPort}:5000","-v","${HostDir}:/data",$Image,
  "osrm-routed","--algorithm","mld","/data/beijing.osrm")

Log-Ok "OSRM is up: http://localhost:$ListenPort"
