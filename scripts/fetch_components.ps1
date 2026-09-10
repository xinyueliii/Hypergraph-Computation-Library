param(
    [ValidateSet("all", "hyper-yolo", "superfast", "soft-hgnn", "yolov13", "e-hrsai", "e-3dtrack", "meshnet", "hyper-pcn", "count-anything", "pvrnet", "hgm2r")]
    [string]$Component = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CheckoutRoot = Join-Path $ProjectRoot "third_party"
$Git = "C:\Program Files\Git\cmd\git.exe"

$Definitions = @{
    "hyper-yolo" = @{
        Directory = "Hyper-YOLO"
        Repository = "https://github.com/iMoonLab/Hyper-YOLO.git"
        Revision = "9bfdabd8b97b5ee5da04e5df30d140a9d15557c5"
    }
    "superfast" = @{
        Directory = "SuperFast"
        Repository = "https://github.com/lisiqi19971013/SuperFast.git"
        Revision = "b4d29efc75c4a6ca07de4c5f56f555de48b65741"
    }
    "soft-hgnn" = @{
        Directory = "SoftHGNN"
        Repository = "https://github.com/Mengqi-Lei/SoftHGNN.git"
        Revision = "3f46b20eeb226616a8af89f321b2e1dc3ee3634f"
    }
    "yolov13" = @{
        Directory = "YOLOv13"
        Repository = "https://github.com/iMoonLab/yolov13.git"
        Revision = "73289949533efac82bb5f72ec19b746618656bd2"
    }
    "e-hrsai" = @{
        Directory = "E-HRSAI"
        Repository = "https://github.com/lisiqi19971013/E-HRSAI.git"
        Revision = "8cf3a1c0d1befb57d0d95df4c4a6c90f6c500ceb"
    }
    "e-3dtrack" = @{
        Directory = "E-3DTrack"
        Repository = "https://github.com/lisiqi19971013/E-3DTrack.git"
        Revision = "9f4d61a0912a6337b7595e87337dd3b144e62a00"
    }
    "meshnet" = @{
        Directory = "MeshNet"
        Repository = "https://github.com/iMoonLab/MeshNet.git"
        Revision = "70f9115a121cef71f62d774088771337c3beaf4b"
    }
    "hyper-pcn" = @{
        Directory = "Hyper-PCN"
        Repository = "https://github.com/Rinfly/Hyper-PCN.git"
        Revision = "e890be8653af8ac55ee73d573936d50d461e80e5"
    }
    "count-anything" = @{
        Directory = "Count-Anything"
        Repository = "https://github.com/Mengqi-Lei/count-anything.git"
        Revision = "0fffaefb3bbdcd930c135dcc02e2e359a038f2cb"
    }
    "pvrnet" = @{
        Directory = "PVRNet"
        Repository = "https://github.com/iMoonLab/PVRNet.git"
        Revision = "7b07d62788e67b4052c36b9ca37f6163234a4648"
    }
    "hgm2r" = @{
        Directory = "HGM2R"
        Repository = "https://github.com/iMoonLab/HGM2R.git"
        Revision = "94d88e5fc727a0abcf730c0528f67484aa1def7f"
    }
}

function Invoke-Git {
    param([string[]]$Arguments)
    & $Git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Fetch-Component {
    param([string]$Id)

    $Definition = $Definitions[$Id]
    $Target = Join-Path $CheckoutRoot $Definition.Directory
    $GitDirectory = Join-Path $Target ".git"

    if (Test-Path -LiteralPath $GitDirectory) {
        $CurrentRevision = (& $Git -c "safe.directory=$Target" -C $Target rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Cannot inspect existing checkout: $Target"
        }
        if ($CurrentRevision -ne $Definition.Revision) {
            throw "$Id is already checked out at $CurrentRevision; expected $($Definition.Revision)."
        }
        Write-Host "$Id already matches $CurrentRevision"
        return
    }

    if (Test-Path -LiteralPath $Target) {
        $ExistingItems = @(Get-ChildItem -Force -LiteralPath $Target)
        if ($ExistingItems.Count -gt 0) {
            throw "Target exists and is not an empty Git checkout: $Target"
        }
    } else {
        New-Item -ItemType Directory -Path $Target | Out-Null
    }

    Invoke-Git -Arguments @("-C", $Target, "init")
    Invoke-Git -Arguments @("-C", $Target, "remote", "add", "origin", $Definition.Repository)
    Invoke-Git -Arguments @("-C", $Target, "fetch", "--depth", "1", "origin", $Definition.Revision)
    Invoke-Git -Arguments @("-C", $Target, "checkout", "--detach", "FETCH_HEAD")
    Write-Host "$Id downloaded to $Target at $($Definition.Revision)"
}

New-Item -ItemType Directory -Force -Path $CheckoutRoot | Out-Null
$Selected = if ($Component -eq "all") {
    @("hyper-yolo", "superfast", "soft-hgnn", "yolov13", "e-hrsai", "e-3dtrack", "meshnet", "hyper-pcn", "count-anything", "pvrnet", "hgm2r")
} else {
    @($Component)
}
foreach ($Id in $Selected) {
    Fetch-Component $Id
}
