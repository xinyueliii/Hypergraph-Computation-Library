param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [int64]$ExpectedSize,

    [ValidateRange(1, 16)]
    [int]$Concurrency = 8,

    [ValidateRange(1048576, 1073741824)]
    [int64]$ChunkSize = 67108864
)

$ErrorActionPreference = "Stop"
$output = [System.IO.Path]::GetFullPath($OutputPath)
$parent = Split-Path -Parent $output
$chunkRoot = "$output.parts"

New-Item -ItemType Directory -Path $parent -Force | Out-Null
New-Item -ItemType Directory -Path $chunkRoot -Force | Out-Null

if ($ExpectedSize -le 0) {
    throw "ExpectedSize must be supplied for a deterministic ranged download."
}

$chunks = @()
for ($start = [int64]0; $start -lt $ExpectedSize; $start += $ChunkSize) {
    $end = [Math]::Min($start + $ChunkSize - 1, $ExpectedSize - 1)
    $index = $chunks.Count
    $chunks += [pscustomobject]@{
        Index = $index
        Start = $start
        End = $end
        Size = $end - $start + 1
        Path = Join-Path $chunkRoot ("part-{0:D5}.bin" -f $index)
    }
}

$pending = [System.Collections.Generic.Queue[object]]::new()
foreach ($chunk in $chunks) {
    if ((Test-Path -LiteralPath $chunk.Path) -and
        ((Get-Item -LiteralPath $chunk.Path).Length -eq $chunk.Size)) {
        continue
    }
    $pending.Enqueue($chunk)
}

$running = @()
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $Concurrency) {
        $chunk = $pending.Dequeue()
        $job = Start-Job -ScriptBlock {
            param($DownloadUrl, $RangeStart, $RangeEnd, $Destination)
            & curl.exe --ssl-no-revoke -L --fail --retry 4 --retry-delay 3 `
                --connect-timeout 30 --max-time 900 -sS `
                -r "$RangeStart-$RangeEnd" -o $Destination $DownloadUrl
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed with exit code $LASTEXITCODE"
            }
        } -ArgumentList $Url, $chunk.Start, $chunk.End, $chunk.Path
        $running += [pscustomobject]@{ Job = $job; Chunk = $chunk }
    }

    $completed = $running | Where-Object { $_.Job.State -in @("Completed", "Failed", "Stopped") }
    if ($completed.Count -eq 0) {
        Start-Sleep -Seconds 2
        continue
    }

    foreach ($item in $completed) {
        Receive-Job -Job $item.Job -ErrorAction Stop | Out-Null
        Remove-Job -Job $item.Job -Force
        if (-not (Test-Path -LiteralPath $item.Chunk.Path)) {
            throw "Missing downloaded chunk $($item.Chunk.Index)."
        }
        $actual = (Get-Item -LiteralPath $item.Chunk.Path).Length
        if ($actual -ne $item.Chunk.Size) {
            throw "Chunk $($item.Chunk.Index) has $actual bytes; expected $($item.Chunk.Size)."
        }
        Write-Host "Completed chunk $($item.Chunk.Index + 1)/$($chunks.Count)"
        $running = @($running | Where-Object { $_.Job.Id -ne $item.Job.Id })
    }
}

$outputStream = [System.IO.File]::Open($output, [System.IO.FileMode]::Create)
try {
    foreach ($chunk in $chunks) {
        $inputStream = [System.IO.File]::OpenRead($chunk.Path)
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $inputStream.Dispose()
        }
    }
} finally {
    $outputStream.Dispose()
}

$finalSize = (Get-Item -LiteralPath $output).Length
if ($finalSize -ne $ExpectedSize) {
    throw "Final file has $finalSize bytes; expected $ExpectedSize."
}

Remove-Item -LiteralPath $chunkRoot -Recurse -Force
Get-Item -LiteralPath $output
