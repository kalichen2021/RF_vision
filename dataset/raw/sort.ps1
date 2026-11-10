$dir = ".\env"
$files = Get-ChildItem -LiteralPath $dir -File | Sort-Object Name

# 第一阶段：改成唯一临时名
$tmpList = @()
foreach ($f in $files) {
    $tmpName = "__tmp_" + [guid]::NewGuid().ToString("N")
    Rename-Item -LiteralPath $f.FullName -NewName $tmpName
    $tmpList += [pscustomobject]@{
        Path = Join-Path $dir $tmpName
        Ext  = $f.Extension      # 含点，如 ".jpg"；无扩展名为 ""
    }
}

# 第二阶段：改成 0000.ext
$i = 0
foreach ($x in $tmpList) {
    $newName = "{0:D4}{1}" -f $i, $x.Ext
    Rename-Item -LiteralPath $x.Path -NewName $newName
    $i++
}
Write-Host "Done. Renamed $i files."