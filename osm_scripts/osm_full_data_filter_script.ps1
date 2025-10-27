$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$IN  = "data\poland-251020.osm.pbf"
$OUT = "data\poland_full_data_filtered.pbf"

if ($env:CONDA_PREFIX) {
  $env:Path = "$env:CONDA_PREFIX\Library\bin;$env:CONDA_PREFIX\Scripts;$env:Path"
}
$OSMIUM = Join-Path $env:CONDA_PREFIX "Library\bin\osmium.exe"
if (-not (Test-Path $OSMIUM)) {
  throw "Aktywuj środowisko:  conda activate osmtools"
}

Write-Host "osmium: $OSMIUM"
Write-Host "pwd   : $(Get-Location)"
Write-Host "IN ok?: $(Test-Path $IN) -> $IN"

$args1 = @(
  "tags-filter", $IN,

  "nwr/shop=supermarket", "nwr/shop=convenience", "nwr/shop=bakery",
  "nwr/amenity=pharmacy",
  "nwr/amenity=clinic", "nwr/amenity=hospital",
  "nwr/amenity=parcel_locker",
  "nwr/highway=bus_stop",
  "nwr/railway=tram_stop",
  "nwr/amenity=university", "nwr/amenity=college",
  "nwr/amenity=library",
  "nwr/amenity=nightclub",
  "nwr/leisure=fitness_centre",
  "nwr/amenity=school",
  "nwr/amenity=kindergarten", "nwr/amenity=childcare",
  "nwr/leisure=playground",
  "nwr/leisure=park",
  "nwr/amenity=veterinary",
  "nwr/shop=pet",
  "nwr/amenity=pub",
  "nwr/railway=station", "nwr/railway=halt",
  "nwr/public_transport=station", "nwr/public_transport=halt",

  "nwr/highway=footway", "nwr/highway=pedestrian", "nwr/highway=path",
  "nwr/highway=steps", "nwr/highway=platform", "nwr/highway=crossing",
  "nwr/highway=living_street",
  "nwr/highway=residential", "nwr/highway=unclassified", "nwr/highway=service",
  "nwr/highway=cycleway",

  "nwr/highway=track",    

  "nwr/highway=primary", "nwr/highway=primary_link",
  "nwr/highway=secondary", "nwr/highway=secondary_link",
  "nwr/highway=tertiary", "nwr/highway=tertiary_link",

  "nwr/foot=yes", "nwr/foot=designated", "nwr/foot=permissive",

  "nwr/sidewalk=yes", "nwr/sidewalk=both", "nwr/sidewalk=left", "nwr/sidewalk=right",

  "-o", $OUT, "-O", "-v"
)

Write-Host "`n> $OSMIUM $($args1 -join ' ')" -ForegroundColor Cyan
& $OSMIUM @args1
if ($LASTEXITCODE -ne 0) { throw "osmium tags-filter failed: exit $LASTEXITCODE" }

& $OSMIUM "fileinfo" "-e" $OUT
& $OSMIUM "tags-count" $OUT | Select-String -Pattern "amenity=|shop=|leisure=|railway=|highway=|foot=|sidewalk="

Write-Host "`nOK -> $OUT"
