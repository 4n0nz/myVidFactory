$ErrorActionPreference='SilentlyContinue'
$src='C:\Users\p4tzr1ot\Desktop\YT'
$out='C:\Users\p4tzr1ot\AppData\Local\Temp\videogen-poc\broll'
New-Item -ItemType Directory -Force $out | Out-Null
Get-ChildItem $out -Filter *.mp4 | Remove-Item -Force
$i=0
foreach($f in (Get-ChildItem $src -Filter *.mp4 | Sort-Object Name)){
  $o = Join-Path $out ("broll_{0:D2}.mp4" -f $i)
  # 4s à partir de ~1s, rempli 1080p (cover), muet
  & ffmpeg -y -ss 1 -t 4 -i $f.FullName -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1" -an -r 30 -c:v libx264 -preset veryfast -crf 23 $o 2>$null
  if(Test-Path $o){ $i++ }
}
"clips broll: $i"
"{0:N0} MB total" -f ((Get-ChildItem $out -Filter *.mp4 | Measure-Object Length -Sum).Sum/1MB)