$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$pptPath = Join-Path $root 'iexplore_final_pre.pptx'
$backupPath = Join-Path $root 'iexplore_final_pre_backup_before_completion.pptx'
$workDir = Join-Path $root '.pptx_edit'

if (Test-Path $backupPath) {
    Remove-Item $backupPath -Force
}
Copy-Item $pptPath $backupPath -Force

if (Test-Path $workDir) {
    Remove-Item $workDir -Recurse -Force
}
New-Item -ItemType Directory -Path $workDir | Out-Null

$zipPath = Join-Path $workDir 'deck.zip'
$unzipDir = Join-Path $workDir 'unzipped'
Copy-Item $pptPath $zipPath
Expand-Archive -Path $zipPath -DestinationPath $unzipDir -Force

[xml]$presentationXml = Get-Content (Join-Path $unzipDir 'ppt\presentation.xml')
[xml]$presentationRelsXml = Get-Content (Join-Path $unzipDir 'ppt\_rels\presentation.xml.rels')
[xml]$contentTypesXml = Get-Content -LiteralPath (Join-Path $unzipDir '[Content_Types].xml')

$nsP = 'http://schemas.openxmlformats.org/presentationml/2006/main'
$nsA = 'http://schemas.openxmlformats.org/drawingml/2006/main'
$nsR = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
$nsPkg = 'http://schemas.openxmlformats.org/package/2006/relationships'

$nsMgr = New-Object System.Xml.XmlNamespaceManager($presentationXml.NameTable)
$nsMgr.AddNamespace('p', $nsP)
$nsMgr.AddNamespace('r', $nsR)

$relsNsMgr = New-Object System.Xml.XmlNamespaceManager($presentationRelsXml.NameTable)
$relsNsMgr.AddNamespace('rel', $nsPkg)

$contentNsMgr = New-Object System.Xml.XmlNamespaceManager($contentTypesXml.NameTable)
$contentNsMgr.AddNamespace('ct', 'http://schemas.openxmlformats.org/package/2006/content-types')

$slidesDir = Join-Path $unzipDir 'ppt\slides'
$slideRelsDir = Join-Path $slidesDir '_rels'

function New-TxBodyXml {
    param(
        [string[]]$Paragraphs,
        [bool]$IsTitle = $false
    )

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('<p:txBody xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">')
    if ($IsTitle) {
        [void]$sb.Append('<a:bodyPr/><a:lstStyle/>')
    } else {
        [void]$sb.Append('<a:bodyPr lIns="109728" tIns="109728" rIns="109728" bIns="91440" anchor="t"/><a:lstStyle/>')
    }

    foreach ($text in $Paragraphs) {
        $escaped = [System.Security.SecurityElement]::Escape($text)
        [void]$sb.Append('<a:p><a:r><a:rPr lang="en-US"><a:solidFill><a:srgbClr val="292929"/></a:solidFill><a:highlight><a:srgbClr val="FFFFFF"/></a:highlight><a:ea typeface="+mn-lt"/><a:cs typeface="+mn-lt"/></a:rPr><a:t>')
        [void]$sb.Append($escaped)
        [void]$sb.Append('</a:t></a:r><a:endParaRPr lang="en-US"><a:solidFill><a:srgbClr val="292929"/></a:solidFill><a:highlight><a:srgbClr val="FFFFFF"/></a:highlight><a:ea typeface="+mn-lt"/><a:cs typeface="+mn-lt"/></a:endParaRPr></a:p>')
    }

    [void]$sb.Append('</p:txBody>')
    return $sb.ToString()
}

function Set-SlideText {
    param(
        [string]$SlidePath,
        [string]$Title,
        [string[]]$BodyParagraphs
    )

    [xml]$slideXml = Get-Content $SlidePath
    $slideNsMgr = New-Object System.Xml.XmlNamespaceManager($slideXml.NameTable)
    $slideNsMgr.AddNamespace('p', $nsP)
    $slideNsMgr.AddNamespace('a', $nsA)
    $slideNsMgr.AddNamespace('r', $nsR)

    $titleShape = $slideXml.SelectSingleNode("//p:sp[p:nvSpPr/p:nvPr/p:ph[@type='title']]", $slideNsMgr)
    $bodyShape = $slideXml.SelectSingleNode("//p:sp[p:nvSpPr/p:nvPr/p:ph[@idx='1']]", $slideNsMgr)

    if (-not $titleShape -or -not $bodyShape) {
        throw "Could not locate title/content placeholders in $SlidePath"
    }

    $null = $titleShape.RemoveChild($titleShape.SelectSingleNode('p:txBody', $slideNsMgr))
    $titleFragment = $slideXml.CreateDocumentFragment()
    $titleFragment.InnerXml = New-TxBodyXml -Paragraphs @($Title) -IsTitle $true
    $null = $titleShape.AppendChild($titleFragment)

    $null = $bodyShape.RemoveChild($bodyShape.SelectSingleNode('p:txBody', $slideNsMgr))
    $bodyFragment = $slideXml.CreateDocumentFragment()
    $bodyFragment.InnerXml = New-TxBodyXml -Paragraphs $BodyParagraphs
    $null = $bodyShape.AppendChild($bodyFragment)

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($SlidePath, $slideXml.OuterXml, $utf8NoBom)
}

function Add-ImageToSlide {
    param(
        [string]$SlidePath,
        [string]$SlideRelsPath,
        [string]$MediaFileName,
        [string]$RelationshipId,
        [int]$ShapeId,
        [string]$Name,
        [int]$X,
        [int]$Y,
        [int]$Cx,
        [int]$Cy
    )

    [xml]$slideXml = Get-Content $SlidePath
    [xml]$relsXml = Get-Content $SlideRelsPath

    $slideNsMgr = New-Object System.Xml.XmlNamespaceManager($slideXml.NameTable)
    $slideNsMgr.AddNamespace('p', $nsP)
    $slideNsMgr.AddNamespace('a', $nsA)
    $slideNsMgr.AddNamespace('r', $nsR)

    $relsNsMgrLocal = New-Object System.Xml.XmlNamespaceManager($relsXml.NameTable)
    $relsNsMgrLocal.AddNamespace('rel', $nsPkg)

    $existingRel = $relsXml.SelectSingleNode("//rel:Relationship[@Id='$RelationshipId']", $relsNsMgrLocal)
    if (-not $existingRel) {
        $rel = $relsXml.CreateElement('Relationship', $nsPkg)
        $null = $rel.SetAttribute('Id', $RelationshipId)
        $null = $rel.SetAttribute('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
        $null = $rel.SetAttribute('Target', "../media/$MediaFileName")
        $null = $relsXml.Relationships.AppendChild($rel)
    }

    $picXml = @"
<p:pic xmlns:p="$nsP" xmlns:a="$nsA" xmlns:r="$nsR">
  <p:nvPicPr>
    <p:cNvPr id="$ShapeId" name="$Name"/>
    <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
    <p:nvPr/>
  </p:nvPicPr>
  <p:blipFill>
    <a:blip r:embed="$RelationshipId"/>
    <a:stretch><a:fillRect/></a:stretch>
  </p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="$X" y="$Y"/><a:ext cx="$Cx" cy="$Cy"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
</p:pic>
"@

    $fragment = $slideXml.CreateDocumentFragment()
    $fragment.InnerXml = $picXml
    $spTree = $slideXml.SelectSingleNode('//p:spTree', $slideNsMgr)
    $null = $spTree.AppendChild($fragment)

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($SlidePath, $slideXml.OuterXml, $utf8NoBom)
    [System.IO.File]::WriteAllText($SlideRelsPath, $relsXml.OuterXml, $utf8NoBom)
}

$newSlides = @(
    @{
        Number = 8
        Title = 'Algorithm and Data Summary'
        Body = @(
            'The figure summarizes the 1.5M+ sample trinary dataset and class imbalance.',
            'XGBoost uses the 5 lagged market and news features shown earlier to predict next-day volatility.'
        )
    },
    @{
        Number = 9
        Title = 'Model Output'
        Body = @(
            'Output is a probability vector: P(low), P(medium), and P(high) for each stock-day.',
            'The confusion matrix below shows where the test split is predicted well and where classes are mixed.'
        )
    },
    @{
        Number = 10
        Title = 'Benchmarks'
        Body = @(
            'Test-set benchmark headline: baseline accuracy 0.722, weighted F1 0.678, weighted OvR AUC 0.725.',
            'Feature importance and ROC curves show that market features dominate, while class weighting helps minority sensitivity.'
        )
    },
    @{
        Number = 11
        Title = 'Things Tried'
        Body = @(
            'Started from a 5-feature trinary XGBoost baseline with confidence sample weights.',
            'Tried inverse-frequency class weights to improve sensitivity on low- and high-volatility labels.',
            'Added engineered features and validation-tuned class thresholds to rebalance decision boundaries.',
            'Ran a chronological split with tuned hyperparameters as a harder, more realistic benchmark.',
            'Main lesson: robustness depends strongly on split design and class-balance handling, not just adding more features.'
        )
    }
)

$templateSlidePath = Join-Path $slidesDir 'slide7.xml'
$templateRelsPath = Join-Path $slideRelsDir 'slide7.xml.rels'

foreach ($slide in $newSlides) {
    $slidePath = Join-Path $slidesDir ("slide{0}.xml" -f $slide.Number)
    $relsPath = Join-Path $slideRelsDir ("slide{0}.xml.rels" -f $slide.Number)
    Copy-Item $templateSlidePath $slidePath -Force
    Copy-Item $templateRelsPath $relsPath -Force
    Set-SlideText -SlidePath $slidePath -Title $slide.Title -BodyParagraphs $slide.Body
}

$mediaDir = Join-Path $unzipDir 'ppt\media'
Copy-Item (Join-Path $root 'result\dataset_overview.png') (Join-Path $mediaDir 'image7.png') -Force
Copy-Item (Join-Path $root 'result\confusion_matrix_test.png') (Join-Path $mediaDir 'image8.png') -Force
Copy-Item (Join-Path $root 'result\roc_curves.png') (Join-Path $mediaDir 'image9.png') -Force
Copy-Item (Join-Path $root 'result\feature_importance.png') (Join-Path $mediaDir 'image10.png') -Force

Add-ImageToSlide -SlidePath (Join-Path $slidesDir 'slide8.xml') -SlideRelsPath (Join-Path $slideRelsDir 'slide8.xml.rels') -MediaFileName 'image7.png' -RelationshipId 'rId2' -ShapeId 20 -Name 'Dataset Overview' -X 900000 -Y 2400000 -Cx 10200000 -Cy 3600000
Add-ImageToSlide -SlidePath (Join-Path $slidesDir 'slide9.xml') -SlideRelsPath (Join-Path $slideRelsDir 'slide9.xml.rels') -MediaFileName 'image8.png' -RelationshipId 'rId2' -ShapeId 20 -Name 'Confusion Matrix' -X 3000000 -Y 2350000 -Cx 6200000 -Cy 3600000
Add-ImageToSlide -SlidePath (Join-Path $slidesDir 'slide10.xml') -SlideRelsPath (Join-Path $slideRelsDir 'slide10.xml.rels') -MediaFileName 'image9.png' -RelationshipId 'rId2' -ShapeId 20 -Name 'ROC Curves' -X 450000 -Y 2400000 -Cx 5400000 -Cy 3000000
Add-ImageToSlide -SlidePath (Join-Path $slidesDir 'slide10.xml') -SlideRelsPath (Join-Path $slideRelsDir 'slide10.xml.rels') -MediaFileName 'image10.png' -RelationshipId 'rId3' -ShapeId 21 -Name 'Feature Importance' -X 6500000 -Y 2400000 -Cx 4900000 -Cy 3000000

Set-SlideText -SlidePath $templateSlidePath -Title 'Future Work and Reflections' -BodyParagraphs @(
    'Current baseline on the trinary task reaches accuracy 0.722 and weighted F1 0.678 on the test split.',
    'Class balancing raises macro F1 to 0.623, showing that minority-class handling changes the conclusion materially.',
    'Future work: stronger financial text processing, sector and macro features, and stricter out-of-time validation.'
)

$relationships = $presentationRelsXml.Relationships
$slideIdList = $presentationXml.presentation.sldIdLst

$nextRid = 14
$nextSlideId = 263
$insertBeforeRid = 'rId8'

foreach ($slide in $newSlides) {
    $rel = $presentationRelsXml.CreateElement('Relationship', $nsPkg)
    $null = $rel.SetAttribute('Id', "rId$nextRid")
    $null = $rel.SetAttribute('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide')
    $null = $rel.SetAttribute('Target', "slides/slide$($slide.Number).xml")
    $null = $relationships.AppendChild($rel)

    $sldId = $presentationXml.CreateElement('p', 'sldId', $nsP)
    $null = $sldId.SetAttribute('id', $nextSlideId.ToString())
    $null = $sldId.SetAttribute('id', $nsR, "rId$nextRid")

    $anchor = $slideIdList.SelectSingleNode("p:sldId[@r:id='$insertBeforeRid']", $nsMgr)
    if (-not $anchor) {
        throw 'Could not find existing final slide relationship for insertion'
    }
    $null = $slideIdList.InsertBefore($sldId, $anchor)

    $override = $contentTypesXml.CreateElement('Override', 'http://schemas.openxmlformats.org/package/2006/content-types')
    $null = $override.SetAttribute('PartName', "/ppt/slides/slide$($slide.Number).xml")
    $null = $override.SetAttribute('ContentType', 'application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
    $null = $contentTypesXml.Types.AppendChild($override)

    $nextRid++
    $nextSlideId++
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $unzipDir 'ppt\presentation.xml'), $presentationXml.OuterXml, $utf8NoBom)
[System.IO.File]::WriteAllText((Join-Path $unzipDir 'ppt\_rels\presentation.xml.rels'), $presentationRelsXml.OuterXml, $utf8NoBom)
[System.IO.File]::WriteAllText((Join-Path $unzipDir '[Content_Types].xml'), $contentTypesXml.OuterXml, $utf8NoBom)

Remove-Item $zipPath -Force
Compress-Archive -Path (Join-Path $unzipDir '*') -DestinationPath $zipPath -Force
Copy-Item $zipPath $pptPath -Force

Write-Output "Updated presentation: $pptPath"
Write-Output "Backup created: $backupPath"
