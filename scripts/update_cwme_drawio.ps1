param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$BackupPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Drawio file not found: $Path"
}

Copy-Item -LiteralPath $Path -Destination $BackupPath -Force
[xml]$doc = Get-Content -Raw -LiteralPath $Path
$mxfile = $doc.DocumentElement

foreach ($old in @($mxfile.SelectNodes('./diagram'))) {
    if ($old.GetAttribute('id') -eq 'pcm-davis-from-8png' -and $old.GetAttribute('name') -eq 'CWME') {
        $old.SetAttribute('name', 'CWME (archived)')
    }
}

$oldPage = $mxfile.SelectSingleNode('./diagram[@id="cwme-iclr-overview"]')
if ($null -ne $oldPage) { $mxfile.RemoveChild($oldPage) | Out-Null }

$page = $doc.CreateElement('diagram')
$page.SetAttribute('id', 'cwme-iclr-overview')
$page.SetAttribute('name', 'CWME Architecture (ICLR)')
$model = $doc.CreateElement('mxGraphModel')
foreach ($pair in @{
    dx='1600'; dy='1200'; grid='0'; gridSize='10'; guides='1'; tooltips='1'; connect='1'; arrows='1'; fold='1'; page='1'; pageScale='1'; pageWidth='1120'; pageHeight='1100'; math='1'; shadow='0'
}.GetEnumerator()) { $model.SetAttribute($pair.Key, $pair.Value) }
$root = $doc.CreateElement('root')
$zero = $doc.CreateElement('mxCell'); $zero.SetAttribute('id','0'); [void]$root.AppendChild($zero)
$layer = $doc.CreateElement('mxCell'); $layer.SetAttribute('id','1'); $layer.SetAttribute('parent','0'); [void]$root.AppendChild($layer)
[void]$model.AppendChild($root); [void]$page.AppendChild($model)
if ($mxfile.FirstChild) { [void]$mxfile.InsertBefore($page, $mxfile.FirstChild) } else { [void]$mxfile.AppendChild($page) }

function Add-Geometry($cell, [string]$X, [string]$Y, [string]$W, [string]$H) {
    $g = $doc.CreateElement('mxGeometry')
    $g.SetAttribute('x', $X); $g.SetAttribute('y', $Y); $g.SetAttribute('width', $W); $g.SetAttribute('height', $H); $g.SetAttribute('as', 'geometry')
    [void]$cell.AppendChild($g)
}

function Add-Vertex([string]$Id, [string]$Value, [string]$Style, [string]$X, [string]$Y, [string]$W, [string]$H) {
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id', $Id); $cell.SetAttribute('value', $Value); $cell.SetAttribute('style', $Style); $cell.SetAttribute('parent', '1'); $cell.SetAttribute('vertex', '1')
    Add-Geometry $cell $X $Y $W $H; [void]$root.AppendChild($cell)
}

function Add-Edge([string]$Id, [string]$Source, [string]$Target, [string]$Style, [string]$Value = '') {
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id', $Id); $cell.SetAttribute('value', $Value); $cell.SetAttribute('style', $Style); $cell.SetAttribute('parent', '1'); $cell.SetAttribute('source', $Source); $cell.SetAttribute('target', $Target); $cell.SetAttribute('edge', '1')
    $g = $doc.CreateElement('mxGeometry'); $g.SetAttribute('relative', '1'); $g.SetAttribute('as', 'geometry'); [void]$cell.AppendChild($g); [void]$root.AppendChild($cell)
}

$outer = 'rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeWidth=1.6;fontFamily=Times New Roman;align=left;verticalAlign=top;spacingLeft=12;spacingTop=6;'
$text = 'text;html=1;fontFamily=Times New Roman;align=center;verticalAlign=middle;'
$node = 'rounded=1;whiteSpace=wrap;html=1;arcSize=10;strokeWidth=1.2;fontFamily=Times New Roman;fontSize=14;align=center;verticalAlign=middle;'
$edge = 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;endSize=6;strokeWidth=1.4;fontFamily=Times New Roman;'

Add-Vertex 'title' '<b><font style="font-size:22px">CWME: Continual World-Model Evolution</font></b><br><font color="#5C6B76">Evidence-bearing external memory around a frozen language agent</font>' $text '80' '10' '960' '58'

# 1. Current interaction and state construction.
Add-Vertex 'b1' '' ($outer + 'fillColor=#F4F8FB;strokeColor=#3C6F9A;') '80' '82' '980' '150'
Add-Vertex 's1' '1' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;strokeWidth=0;fontFamily=Times New Roman;fontStyle=1;fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;fillColor=#3C6F9A;' '96' '94' '26' '26'
Add-Vertex 't1' '<font color="#3C6F9A"><b>Perception</b></font>' ($text + 'fontSize=16;align=left;') '130' '92' '220' '30'
Add-Vertex 'e1' '$$s_{e,t}=\psi(g_e,o_{e,t})$$' ($text + 'fontSize=13;align=right;fontColor=#3C6F9A;') '620' '92' '420' '30'
Add-Vertex 'task' '<b>Task</b><br>$$g_e$$' ($node + 'fillColor=#FFFFFF;strokeColor=#3C6F9A;fontColor=#1F2A33;') '108' '136' '168' '72'
Add-Vertex 'obs' '<b>Observation</b><br>$$o_{e,t}$$' ($node + 'fillColor=#FFFFFF;strokeColor=#3C6F9A;fontColor=#1F2A33;') '316' '136' '188' '72'
Add-Vertex 'parser' '<b>Parser</b><br>$$\psi$$' ($node + 'fillColor=#E8F1F8;strokeColor=#3C6F9A;fontColor=#1F2A33;') '544' '136' '188' '72'
Add-Vertex 'state' '<b>State</b><br>$$s_{e,t}$$' ($node + 'fillColor=#FFFFFF;strokeColor=#3C6F9A;fontColor=#1F2A33;') '772' '136' '168' '72'
Add-Edge 'a-task-obs' 'task' 'obs' ($edge + 'strokeColor=#3C6F9A;')
Add-Edge 'a-obs-parser' 'obs' 'parser' ($edge + 'strokeColor=#3C6F9A;')
Add-Edge 'a-parser-state' 'parser' 'state' ($edge + 'strokeColor=#3C6F9A;')

# 2. Factorized external world model.
Add-Vertex 'b2' '' ($outer + 'fillColor=#F3F9F7;strokeColor=#2C7A6B;') '80' '252' '980' '292'
Add-Vertex 's2' '2' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;strokeWidth=0;fontFamily=Times New Roman;fontStyle=1;fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;fillColor=#2C7A6B;' '96' '264' '26' '26'
Add-Vertex 't2' '<font color="#2C7A6B"><b>External world model</b></font>' ($text + 'fontSize=16;align=left;') '130' '262' '320' '30'
Add-Vertex 'e2' '$$\mathcal{M}^{t}=(\mathcal{W}^{t},\mathcal{H}^{t},\mathcal{P}^{t})$$' ($text + 'fontSize=13;align=right;fontColor=#2C7A6B;') '520' '262' '520' '30'
Add-Vertex 'wm' '<b><font color="#2C7A6B">Working memory</font></b><br>$$\mathcal{W}^{t}$$<br><font color="#5C6B76">episode-local trace<br>reset at episode start</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2C7A6B;') '108' '308' '292' '132'
Add-Vertex 'hkg' '<b><font color="#2C7A6B">Historical KG</font></b><br>$$\mathcal{H}^{t}$$<br><font color="#5C6B76">typed facts and transitions<br>confidence / provenance</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2C7A6B;') '424' '308' '292' '132'
Add-Vertex 'plib' '<b><font color="#2C7A6B">Pattern library</font></b><br>$$p=(\sigma_p,A_p,C_p,\Delta_p,\mathcal{E}_p)$$<br><font color="#5C6B76">executable procedure + evidence</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2C7A6B;') '740' '308' '292' '132'
Add-Vertex 'read' '$$m_{e,t}=\mathrm{Read}(\mathcal{W},\mathcal{H},\mathcal{P}\mid s_{e,t},g_e)$$' ($node + 'fillColor=#E6F4F0;strokeColor=#2C7A6B;fontSize=15;') '108' '456' '924' '52'
Add-Edge 'a-wm-read' 'wm' 'read' ($edge + 'strokeColor=#2C7A6B;')
Add-Edge 'a-hkg-read' 'hkg' 'read' ($edge + 'strokeColor=#2C7A6B;')
Add-Edge 'a-plib-read' 'plib' 'read' ($edge + 'strokeColor=#2C7A6B;')
Add-Edge 'a-state-b2' 'state' 'b2' ($edge + 'strokeColor=#3C6F9A;')

# 3. Reuse quality and selective cognition.
Add-Vertex 'b3' '' ($outer + 'fillColor=#F7F8F9;strokeColor=#4A5560;') '80' '570' '980' '258'
Add-Vertex 's3' '3' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;strokeWidth=0;fontFamily=Times New Roman;fontStyle=1;fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;fillColor=#4A5560;' '96' '582' '26' '26'
Add-Vertex 't3' '<font color="#4A5560"><b>State-conditioned reuse and selective cognition</b></font>' ($text + 'fontSize=16;align=left;') '130' '580' '720' '30'
Add-Vertex 'qreuse' '$$Q_{\mathrm{reuse}}=w_cC(p)+w_sQ_{\mathrm{state}}+w_vQ_{\mathrm{valid}}+w_f(1-Q_{\mathrm{failure}})$$' ($node + 'fillColor=#FFFFFF;strokeColor=#4A5560;fontSize=14;') '108' '620' '924' '48'
Add-Vertex 'direct' '<b><font color="#3A7A4C">Direct reuse</font></b><br>$$Q_{\mathrm{reuse}}\geq\tau_{\mathrm{high}}$$<br><font color="#5C6B76">fast executor<br>0 model calls</font>' ($node + 'fillColor=#E8F4EB;strokeColor=#3A7A4C;') '108' '682' '292' '126'
Add-Vertex 'verified' '<b><font color="#B67A22">Verified reuse</font></b><br>$$\tau_{\mathrm{mid}}\leq Q<\tau_{\mathrm{high}}$$<br><font color="#5C6B76">4B verifier<br>then execute</font>' ($node + 'fillColor=#F8F1DF;strokeColor=#B67A22;') '424' '682' '292' '126'
Add-Vertex 'full' '<b><font color="#3C6F9A">Full cognition</font></b><br>$$Q_{\mathrm{reuse}}<\tau_{\mathrm{mid}}$$<br><font color="#5C6B76">9B planner / actor / refiner<br>parameters frozen</font>' ($node + 'fillColor=#E8F1F8;strokeColor=#3C6F9A;') '740' '682' '292' '126'
Add-Edge 'a-read-q' 'read' 'qreuse' ($edge + 'strokeColor=#2C7A6B;')
Add-Edge 'a-q-direct' 'qreuse' 'direct' ($edge + 'strokeColor=#3A7A4C;')
Add-Edge 'a-q-verified' 'qreuse' 'verified' ($edge + 'strokeColor=#B67A22;')
Add-Edge 'a-q-full' 'qreuse' 'full' ($edge + 'strokeColor=#3C6F9A;')

# 4. Grounding and execution.
Add-Vertex 'b4' '' ($outer + 'fillColor=#FBF6F0;strokeColor=#C56A20;') '80' '852' '980' '98'
Add-Vertex 's4' '4' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;strokeWidth=0;fontFamily=Times New Roman;fontStyle=1;fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;fillColor=#C56A20;' '96' '864' '26' '26'
Add-Vertex 't4' '<font color="#C56A20"><b>Grounding and execution</b></font>' ($text + 'fontSize=16;align=left;') '130' '862' '360' '30'
Add-Vertex 'ground' '$$\tilde{a}_{e,t}=\mathrm{Ground}(a_p,\mathcal{A}(o_{e,t}))$$<br><font color="#5C6B76">admissible action only</font>' ($node + 'fillColor=#F8EEE4;strokeColor=#C56A20;fontSize=14;') '108' '894' '430' '42'
Add-Vertex 'env' '$$a_{e,t}\;\rightarrow\;\mathrm{Env}\;\rightarrow\;(o_{e,t+1},r_{e,t})$$' ($node + 'fillColor=#F8EEE4;strokeColor=#C56A20;fontSize=14;') '602' '894' '430' '42'
Add-Edge 'a-direct-ground' 'direct' 'ground' ($edge + 'strokeColor=#C56A20;')
Add-Edge 'a-verified-ground' 'verified' 'ground' ($edge + 'strokeColor=#C56A20;')
Add-Edge 'a-full-ground' 'full' 'ground' ($edge + 'strokeColor=#C56A20;')
Add-Edge 'a-ground-env' 'ground' 'env' ($edge + 'strokeColor=#C56A20;')

# 5. Outcome-aware evolution and procedural write-back.
Add-Vertex 'b5' '' ($outer + 'fillColor=#FBF5F5;strokeColor=#B04A4A;') '80' '982' '980' '90'
Add-Vertex 's5' '5' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;strokeWidth=0;fontFamily=Times New Roman;fontStyle=1;fontSize=14;fontColor=#FFFFFF;align=center;verticalAlign=middle;fillColor=#B04A4A;' '96' '994' '26' '26'
Add-Vertex 't5' '<font color="#B04A4A"><b>Continual evolution</b></font>' ($text + 'fontSize=16;align=left;') '130' '992' '260' '30'
Add-Vertex 'e5' '$$\mathcal{M}^{t+1}=\mathcal{F}(\mathcal{M}^{t},s_{e,t},a_{e,t},o_{e,t+1})$$' ($text + 'fontSize=13;align=right;fontColor=#B04A4A;') '430' '992' '610' '30'
Add-Vertex 'audit' '<font color="#5C6B76">audit expected vs. observed transition<br>reinforce, consolidate, revise, or retire</font>' ($text + 'fontSize=13;') '108' '1028' '924' '28'
Add-Vertex 'wb-label' '<b><font color="#2C7A6B">write-back $\Delta\mathcal{P}_t$</font></b><br><font color="#5C6B76">procedural memory update</font>' ($text + 'fontSize=11;') '730' '470' '280' '42'
Add-Edge 'a-b4-b5' 'b4' 'b5' ($edge + 'strokeColor=#B04A4A;')
Add-Edge 'a-writeback' 'b5' 'plib' ($edge + 'strokeColor=#2C7A6B;dashed=1;dashPattern=8 6;exitX=0;exitY=0.5;entryX=0.5;entryY=1;')
Add-Edge 'a-hkg-optional' 'b5' 'hkg' ($edge + 'strokeColor=#7B8794;dashed=1;dashPattern=4 4;exitX=0.2;exitY=0.5;entryX=0.5;entryY=1;') 'optional $\\Delta\\mathcal{H}$'
Add-Vertex 'footer' '<font color="#5C6B76">episode boundary: $\mathcal{W}_e$ resets; $\mathcal{H}$ and $\mathcal{P}$ persist</font>' ($text + 'fontSize=11;align=left;') '80' '1072' '980' '24'

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Indent = $true
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$settings.OmitXmlDeclaration = $true
$writer = [System.Xml.XmlWriter]::Create($Path, $settings)
try { $doc.Save($writer) } finally { $writer.Dispose() }

Write-Output "Updated: $Path"
Write-Output "Backup:  $BackupPath"
