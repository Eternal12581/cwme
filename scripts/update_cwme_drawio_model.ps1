param(
    [Parameter(Mandatory = $true)] [string]$Path,
    [Parameter(Mandatory = $true)] [string]$BackupPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Copy-Item -LiteralPath $Path -Destination $BackupPath -Force
[xml]$doc = Get-Content -Raw -LiteralPath $Path
$mxfile = $doc.DocumentElement

$oldPage = $mxfile.SelectSingleNode('./diagram[@id="cwme-iclr-overview"]')
if ($null -ne $oldPage) { $mxfile.RemoveChild($oldPage) | Out-Null }

$page = $doc.CreateElement('diagram')
$page.SetAttribute('id', 'cwme-iclr-overview')
$page.SetAttribute('name', 'CWME Architecture (ICLR)')
$model = $doc.CreateElement('mxGraphModel')
foreach ($pair in @{
    dx='1855'; dy='1243'; grid='0'; gridSize='10'; guides='1'; tooltips='1'; connect='1'; arrows='1'; fold='1'; page='1'; pageScale='1'; pageWidth='1600'; pageHeight='1000'; math='1'; shadow='0'
}.GetEnumerator()) { $model.SetAttribute($pair.Key, $pair.Value) }
$root = $doc.CreateElement('root')
$zero = $doc.CreateElement('mxCell'); $zero.SetAttribute('id','0'); [void]$root.AppendChild($zero)
$layer = $doc.CreateElement('mxCell'); $layer.SetAttribute('id','1'); $layer.SetAttribute('parent','0'); [void]$root.AppendChild($layer)
[void]$model.AppendChild($root); [void]$page.AppendChild($model)
if ($mxfile.FirstChild) { [void]$mxfile.InsertBefore($page, $mxfile.FirstChild) } else { [void]$mxfile.AppendChild($page) }

function Add-Geometry($cell, [string]$X, [string]$Y, [string]$W, [string]$H) {
    $g = $doc.CreateElement('mxGeometry')
    $g.SetAttribute('x',$X); $g.SetAttribute('y',$Y); $g.SetAttribute('width',$W); $g.SetAttribute('height',$H); $g.SetAttribute('as','geometry')
    [void]$cell.AppendChild($g)
}

function Add-Vertex([string]$Id, [string]$Value, [string]$Style, [string]$X, [string]$Y, [string]$W, [string]$H) {
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id',$Id); $cell.SetAttribute('value',$Value); $cell.SetAttribute('style',$Style); $cell.SetAttribute('parent','1'); $cell.SetAttribute('vertex','1')
    Add-Geometry $cell $X $Y $W $H; [void]$root.AppendChild($cell)
}

function Add-Edge([string]$Id, [string]$Source, [string]$Target, [string]$Style, [string]$Value = '') {
    $cell = $doc.CreateElement('mxCell')
    $cell.SetAttribute('id',$Id); $cell.SetAttribute('value',$Value); $cell.SetAttribute('style',$Style); $cell.SetAttribute('parent','1'); $cell.SetAttribute('source',$Source); $cell.SetAttribute('target',$Target); $cell.SetAttribute('edge','1')
    $g = $doc.CreateElement('mxGeometry'); $g.SetAttribute('relative','1'); $g.SetAttribute('as','geometry'); [void]$cell.AppendChild($g); [void]$root.AppendChild($cell)
}

$box = 'rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeWidth=2;fontFamily=Helvetica;align=left;verticalAlign=top;spacingLeft=14;spacingTop=8;'
$node = 'rounded=1;whiteSpace=wrap;html=1;arcSize=14;strokeWidth=1.5;fontFamily=Helvetica;fontSize=14;align=center;verticalAlign=middle;'
$text = 'text;html=1;fontFamily=Helvetica;align=center;verticalAlign=middle;'
$arrow = 'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;endSize=6;strokeWidth=1.5;fontFamily=Helvetica;'
$dashed = $arrow + 'dashed=1;dashPattern=8 6;'

Add-Vertex 'title' '<b><font style="font-size:24px">CWME: Continual World-Model Evolution</font></b><br><font color="#5C6B76" style="font-size:14px">External evidence-bearing world model for a frozen language agent</font>' ($text + 'align=left;') '80' '24' '900' '56'

# Left: perception is a model input, not a sequential stage.
Add-Vertex 'perception-panel' '' ($box + 'fillColor=#E6F2FB;strokeColor=#4C78A8;') '100' '140' '330' '480'
Add-Vertex 'perception-title' '<b><font color="#2F5A84" style="font-size:20px">Perception</font></b>' $text '130' '160' '270' '36'
Add-Vertex 'task' '<b>Task</b><br>$$g_e$$' ($node + 'fillColor=#FFFFFF;strokeColor=#4C78A8;') '135' '230' '120' '72'
Add-Vertex 'observation' '<b>Observation</b><br>$$o_{e,t}$$' ($node + 'fillColor=#FFFFFF;strokeColor=#4C78A8;') '275' '230' '120' '72'
Add-Vertex 'parser' '<b>State parser</b><br>$$\psi$$' ($node + 'fillColor=#BBDEFB;strokeColor=#4C78A8;') '195' '350' '140' '78'
Add-Vertex 'state' '<b>Structured state</b><br>$$s_{e,t}=\psi(g_e,o_{e,t})$$' ($node + 'fillColor=#FFFFFF;strokeColor=#4C78A8;') '145' '480' '240' '76'
Add-Edge 'task-parser' 'task' 'parser' ($arrow + 'strokeColor=#4C78A8;')
Add-Edge 'obs-parser' 'observation' 'parser' ($arrow + 'strokeColor=#4C78A8;')
Add-Edge 'parser-state' 'parser' 'state' ($arrow + 'strokeColor=#4C78A8;')

# Center: the CWME model itself, with the original nested-memory visual grammar.
Add-Vertex 'world-panel' '' ($box + 'fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2.5;') '470' '120' '700' '530'
Add-Vertex 'world-title' '<b><font color="#1B5E20" style="font-size:20px">CWME External World Model</font></b><br><font color="#1B5E20" style="font-size:15px">$\mathcal{M}^{t}=(\mathcal{W}^{t},\mathcal{H}^{t},\mathcal{P}^{t})$</font>' $text '570' '142' '500' '62'
Add-Vertex 'wm' '<b><font color="#1B5E20" style="font-size:16px">Working Memory</font></b><br>$$\mathcal{W}^{t}$$<br><font color="#5C6B76">episode-local<br>active trace + progress</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2E7D32;') '510' '230' '190' '140'
Add-Vertex 'hkg' '<b><font color="#1B5E20" style="font-size:16px">Historical KG</font></b><br>$$\mathcal{H}^{t}$$<br><font color="#5C6B76">typed facts<br>state transitions<br>confidence</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2E7D32;') '725' '230' '190' '140'
Add-Vertex 'patterns' '<b><font color="#1B5E20" style="font-size:16px">Pattern Library</font></b><br>$$p=(\sigma_p,A_p,C_p,\Delta_p,\mathcal{E}_p)$$<br><font color="#5C6B76">procedures + outcome evidence</font>' ($node + 'fillColor=#FFFFFF;strokeColor=#2E7D32;') '940' '230' '190' '140'
Add-Vertex 'hub' '<b><font color="#1B5E20" style="font-size:18px">Memory Hub</font></b><br><font color="#1B5E20" style="font-size:14px">bounded indexing &amp; retrieval</font>' 'ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#66BB6A;strokeColor=#1B5E20;strokeWidth=2;fontFamily=Helvetica;fontSize=16;fontColor=#1B5E20;align=center;verticalAlign=middle;' '705' '405' '230' '86'
Add-Vertex 'read' '$$m_{e,t}=\mathrm{Read}(\mathcal{W},\mathcal{H},\mathcal{P}\mid s_{e,t},g_e)$$' ($node + 'fillColor=#C8E6C9;strokeColor=#2E7D32;fontSize=14;') '535' '535' '570' '58'
Add-Edge 'state-world' 'state' 'world-panel' ($arrow + 'strokeColor=#4C78A8;')
Add-Edge 'wm-hub' 'wm' 'hub' ($arrow + 'strokeColor=#2E7D32;')
Add-Edge 'hkg-hub' 'hkg' 'hub' ($arrow + 'strokeColor=#2E7D32;')
Add-Edge 'patterns-hub' 'patterns' 'hub' ($arrow + 'strokeColor=#2E7D32;')
Add-Edge 'hub-read' 'hub' 'read' ($arrow + 'strokeColor=#2E7D32;')

# Right: cognition/control is a model head with selective routes.
Add-Vertex 'control-panel' '' ($box + 'fillColor=#E6F2FB;strokeColor=#4C78A8;') '1210' '140' '300' '480'
Add-Vertex 'control-title' '<b><font color="#2F5A84" style="font-size:20px">Cognition &amp; Control</font></b>' $text '1230' '160' '260' '36'
Add-Vertex 'router' '<b>Reuse quality</b><br>$$Q_{reuse}=f(C(p),Q_{state},Q_{valid},Q_{failure})$$' ($node + 'fillColor=#FFFFFF;strokeColor=#2E7D32;fontSize=13;') '1240' '220' '240' '74'
Add-Vertex 'direct' '<b><font color="#2E7D32">Direct reuse</font></b><br>$$Q\geq\tau_{high}$$<br><font color="#5C6B76">fast executor<br>0 LM calls</font>' ($node + 'fillColor=#E8F5E9;strokeColor=#2E7D32;fontSize=12;') '1230' '325' '260' '70'
Add-Vertex 'verified' '<b><font color="#B67A22">Verified reuse</font></b><br>$$\tau_{mid}\leq Q<\tau_{high}$$<br><font color="#5C6B76">4B verifier</font>' ($node + 'fillColor=#FFF8E1;strokeColor=#B67A22;fontSize=12;') '1230' '415' '260' '70'
Add-Vertex 'full' '<b><font color="#3C6F9A">Full cognition</font></b><br>$$Q<\tau_{mid}$$<br><font color="#5C6B76">frozen 9B planner / actor / refiner</font>' ($node + 'fillColor=#E8F1F8;strokeColor=#3C6F9A;fontSize=12;') '1230' '505' '260' '70'
Add-Edge 'read-router' 'read' 'router' ($arrow + 'strokeColor=#2E7D32;')
Add-Edge 'router-direct' 'router' 'direct' ($arrow + 'strokeColor=#2E7D32;')
Add-Edge 'router-verified' 'router' 'verified' ($arrow + 'strokeColor=#B67A22;')
Add-Edge 'router-full' 'router' 'full' ($arrow + 'strokeColor=#3C6F9A;')

# Bottom: environment is external to the model; observed outcomes drive the outer loop.
Add-Vertex 'grounding' '<b><font color="#E65100">Grounding</font></b><br>map candidate action to valid affordance' ($node + 'fillColor=#FFE0B2;strokeColor=#EF6C00;fontSize=13;') '1230' '635' '260' '60'
Add-Vertex 'environment' '<b>Environment</b><br>$$a_{e,t}\rightarrow(o_{e,t+1},r_{e,t})$$' ($node + 'fillColor=#F5F5F5;strokeColor=#666666;fontSize=13;') '1230' '730' '260' '74'
Add-Edge 'direct-ground' 'direct' 'grounding' ($arrow + 'strokeColor=#EF6C00;')
Add-Edge 'verified-ground' 'verified' 'grounding' ($arrow + 'strokeColor=#EF6C00;')
Add-Edge 'full-ground' 'full' 'grounding' ($arrow + 'strokeColor=#EF6C00;')
Add-Edge 'ground-env' 'grounding' 'environment' ($arrow + 'strokeColor=#EF6C00;')

Add-Vertex 'update-panel' '' ($box + 'fillColor=#FFEBEE;strokeColor=#C62828;') '100' '720' '1040' '135'
Add-Vertex 'update-title' '<b><font color="#C62828" style="font-size:20px">Continual Evolution across Episodes</font></b>' $text '130' '742' '440' '34'
Add-Vertex 'update-eq' '$$\mathcal{M}^{t+1}=\mathcal{F}(\mathcal{M}^{t},s_{e,t},a_{e,t},o_{e,t+1})$$' ($text + 'fontSize=14;fontColor=#C62828;align=right;') '570' '742' '530' '34'
Add-Vertex 'audit' '<font color="#5C6B76">Outcome audit: expected $\Delta_p$ vs. observed transition<br>consolidate supported patterns · revise contradicted patterns · EpisodeGuard suppresses repetition</font>' ($text + 'fontSize=13;') '145' '790' '950' '48'
Add-Edge 'env-update' 'environment' 'update-panel' ($dashed + 'strokeColor=#C62828;')
Add-Edge 'update-patterns' 'update-panel' 'patterns' ($dashed + 'strokeColor=#2E7D32;')
Add-Edge 'update-hkg' 'update-panel' 'hkg' ($dashed + 'strokeColor=#7B8794;')
Add-Vertex 'feedback' '<font color="#5C6B76">environment feedback</font>' ($text + 'fontSize=12;') '1000' '665' '220' '26'
Add-Edge 'env-feedback' 'environment' 'observation' ($dashed + 'strokeColor=#7B8794;')

Add-Vertex 'legend' '<font color="#5C6B76" style="font-size:11px">solid = within-step control &nbsp; · &nbsp; dashed = feedback / continual update &nbsp; · &nbsp; green = world model &nbsp; · &nbsp; blue = frozen LM &nbsp; · &nbsp; orange = grounding</font>' ($text + 'align=left;') '100' '900' '1410' '30'

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Indent = $true
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$settings.OmitXmlDeclaration = $true
$writer = [System.Xml.XmlWriter]::Create($Path, $settings)
try { $doc.Save($writer) } finally { $writer.Dispose() }
Write-Output "Updated model-style page: $Path"
Write-Output "Backup: $BackupPath"
