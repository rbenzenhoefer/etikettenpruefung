# Etikettenprüfung: Schief geklebte Etiketten auf Farbeimern erkennen

Prototyp für eine kamerabasierte Inline-Prüfung, die schief oder falsch aufgeklebte Rundumetiketten auf 3-l-Farbeimern erkennt. Entwickelt und getestet vollständig in einer 3D-Simulation, noch ohne echte Kameras.

Dieses Dokument hat zwei Teile: [Teil 1](#teil-1--überblick-ohne-technik) erklärt das Projekt ohne Vorkenntnisse, [Teil 2](#teil-2--technische-dokumentation) beschreibt Mathematik, Code und Ergebnisse im Detail.

---

# Teil 1 – Überblick ohne Technik

## Das Problem

Auf der Produktionslinie wird jedem Eimer ein Etikett rundherum aufgeklebt. Manchmal sitzt es schief oder zu hoch bzw. zu tief. Solche Eimer sollen automatisch erkannt und vom Band genommen werden, bevor sie beim Kunden landen.

## Die Idee in einem Satz

Drei Kameras fotografieren jeden Eimer von drei Seiten, ein Programm findet darauf die obere und untere Kante des Etiketts und misst, ob diese Kanten waagerecht verlaufen.

## Wie das funktioniert

Man kann sich das vorstellen wie mit einem Klebestreifen, den man um eine Konservendose wickelt. Klebt man ihn gerade, liegt seine Kante überall auf derselben Höhe. Klebt man ihn schief, steigt die Kante einmal rund um die Dose gleichmäßig an, wie ein Gewinde. An der Stelle, wo Anfang und Ende des Streifens aufeinandertreffen, entsteht dann eine kleine Stufe. Genau dieses Ansteigen misst das Programm.

Der Ablauf hat zwei Schritte:

**1. Eine KI zeichnet den Umriss des Etiketts.** Auf einem Foto ist gar nicht so leicht zu sagen, wo genau das Etikett endet. Metallränder glänzen, der Henkel liegt manchmal davor, und der Aufdruck hat selbst viele Kanten. Ein neuronales Netz hat gelernt, auf jedem Bild pixelgenau zu markieren: „Das hier ist Etikett, das nicht."

**2. Klassische Mathematik misst die Schieflage.** Aus dem markierten Umriss nimmt das Programm die obere und untere Kante. Weil bekannt ist, wo die Kameras stehen, lässt sich jeder Bildpunkt in eine echte Position auf dem Eimer umrechnen, in Millimetern. Daraus ergibt sich, um wie viel Grad das Etikett schief sitzt. Liegt der Wert über der Toleranz (derzeit 0,5°), wird der Eimer ausgeschleust.

Diese Aufteilung ist bewusst gewählt. Die KI löst den Teil, der für feste Regeln zu unübersichtlich ist (Was ist Etikett, was Glanz, was Henkel?). Die Messung selbst bleibt nachvollziehbare Geometrie: Man kann auf jedem Bild nachsehen, welche Kante gemessen wurde.

## Die virtuelle Fabrik

Echte Fehlerbilder in großer Zahl zu sammeln ist aufwendig. Deshalb wurde zuerst eine virtuelle Version der Prüfstation gebaut, im 3D-Programm Blender: ein Eimer mit Henkel, ein Etikett, drei Kameras und Beleuchtung. Ein Programm erzeugt daraus automatisch Hunderte Eimer mit zufällig schiefen oder verschobenen Etiketten, zufälliger Drehung und Henkelstellung. Der große Vorteil: Für jedes Bild ist exakt bekannt, wie schief das Etikett wirklich ist. So lässt sich genau prüfen, wie gut das System misst.

## Ergebnis bisher

Auf 40 Test-Eimern, die das System vorher nie gesehen hat, wich die gemessene Schieflage im Durchschnitt um 0,013° von der Wahrheit ab, im schlechtesten Fall um 0,05°. Zum Vergleich: Die Toleranz liegt bei 0,5°, also etwa zehnmal höher als der größte Messfehler. Alle 40 Eimer wurden korrekt eingeordnet, bis auf einen Grenzfall, der nur 0,0015° über der Toleranz lag.

## Was noch fehlt, bevor das in der Produktion läuft

Das Ergebnis zeigt, dass das Verfahren funktioniert, aber nur in der Simulation. Echte Fotos haben andere Spiegelungen, Staub und Druckschwankungen, und echte Kameras stehen nie millimetergenau am geplanten Ort. Die nächsten Schritte sind deshalb: testen, wie genau die Kameras ausgerichtet sein müssen, und das System mit echten Fotos von der Linie nachtrainieren und prüfen.

---

# Teil 2 – Technische Dokumentation

## Architektur

```
Blender-Simulation ──> RGB-Bilder + Etikettenmasken + Ground Truth (α, δ, Rotation)
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        U-Net-Training (Masken)      Geometrie-Check mit GT-Masken
                  │                     (Obergrenze der Genauigkeit)
                  ▼
   Inferenz: Bild ──> U-Net ──> Maske ──> Kantenpunkte ──> Rückprojektion
             ──> lokaler RANSAC-Fit je Kamera und Kante ──> Median ──> α ──> Entscheidung
```

## Projektstruktur

```
3D_dosen_prototyp/
├── blender_scene/
│   └── generate_dataset.py     # Szene, Zufallsvariation, Rendering, Masken, Labels
├── analysis/
│   ├── measure_skew.py         # Kameramodell, Geometrie, Fit (Gradienten-Variante, Basismodul)
│   ├── measure_from_mask.py    # Hybrid-Messung aus Masken (GT oder Netz)
│   ├── seg_model.py            # U-Net: Aufbau, Laden, Vorhersage
│   └── train_seg.py            # Training
├── run_pipeline.sh             # Rendern -> Geometrie-Check -> Training -> Testmessung
├── data/  data_test/           # generiert: images/, masks/, labels.jsonl, meta.json
├── models/label_seg.pt         # generiert: bestes Modell
└── logs/                       # generiert
```

## Installation und Ausführung

Voraussetzungen: macOS mit Apple Silicon (getestet auf MacBook Air M4), Blender 5.2, Python 3.12, [uv](https://github.com/astral-sh/uv).

```bash
uv python pin 3.12
uv sync
uv add torch segmentation-models-pytorch opencv-python scipy scikit-learn
source .venv/bin/activate
```

Komplette Pipeline:

```bash
chmod +x run_pipeline.sh
caffeinate -i ./run_pipeline.sh
```

Einzelschritte:

```bash
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER --background --python blender_scene/generate_dataset.py -- --n 200 --seed 1 --out data
$BLENDER --background --python blender_scene/generate_dataset.py -- --n 40  --seed 2 --out data_test
python analysis/measure_from_mask.py --data data_test --source gt
python analysis/train_seg.py --data data --epochs 30
python analysis/measure_from_mask.py --data data_test --source model --debug
```

Blender läuft mit seinem eigenen Python-Interpreter. Die venv wird nur für Analyse und Training benötigt.

## 1. Simulation (`generate_dataset.py`)

### Szene

Alle Maße in mm, Eimerachse = $z$, Boden bei $z = 0$.

| Größe | Wert | Anmerkung |
|---|---|---|
| Eimer $\varnothing \times h$ | 155 × 185 | **geschätzt** aus Fotoproportionen ($h/d \approx 1{,}2$) und 3 l Nennvolumen; durch Datenblattwerte ersetzen |
| Radius $R$ | 77,5 | |
| Etikett Soll-Unterkante / -Oberkante | 8 / 177 | |
| Überlappung an der Naht | 8 | Ende liegt über dem Anfang (Radius-Rampe 0,3 mm) |
| Rollsicken | Torus, $r = 2{,}5$ | oben bei $z = h$, unten bei $z = 2{,}5$ |
| Kameras | 3 × im Azimut 0°, 120°, 240° | Abstand $d = 450$ zur Achse, Höhe 92,5 |
| Optik | $F = 12$ mm, Sensorbreite 6,4 mm | ~1/2"-Sensor, 1024 × 1024 px |
| Licht | je Kamera ein Flächenlicht 1,2 W | Energie pro Eimer × U(0,7; 1,3) |

Die Etikettentextur ist ein markenneutraler Platzhalter im Layout der Vorlage. Eine echte, flach abgewickelte Druckvorlage lässt sich über `LABEL_TEXTURE_PATH` einbinden.

### Etikettenmodell

Ein schief aufgewickeltes Rechteck-Etikett bildet eine Schraubenlinie. Mit dem Umfangswinkel $\varphi$ im Eimer-Koordinatensystem (Front bei $\varphi = 0$, Naht bei $\varphi = -\pi$) gilt für jede Etikettenkante

$$z(\varphi) = z_\text{soll} + \delta + R \tan\alpha \cdot \varphi, \qquad \varphi \in [-\pi,\ -\pi + \Phi]$$

mit Schiefewinkel $\alpha$, Höhenversatz $\delta$ und Bogenlänge $\Phi = 2\pi + 8\,\text{mm}/R$. Das ist genau die Gerade $z = z_0 + s \tan\alpha$ in der Abwicklung mit Bogenlänge $s = R\varphi$, also ein geometrisch exaktes Modell eines schief aufgewickelten Etiketts. An der Naht entsteht der Höhensprung

$$\Delta z_\text{Naht} = 2\pi R \tan\alpha, \qquad \text{z. B. } \alpha = 0{,}87^\circ \Rightarrow \Delta z \approx 7{,}4\ \text{mm}.$$

Das Etikett wird als eigenes Mesh mit UV-Abwicklung (512 Segmente) und Solidify-Modifier (0,1 mm) erzeugt, nicht als Textur auf dem Eimer. Dadurch hat es echte geometrische Kanten und eine physikalisch korrekte Naht.

### Zufallsvariation pro Eimer

| Parameter | Verteilung |
|---|---|
| $\alpha$ | 50 %: $U(-0{,}3°, 0{,}3°)$ („gut"); 50 %: $\pm U(0{,}3°, 2{,}5°)$ |
| $\delta$ | $U(-3, 3)$ mm |
| Eimerrotation | $U(0°, 360°)$ |
| Henkel | 70 % am Mantel anliegend ($\pm\theta_\max$), 30 % $U(-\theta_\max, \theta_\max)$ |
| Lichtstärke | $\times\, U(0{,}7; 1{,}3)$ |

Der maximale Klappwinkel des Henkels folgt aus der Bedingung, dass der Scheitel des Bügels (Radius $R_b = R + 6$ mm, Drahtradius $r_w = 2{,}5$ mm) den Mantel gerade berührt:

$$(R_b - r_w)\sin\theta \ge R \;\Rightarrow\; \theta_\max = 180^\circ - \arcsin\frac{R}{R_b - r_w} \approx 106{,}9^\circ.$$

### Masken

Pro Kamera folgt auf das RGB-Rendering ein zweiter Durchgang: Alle Materialien werden durch Emissions-Shader ersetzt (Etikett = 1, alles andere = 0), Welthintergrund aus, 8 Samples, Denoiser aus, Ausgabe als 8-bit-Graustufen-PNG. Verdeckung durch den Henkel und die Naht sind dadurch automatisch pixelgenau korrekt, die Kanten sind durch das Pixelfilter antialiasiert. Anschließend werden alle Materialien wiederhergestellt.

### Ausgabe

`labels.jsonl` enthält pro Eimer `alpha_deg`, `offset_mm`, `can_rot_deg`, `seam_world_deg`, `handle_deg`, `light_scale` sowie die Datei­namen von Bildern und Masken. `meta.json` enthält Maße, Kamerapositionen und Optik, die die Messung benötigt.

## 2. Geometrie (`measure_skew.py`, `measure_from_mask.py`)

### Kameramodell

Für Kamera $k$ an Position $\mathbf{C}_k = (d\cos\theta_k,\ d\sin\theta_k,\ z_c)$ mit waagerechter Blickrichtung auf die Achse:

$$\mathbf{f} = \frac{-(C_x, C_y, 0)}{\lVert(C_x, C_y)\rVert},\qquad \mathbf{r} = \frac{\mathbf{f}\times\mathbf{e}_z}{\lVert\mathbf{f}\times\mathbf{e}_z\rVert},\qquad \mathbf{u} = \mathbf{r}\times\mathbf{f}.$$

Lochkamera mit Skalierung $s = \dfrac{F}{w/2}\cdot\dfrac{N}{2}$ (Brennweite $F$, Sensorbreite $w$, Auflösung $N$). Projektion eines Weltpunkts $\mathbf{P}$ mit $\mathbf{V} = \mathbf{P} - \mathbf{C}$:

$$x = s\,\frac{\mathbf{V}\cdot\mathbf{r}}{\mathbf{V}\cdot\mathbf{f}} + \frac{N}{2}, \qquad y = -s\,\frac{\mathbf{V}\cdot\mathbf{u}}{\mathbf{V}\cdot\mathbf{f}} + \frac{N}{2}.$$

Bildkoordinaten sind kontinuierlich, Pixel $i$ überdeckt $[i, i+1)$, die Pixelmitte liegt also bei $i + 0{,}5$.

### Rückprojektion: Strahl-Zylinder-Schnitt

Sichtstrahl durch Bildpunkt $(x, y)$:

$$\mathbf{D} \propto \frac{x - N/2}{s}\,\mathbf{r} - \frac{y - N/2}{s}\,\mathbf{u} + \mathbf{f}.$$

Schnitt von $\mathbf{C} + t\mathbf{D}$ mit dem Zylinder $x^2 + y^2 = R^2$:

$$\underbrace{(D_x^2 + D_y^2)}_{a}\,t^2 + \underbrace{2(C_xD_x + C_yD_y)}_{b}\,t + \underbrace{C_x^2 + C_y^2 - R^2}_{c} = 0.$$

Die kleinere positive Wurzel ist der der Kamera zugewandte Punkt. Daraus folgen $\varphi = \operatorname{atan2}(P_y, P_x)$ und die Höhe $z = P_z$.

### Sichtbereich

Von einer Kamera im Abstand $d$ ist der Zylinder bis zum Tangentenpunkt sichtbar, also im Bogen $\pm\arccos(R/d) = \pm 80{,}1^\circ$ um die Blickrichtung. Ausgewertet werden nur $\pm 50^\circ$ (`MAX_VIEW_DEG`), weil die Fläche zum Rand hin immer schräger zur Kamera steht und die Lokalisierung ungenauer wird. Drei Kameras decken so $3 \times 100^\circ = 300^\circ$ des Umfangs ab.

### Soll-Kurve und Kantensuche

Für jede Kante wird der Soll-Kreis $\{(R\cos\varphi, R\sin\varphi, z_\text{soll})\}$ ins Bild projiziert. Das ergibt je Bildspalte eine erwartete Zeile $y_\text{exp}(x)$. Gesucht wird in einem Fenster von $\pm 25$ mm (`SEARCH_MM`), umgerechnet über den lokalen vertikalen Maßstab in Pixel/mm an der Blickrichtung.

In der Wahrscheinlichkeitskarte $p$ der Maske wird je Spalte der Übergang über die Schwelle 0,5 gesucht (oben: außen → Etikett, unten: Etikett → außen). Bei mehreren Übergängen, etwa durch den Henkel, wird der der Soll-Kurve nächste gewählt. Die Position wird subpixelgenau interpoliert:

$$y_\text{Kante} = y_0 + i + 0{,}5 + t, \qquad t = \frac{0{,}5 - p_i}{p_{i+1} - p_i}.$$

### Lokaler Fit

Um den Sprung von $\operatorname{atan2}$ bei $\pm 180^\circ$ zu vermeiden und ohne die (in der Realität unbekannte) Eimerrotation auszukommen, wird der Winkel relativ zur Kamera gemessen: $\varphi_\text{rel} = \operatorname{wrap}(\varphi - \theta_k) \in [-\pi, \pi)$. Da die Steigung der Schraubenlinie überall gleich ist, genügt je Kamera $k$ und Kante $e$ ein lokaler Geradenfit

$$z = a_{k,e} + b_{k,e}\,\varphi_\text{rel}, \qquad \hat\alpha_{k,e} = \arctan\frac{b_{k,e}}{R}.$$

Der Achsenabschnitt $a_{k,e}$ ist die Kantenhöhe in Blickrichtung. Der Fit läuft mit RANSAC (Residuenschwelle 1 mm). Verworfen wird ein Fit bei weniger als 30 Punkten oder unter 50 % Inliern. Liegt die Naht im Bild, zerfällt die Kante in zwei parallele Geradenstücke gleicher Steigung; RANSAC nimmt das größere, die Steigung bleibt korrekt.

Das Endergebnis ist der Median über alle gültigen Einzelfits (maximal 3 Kameras × 2 Kanten = 6). Ausgeschleust wird bei $|\hat\alpha| > 0{,}5^\circ$ (`ALPHA_TOL_DEG`).

### Warum Masken statt Bildgradienten

Die erste Version suchte die Kante direkt über den vertikalen Sobel-Gradienten. Das scheitert systematisch bei großer Schiefe: Beim Rundumetikett summiert sich die Abweichung über den Umfang auf $R\tan\alpha\cdot\varphi$, bei $\alpha = 2{,}4^\circ$ und $\varphi = \pm 180^\circ$ auf bis zu $\pm 10$ mm. Die Rollsicke liegt aber nur 8 mm neben der Soll-Kante und hat einen stärkeren Kontrast. Die Suche fand dann die Sicke statt der Etikettenkante, mit Fehlern von bis zu 3°. In der Maske gehören Sicke, Deckel und Aufdruck nie zum Etikett, die Verwechslung ist damit ausgeschlossen, und das Suchfenster darf auf $\pm 25$ mm wachsen.

## 3. Segmentierungsnetz (`seg_model.py`, `train_seg.py`)

| Komponente | Wahl |
|---|---|
| Architektur | U-Net, Encoder ResNet18 (ImageNet-vortrainiert), `segmentation_models_pytorch` |
| Ausgabe | 1 Kanal, Logit „Etikett" |
| Loss | $\mathcal{L} = \text{BCE} + \big(1 - \frac{2\sum p\,y + 1}{\sum p + \sum y + 1}\big)$ |
| Optimierer | AdamW, lr $3\cdot10^{-4}$, weight decay $10^{-4}$, Cosine-Schedule |
| Training | 30 Epochen, Batch 8, Crops 384 × 384 in voller Auflösung |
| Aufteilung | 80/20 **nach Eimer**, nicht nach Bild |
| Metrik | IoU auf ganzen Validierungsbildern |
| Gerät | MPS (Apple-GPU), sonst CUDA/CPU |

Zwei Designentscheidungen sind für die Messgenauigkeit wichtig:

- **Kantenbetonte Crops in voller Auflösung.** 70 % der Ausschnitte werden um einen zufälligen Punkt der Maskenkante gelegt. Die Messung hängt fast nur von der Kantenlage ab, und Herunterskalieren würde genau diese Präzision kosten. Inferenz läuft fully-convolutional auf dem ganzen Bild (auf Vielfache von 32 gepolstert).
- **Aufteilung nach Eimer.** Sonst sähe das Netz im Validierungsset denselben Eimer aus einer anderen Kamera, und die Werte wären geschönt.

Augmentierung: Helligkeit/Kontrast $\times U(0{,}7; 1{,}3) + U(-20, 20)$, Farbstich je Kanal $\pm 7\,\%$, Gaußunschärfe ($p = 0{,}3$, $\sigma \in [0{,}3; 1{,}2]$), Sensorrauschen $\sigma \le 6$, horizontales Spiegeln.

## 4. Ergebnisse

Datensatz: 200 Eimer Training (600 Bilder, davon 160 Eimer Training / 40 Validierung), 40 unabhängige Test-Eimer (120 Bilder, anderer Seed).

| Messung auf 40 Test-Eimern | mittl. Betragsfehler $\alpha$ | Median | max | Fehlentscheidungen |
|---|---|---|---|---|
| Geometrie + Blender-Masken (Obergrenze) | 0,010° | 0,011° | 0,023° | 0 FP, 1 FN* |
| Geometrie + U-Net-Masken | 0,013° | 0,009° | 0,049° | 0 FP, 1 FN* |
| Geometrie + Sobel-Gradient (frühe Version, 9 Eimer) | bis 3° | – | – | – |

\* Grenzfall: $\alpha_\text{wahr} = -0{,}5015^\circ$, gemessen $-0{,}48^\circ$, also 0,0015° über der Toleranz.

Training: Val-IoU 0,9824 nach Epoche 1, 0,9985 ab Epoche 22, danach Plateau. Laufzeiten auf MacBook Air M4: Rendern ≈ 22 s pro Eimer (3 RGB-Bilder bei 64 Samples + 3 Masken), Training ≈ 40 s pro Epoche.

## 5. Grenzen und bekannte Probleme

**Nur Simulation.** Trainings- und Testdaten stammen aus derselben Verteilung (gleiche Textur, gleiches Lichtmodell, exakt bekannte Kameras). Die Ergebnisse belegen die Funktionsfähigkeit der Pipeline, nicht die Genauigkeit an der echten Linie. Auf realen Bildern ist mit geringerer IoU zu rechnen.

**Kamerakalibrierung.** Die Geometrie verwendet die exakten Kamerapositionen aus `meta.json`. Real muss jede Kamera intrinsisch und extrinsisch kalibriert werden (z. B. ChArUco-Board, `cv2.calibrateCamera`, `cv2.solvePnP`). Die nötige Genauigkeit ist noch nicht bestimmt.

**Toleranzgrenze.** Ohne Sicherheitsabstand rutschen Eimer knapp über der Toleranz durch. Bei einem maximalen Messfehler von ≈ 0,05° empfiehlt sich eine Ausschleuseschwelle von etwa 0,45°.

**Referenz der Kantenhöhe.** Die Soll-Höhe in der Auswertung (`true_edge_height`) ist in der Nähe der Naht nicht korrekt berechnet (Ausreißer bis ≈ 11 mm, identisch mit GT- und Netz-Masken). Betrifft nur die Auswertung der Höhenlage, nicht die Schiefemessung.

**Geschätzte Maße, Platzhalter-Etikett.** Eimermaße und Etikettenposition sind aus einem Produktfoto geschätzt, die Textur ist ein Platzhalter.

## 6. Nächste Schritte

1. **Kalibrierungs-Empfindlichkeit in der Simulation:** Kameras beim Rendern um bekannte Beträge verschieben und verdrehen, ohne `meta.json` anzupassen, und den Anstieg des α-Fehlers messen. Ergibt eine konkrete Anforderung an die spätere Kalibrierung.
2. **Echte Maße und Druckvorlage** einsetzen.
3. **Mehr Domain Randomization:** Texturvarianten, Hintergrund, Beleuchtungsrichtung, Kamerarauschen und Bewegungsunschärfe.
4. **Sim-to-Real:** echte Fotos aufnehmen, einige Dutzend von Hand maskieren, Netz nachtrainieren und auf echten Eimern mit bekannter Schieflage validieren.
5. **Sicherheitsabstand** in die Entscheidungslogik aufnehmen, Höhenreferenz an der Naht korrigieren.
