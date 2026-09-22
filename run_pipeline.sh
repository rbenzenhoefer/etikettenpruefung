#!/usr/bin/env bash
# Komplette Pipeline: Rendern -> Geometrie-Check -> Training -> Messung mit dem Netz
# Aufruf aus dem Projektordner:   caffeinate -i ./run_pipeline.sh
set -euo pipefail

# Aliase gelten in Skripten nicht, deshalb der volle Blender-Pfad
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
PY=".venv/bin/python"
GEN="blender_scene/generate_dataset.py"

N_TRAIN=200      # 200 Eimer x 3 Kameras = 600 Bilder (+ 600 Masken)
N_TEST=40        #  40 Eimer x 3 Kameras = 120 Bilder (+ 120 Masken)
EPOCHS=30

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M)
step() { echo; echo "=== $(date +%H:%M:%S)  $1 ==="; }

check_masks() {   # bricht ab, wenn Masken fehlen oder nicht zu den Bildern passen
  local imgs masks
  imgs=$(ls "$1/images" | wc -l | tr -d ' ')
  masks=$(ls "$1/masks" 2>/dev/null | wc -l | tr -d ' ')
  echo "$1: $imgs Bilder, $masks Masken"
  if [ "$imgs" -eq 0 ] || [ "$imgs" -ne "$masks" ]; then
    echo "FEHLER: Masken fehlen oder Anzahl passt nicht - Abbruch."; exit 1
  fi
}

step "Vorab-Check"
[ -x "$BLENDER" ] || { echo "Blender nicht gefunden: $BLENDER"; exit 1; }
[ -x "$PY" ]      || { echo "venv nicht gefunden: $PY"; exit 1; }
grep -q "render_masks" "$GEN" || { echo "$GEN ist die alte Version ohne Masken"; exit 1; }
rm -rf data data_test

step "1a  Trainingsdaten rendern ($N_TRAIN Eimer)"
"$BLENDER" --background --python "$GEN" -- --n "$N_TRAIN" --seed 1 --out data \
  > "logs/render_train_$STAMP.log" 2>&1
check_masks data

step "1b  Testdaten rendern ($N_TEST Eimer)"
"$BLENDER" --background --python "$GEN" -- --n "$N_TEST" --seed 2 --out data_test \
  > "logs/render_test_$STAMP.log" 2>&1
check_masks data_test

step "2   Geometrie-Check mit Blender-Masken"
"$PY" analysis/measure_from_mask.py --data data_test --source gt | tee "logs/geometrie_gt_$STAMP.log"
grep -q "Quelle: gt" "logs/geometrie_gt_$STAMP.log" || { echo "Geometrie-Check ohne Messwerte - Abbruch."; exit 1; }

step "3   Training ($EPOCHS Epochen)"
"$PY" analysis/train_seg.py --data data --epochs "$EPOCHS" | tee "logs/training_$STAMP.log"

step "4   Messung mit dem trainierten Netz auf den Testdaten"
"$PY" analysis/measure_from_mask.py --data data_test --source model \
  --model models/label_seg.pt --debug | tee "logs/geometrie_model_$STAMP.log"

step "Fertig. Logs in logs/, Overlays in data_test/debug_model/"