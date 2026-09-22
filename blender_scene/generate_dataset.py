"""
Synthetischer Datensatz: 3-l-Farbeimer (vereinfacht) mit aufgeklebtem Etikett.

Aufruf:
    blender --background --python blender_scene/generate_dataset.py -- --n 20

Optionen (nach dem "--"):
    --n N         Anzahl Eimer (Default 20)
    --out PFAD    Ausgabeordner (Default <projekt>/data)
    --seed S      Zufalls-Seed, für reproduzierbare Datensätze
    --samples K   Cycles-Samples pro Bild (Default 64)
    --res R       Bildauflösung in Pixeln, quadratisch (Default 1024)

Ausgabe:
    <out>/images/<id>_cam<k>.png
    <out>/labels.jsonl   eine Zeile pro Eimer mit den wahren Werten
    <out>/meta.json      feste Szenenparameter (Maße, Kameras) für die Auswertung

Geometrie-Konvention:
    Eimerachse = Z, Boden bei z = 0. Im Eimer-Koordinatensystem liegt die
    Etiketten-Vorderseite bei phi = 0 (Richtung +X), die Naht bei phi = -180°.
    Schiefe: z(phi) = z_soll + offset + r * tan(alpha) * phi   (phi in rad)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy  # vor bmesh importieren (nötig, wenn als pip-Modul genutzt)
import bmesh
import numpy as np

# ==========================================================================
# KONFIGURATION
# ==========================================================================

# Maße GESCHÄTZT aus Fotoproportionen (Höhe/Durchmesser ~1,2) und 3 l Nennvolumen.
# Bitte durch die Werte aus dem Gebinde-Datenblatt ersetzen.
CAN_DIAMETER_MM = 155.0
CAN_HEIGHT_MM = 185.0
BEAD_RADIUS_MM = 2.5            # Rollsicke oben und unten
LID_RECESS_MM = 3.0             # Deckel liegt so tief unter der Oberkante

# Etikett
LABEL_ARC_DEG = 360.0           # 360 = Rundumetikett, kleiner = Teiletikett
LABEL_OVERLAP_MM = 8.0          # Überlappung an der Naht (nur Rundumetikett)
LABEL_BOTTOM_MM = 8.0           # Soll-Unterkante über dem Boden
LABEL_TOP_MM = 177.0            # Soll-Oberkante über dem Boden
LABEL_THICKNESS_MM = 0.1
LABEL_TEXTURE_PATH = None       # z.B. "assets/etikett.png": echte, flache Druckvorlage
LABEL_TEXTURE_WIDTH_PX = 4096   # nur für die generierte Platzhalter-Textur

# Henkel (Bügel)
HANDLE_ENABLED = True
HANDLE_EAR_BELOW_TOP_MM = 25.0
HANDLE_WIRE_RADIUS_MM = 2.5
HANDLE_CLEARANCE_MM = 6.0       # Bügelradius = Eimerradius + Abstand
HANDLE_DOWN_PROB = 0.7          # Anteil Bilder mit heruntergeklapptem Bügel

# Kameras und Licht
N_CAMERAS = 3
CAMERA_DISTANCE_MM = 450.0      # Abstand Kamera zur Eimerachse
CAMERA_FOCAL_MM = 12.0
CAMERA_SENSOR_MM = 6.4          # Sensorbreite (~1/2"-Sensor), an echte Kamera anpassen
LIGHT_ENERGY_W = 1.2            # je Kamera ein Flächenlicht (W); ~80 W/m² wären massiv überbelichtet
LIGHT_JITTER = 0.3              # +-30 % Helligkeitsvariation

# Zufallsvariation pro Eimer
ALPHA_RANGE_DEG = (-2.5, 2.5)   # Schiefe fehlerhafter Eimer
GOOD_ALPHA_DEG = 0.3            # "gute" Eimer: |alpha| <= diesem Wert
OFFSET_RANGE_MM = (-3.0, 3.0)   # Höhenversatz
GOOD_FRACTION = 0.5

# ==========================================================================


def mm(v):
    """Millimeter -> Blender-Einheiten (Meter)."""
    return v / 1000.0


def parse_args():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    elif "blender" not in Path(sys.argv[0]).name.lower():
        argv = sys.argv[1:]          # als reines Python-Modul (pip install bpy) gestartet
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--no-masks", action="store_true", help="keine Etikettenmasken rendern")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Szene
# --------------------------------------------------------------------------

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights, bpy.data.curves, bpy.data.images):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)


def new_material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def link_to(obj, parent, mat=None):
    obj.parent = parent
    if mat is not None:
        obj.data.materials.append(mat)


def set_smooth(obj, only_sides=False):
    """Glatte Schattierung ohne bpy.ops (funktioniert auch im Background-Modus)."""
    for poly in obj.data.polygons:
        poly.use_smooth = (abs(poly.normal.z) < 0.5) if only_sides else True


def create_bucket(assembly):
    r = mm(CAN_DIAMETER_MM) / 2
    h = mm(CAN_HEIGHT_MM)
    bead = mm(BEAD_RADIUS_MM)
    metal = new_material("Tinplate", (0.78, 0.79, 0.80), metallic=1.0, roughness=0.3)

    # Mantel
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=r, depth=h, location=(0, 0, h / 2))
    body = bpy.context.active_object
    body.name = "Body"
    set_smooth(body, only_sides=True)
    link_to(body, assembly, metal)

    # Rollsicken oben und unten
    for name, z in (("BeadTop", h), ("BeadBottom", bead)):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=bead,
                                         major_segments=128, minor_segments=16,
                                         location=(0, 0, z))
        t = bpy.context.active_object
        t.name = name
        set_smooth(t)
        link_to(t, assembly, metal)

    # Deckel (leicht vertieft) mit innerem Rand
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=r - mm(4), depth=mm(1),
                                        location=(0, 0, h - mm(LID_RECESS_MM)))
    lid = bpy.context.active_object
    lid.name = "Lid"
    set_smooth(lid, only_sides=True)
    link_to(lid, assembly, metal)

    bpy.ops.mesh.primitive_torus_add(major_radius=r - mm(5), minor_radius=mm(1.5),
                                     major_segments=128, minor_segments=12,
                                     location=(0, 0, h - mm(1)))
    rim = bpy.context.active_object
    rim.name = "LidRim"
    set_smooth(rim)
    link_to(rim, assembly, metal)

    return r, h, metal


def create_handle(assembly, r, h, metal):
    """Ohren + Drahtbügel. Gibt das Pivot-Objekt zurück (Drehung um lokale Y-Achse)."""
    z_ear = h - mm(HANDLE_EAR_BELOW_TOP_MM)
    wire = mm(HANDLE_WIRE_RADIUS_MM)
    rb = r + mm(HANDLE_CLEARANCE_MM)

    for sign in (1, -1):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=mm(7), depth=mm(5),
                                            location=(0, sign * (r + mm(2.5)), z_ear),
                                            rotation=(math.pi / 2, 0, 0))
        ear = bpy.context.active_object
        ear.name = "Ear"
        set_smooth(ear, only_sides=False)
        link_to(ear, assembly, metal)

    pivot = bpy.data.objects.new("HandlePivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = (0, 0, z_ear)
    pivot.parent = assembly

    # Bügel als Halbkreis in der YZ-Ebene (aufrecht), Drahtdicke über Bevel
    curve = bpy.data.curves.new("HandleCurve", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = wire
    curve.bevel_resolution = 4
    spline = curve.splines.new("POLY")
    n = 64
    spline.points.add(n - 1)
    for i in range(n):
        s = math.pi * i / (n - 1)
        spline.points[i].co = (0.0, rb * math.cos(s), rb * math.sin(s), 1.0)
    bail = bpy.data.objects.new("Handle", curve)
    bpy.context.collection.objects.link(bail)
    bail.parent = pivot
    curve.materials.append(metal)

    # Maximaler Klappwinkel, bis der Bügel am Mantel anliegt: rb_innen * sin(theta) >= r
    theta_max = math.degrees(math.pi - math.asin(r / (rb - wire)))
    return pivot, theta_max


# --------------------------------------------------------------------------
# Etikett
# --------------------------------------------------------------------------

def label_geometry():
    """Bogenlänge und Startwinkel des Etiketts im Eimer-Koordinatensystem."""
    r = mm(CAN_DIAMETER_MM) / 2
    if LABEL_ARC_DEG >= 360:
        arc = 2 * math.pi + mm(LABEL_OVERLAP_MM) / r
        phi_start = -math.pi
    else:
        arc = math.radians(LABEL_ARC_DEG)
        phi_start = -arc / 2
    return r, arc, phi_start


def make_placeholder_texture():
    """
    Markenneutrale Platzhalter-Textur im Farblayout der Vorlage:
    greige Grundfläche, weißes Band oben, dunkle Textblöcke, Icon-Reihe, Barcode-Streifen.
    Koordinaten in mm relativ zur Etiketten-Vorderseite (x) und zur Unterkante (z).
    """
    r, arc, phi_start = label_geometry()
    length_mm = arc * r * 1000
    height_mm = LABEL_TOP_MM - LABEL_BOTTOM_MM
    W = LABEL_TEXTURE_WIDTH_PX
    ppm = W / length_mm
    H = int(round(height_mm * ppm))
    front_px = (0 - phi_start) * r * 1000 * ppm

    img = np.empty((H, W, 3), dtype=np.float32)
    img[:] = (0.72, 0.69, 0.62)                        # greige

    def rect(x0, x1, z0, z1, color):
        c0 = int(np.clip(front_px + x0 * ppm, 0, W)); c1 = int(np.clip(front_px + x1 * ppm, 0, W))
        r0 = int(np.clip(z0 * ppm, 0, H)); r1 = int(np.clip(z1 * ppm, 0, H))
        img[r0:r1, c0:c1] = color

    rng = random.Random(1)                              # Layout immer gleich
    dark, mid, light = (0.12, 0.12, 0.12), (0.30, 0.30, 0.30), (0.86, 0.84, 0.79)

    # weißes Band + helle Linie darunter
    rect(-1e4, 1e4, 110, 150, (0.97, 0.97, 0.96))
    rect(-1e4, 1e4, 106, 108, light)
    # "Schriftzug" im weißen Band: Balken unterschiedlicher Breite
    x = -60
    while x < 55:
        w = rng.uniform(6, 14)
        rect(x, x + w, 118, 142, dark)
        x += w + 3
    rect(-82, -68, 116, 144, mid)                       # Platzhalter-Emblem
    # runde Plakette rechts
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cz, cr = front_px + 82 * ppm, 102 * ppm, 16 * ppm
    img[(xx - cx) ** 2 + (yy - cz) ** 2 < cr ** 2] = light
    # Produktbezeichnung
    rect(-55, -15, 92, 100, light)
    rect(-55, 30, 76, 88, dark)
    rect(-55, -25, 62, 74, dark)
    # Kleintext-Zeilen
    for i, z in enumerate((52, 48, 42, 38)):
        rect(-55, 55 - 25 * (i % 2), z, z + 2, mid)
    # Icon-Reihe
    for i in range(7):
        x0 = -55 + 17 * i
        rect(x0, x0 + 14, 10, 24, (0.60, 0.58, 0.52))
        rect(x0 + 1, x0 + 13, 11, 23, (0.80, 0.78, 0.72))
    rect(62, 75, 12, 24, dark)                          # Volumenangabe
    # Barcode-Streifen auf der linken Seite (90° neben der Front)
    bx = -r * 1000 * math.pi / 2
    rect(bx - 4, bx + 4, 40, 100, (0.98, 0.98, 0.98))
    for z in np.arange(44, 96, 1.6):
        rect(bx - 3, bx + 3, z, z + rng.choice((0.4, 0.8)), dark)

    image = bpy.data.images.new("LabelTexture", width=W, height=H, alpha=False)
    rgba = np.concatenate([img, np.ones((H, W, 1), dtype=np.float32)], axis=2)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def create_label_material():
    if LABEL_TEXTURE_PATH:
        image = bpy.data.images.load(str(Path(LABEL_TEXTURE_PATH).resolve()))
    else:
        image = make_placeholder_texture()
    mat = new_material("LabelPaper", (1, 1, 1), metallic=0.0, roughness=0.55)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"
    links.new(tex.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def create_label(assembly, material, alpha_deg, offset_mm, n_u=512):
    """
    Etikett als eigene Fläche mit UV-Abwicklung.
    Bei Rundumetikett: Anfang und Ende überlappen an der Naht, das Ende liegt
    minimal weiter außen (Radius-Rampe), damit es sauber über dem Anfang liegt.
    """
    r, arc, phi_start = label_geometry()
    tan_a = math.tan(math.radians(alpha_deg))
    z0 = mm(LABEL_BOTTOM_MM + offset_mm)
    z1 = mm(LABEL_TOP_MM + offset_mm)
    ramp = mm(3 * LABEL_THICKNESS_MM) if LABEL_ARC_DEG >= 360 else 0.0

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    cols = []
    for i in range(n_u + 1):
        u = i / n_u
        phi = phi_start + u * arc
        rad = r + mm(0.05) + ramp * u
        dz = r * tan_a * phi
        x, y = rad * math.cos(phi), rad * math.sin(phi)
        cols.append((bm.verts.new((x, y, z0 + dz)), bm.verts.new((x, y, z1 + dz)), u))

    for (a0, a1, ua), (b0, b1, ub) in zip(cols[:-1], cols[1:]):
        f = bm.faces.new((a0, b0, b1, a1))
        f.smooth = True
        for loop, uv in zip(f.loops, ((ua, 0), (ub, 0), (ub, 1), (ua, 1))):
            loop[uv_layer].uv = uv

    mesh = bpy.data.meshes.new("Label")
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    label = bpy.data.objects.new("Label", mesh)
    bpy.context.collection.objects.link(label)
    label.parent = assembly
    mesh.materials.append(material)
    sol = label.modifiers.new("Thickness", "SOLIDIFY")
    sol.thickness = mm(LABEL_THICKNESS_MM)
    sol.offset = 1.0                                     # nach außen, nicht in den Eimer
    return label


def remove_label():
    old = bpy.data.objects.get("Label")
    if old:
        mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        bpy.data.meshes.remove(mesh)


# --------------------------------------------------------------------------
# Kameras, Licht, Rendering
# --------------------------------------------------------------------------

def create_cameras_and_lights(target_z):
    target = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, target_z)

    cams, lights, cam_meta = [], [], []
    d = mm(CAMERA_DISTANCE_MM)
    for i in range(N_CAMERAS):
        az = 360.0 / N_CAMERAS * i
        x, y = d * math.cos(math.radians(az)), d * math.sin(math.radians(az))

        cam_data = bpy.data.cameras.new(f"Cam{i}")
        cam_data.lens = CAMERA_FOCAL_MM
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = CAMERA_SENSOR_MM
        cam = bpy.data.objects.new(f"Cam{i}", cam_data)
        bpy.context.collection.objects.link(cam)
        cam.location = (x, y, target_z)
        c = cam.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
        cams.append(cam)

        light_data = bpy.data.lights.new(f"Light{i}", type="AREA")
        light_data.size = 0.25
        light = bpy.data.objects.new(f"Light{i}", light_data)
        bpy.context.collection.objects.link(light)
        light.location = (x * 0.95, y * 0.95, target_z + 0.12)
        c = light.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
        lights.append(light)

        cam_meta.append({"name": f"cam{i}", "azimuth_deg": az,
                         "position_mm": [x * 1000, y * 1000, target_z * 1000]})
    return cams, lights, cam_meta


def setup_world_and_ground():
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    bg.inputs["Strength"].default_value = 0.08       # sonst spiegelt das Metall nur Schwarz

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Conveyor"
    ground.data.materials.append(new_material("ConveyorMat", (0.08, 0.08, 0.08), roughness=0.8))


def try_enable_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        return "CPU"
    for dev in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = dev
            prefs.get_devices()
            if any(d.type == dev for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                bpy.context.scene.cycles.device = "GPU"
                return dev
        except Exception:
            continue
    return "CPU"


def setup_render(samples, res):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"   # vorhersagbare Farben statt Filmlook
    return try_enable_gpu()


# --------------------------------------------------------------------------
# Etikettenmasken (Trainingslabels für das Segmentierungsnetz)
# --------------------------------------------------------------------------

def emission_material(name, value):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (value, value, value, 1.0)
    em.inputs["Strength"].default_value = 1.0
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def render_masks(scene, cams, mask_dir, sid, white, black):
    """
    Zweiter, billiger Render-Durchgang je Kamera: Etikett = weiß, alles andere
    (Eimer, Sicken, Henkel, Band) = schwarz. Verdeckung durch den Henkel ist damit
    automatisch korrekt. Danach wird alles wiederhergestellt.
    """
    saved_mats = []
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE"} and obj.data is not None and len(obj.data.materials):
            saved_mats.append((obj, list(obj.data.materials)))
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = white if obj.name == "Label" else black

    bg = scene.world.node_tree.nodes["Background"]
    saved = (bg.inputs["Strength"].default_value, scene.cycles.samples,
             scene.cycles.use_denoising, scene.render.image_settings.color_mode)
    bg.inputs["Strength"].default_value = 0.0
    scene.cycles.samples = 8              # Emission braucht kaum Samples, nur Kanten-Antialiasing
    scene.cycles.use_denoising = False    # Denoiser würde die Maskenkante verschmieren
    scene.render.image_settings.color_mode = "BW"

    names = []
    for k, cam in enumerate(cams):
        scene.camera = cam
        name = f"{sid:04d}_cam{k}.png"
        scene.render.filepath = str(mask_dir / name)
        bpy.ops.render.render(write_still=True)
        names.append(name)

    for obj, mats in saved_mats:
        for i, m in enumerate(mats):
            obj.data.materials[i] = m
    (bg.inputs["Strength"].default_value, scene.cycles.samples,
     scene.cycles.use_denoising, scene.render.image_settings.color_mode) = saved
    return names


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "masks"
    if not args.no_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    assembly = bpy.data.objects.new("Bucket", None)
    bpy.context.collection.objects.link(assembly)

    r, h, metal = create_bucket(assembly)
    pivot, theta_max = (create_handle(assembly, r, h, metal) if HANDLE_ENABLED else (None, 0.0))
    label_mat = create_label_material()
    setup_world_and_ground()
    target_z = mm(LABEL_BOTTOM_MM + LABEL_TOP_MM) / 2
    cams, lights, cam_meta = create_cameras_and_lights(target_z)
    device = setup_render(args.samples, args.res)
    print(f"Render-Gerät: {device}")
    mask_white = emission_material("MaskWhite", 1.0)
    mask_black = emission_material("MaskBlack", 0.0)

    meta = {
        "can_diameter_mm": CAN_DIAMETER_MM, "can_height_mm": CAN_HEIGHT_MM,
        "label_arc_deg": LABEL_ARC_DEG, "label_overlap_mm": LABEL_OVERLAP_MM,
        "label_bottom_mm": LABEL_BOTTOM_MM, "label_top_mm": LABEL_TOP_MM,
        "camera_focal_mm": CAMERA_FOCAL_MM, "camera_sensor_width_mm": CAMERA_SENSOR_MM,
        "resolution_px": args.res, "cameras": cam_meta,
        "convention": "z(phi) = z_soll + offset + r*tan(alpha)*phi, phi=0 an der Etikettenfront, "
                      "Welt-Azimut der Front = can_rot_deg",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    scene = bpy.context.scene
    with open(out / "labels.jsonl", "w") as f:
        for sid in range(args.n):
            good = random.random() < GOOD_FRACTION
            alpha = (random.uniform(-GOOD_ALPHA_DEG, GOOD_ALPHA_DEG) if good
                     else random.choice((-1, 1)) * random.uniform(GOOD_ALPHA_DEG, ALPHA_RANGE_DEG[1]))
            offset = random.uniform(*OFFSET_RANGE_MM)
            can_rot = random.uniform(0, 360)

            remove_label()
            create_label(assembly, label_mat, alpha, offset)
            assembly.rotation_euler = (0, 0, math.radians(can_rot))

            handle_deg = None
            if pivot is not None:
                if random.random() < HANDLE_DOWN_PROB:
                    handle_deg = random.choice((-1, 1)) * (theta_max - 0.5)
                else:
                    handle_deg = random.uniform(-theta_max + 0.5, theta_max - 0.5)
                pivot.rotation_euler = (0, math.radians(handle_deg), 0)

            light_scale = random.uniform(1 - LIGHT_JITTER, 1 + LIGHT_JITTER)
            for light in lights:
                light.data.energy = LIGHT_ENERGY_W * light_scale

            images = []
            for k, cam in enumerate(cams):
                scene.camera = cam
                name = f"{sid:04d}_cam{k}.png"
                scene.render.filepath = str(img_dir / name)
                bpy.ops.render.render(write_still=True)
                images.append(name)

            masks = [] if args.no_masks else render_masks(scene, cams, mask_dir, sid, mask_white, mask_black)

            rec = {
                "id": sid, "good": good,
                "alpha_deg": round(alpha, 4), "offset_mm": round(offset, 4),
                "can_rot_deg": round(can_rot, 3),
                "seam_world_deg": round((can_rot - 180) % 360, 3),
                "handle_deg": None if handle_deg is None else round(handle_deg, 2),
                "light_scale": round(light_scale, 3),
                "images": images,
                "masks": masks,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{sid + 1}/{args.n}] alpha={alpha:+.2f}° offset={offset:+.2f} mm "
                  f"rot={can_rot:.0f}° henkel={handle_deg}")

    print(f"Fertig: {args.n} Eimer -> {img_dir}")


if __name__ == "__main__":
    main()