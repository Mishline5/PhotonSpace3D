# Rapport de conception — Moteur 3D from scratch

## 0. Décisions préalables

### 0.1 OpenGL 4.1, pas 4.3

La stack imposée demandait un contexte OpenGL 4.3 Core (pour les compute shaders). La machine de développement est un Mac (Apple Silicon, macOS). Or **Apple a gelé son implémentation OpenGL à la version 4.1 Core depuis la dépréciation de l'API en 2018** (macOS Mojave) : aucune machine Apple, quelle que soit sa puissance (M1 comme M5), ne peut créer un contexte 4.3, et `glfw.create_window` échoue purement et simplement (retourne `NULL`) si on le lui demande — ce n'est pas une fonctionnalité manquante, c'est un échec de création de fenêtre.

Décision (validée avant implémentation) : **cibler OpenGL 4.1 Core / GLSL 410 partout, sur toutes les plateformes**, et remplacer le ray tracing par compute shader par une technique équivalente en fragment shader (voir §5). Le moteur est prévu pour tourner aussi bien sur Nvidia/Windows (l'usage principal visé, y compris sur une RTX 4070 Ti) que sur AMD ou Apple Silicon — cibler 4.1 uniformément plutôt que d'avoir un chemin "compute shader" séparé pour les machines qui le permettraient garde un seul chemin de rendu à maintenir et à vérifier, alors qu'un second chemin compute-shader n'aurait pas pu être testé du tout depuis cette machine de développement. Voir §7 (limites connues) pour la discussion complète de ce compromis.

### 0.2 Architecture en composants séparés : `engine/` maintenant, `editing/` plus tard

Le moteur est réorganisé en un package Python **auto-suffisant** (`engine/`) : tout ce dont il a besoin (modules Python, shaders GLSL) vit à l'intérieur du dossier, sans dépendance vers l'extérieur (vérifié : aucun import dans `engine/` ne référence quoi que ce soit hors du package). `main.py` et `benchmark.py`, à la racine du projet, sont des **exemples d'usage** du moteur, pas le moteur lui-même — un futur outil "Editing" (à la Blender : gizmos de déplacement/rotation, panneau de matériaux, etc.) viendrait se brancher de la même façon, comme un second consommateur du même package `engine/`, sans avoir à dupliquer ou déplacer quoi que ce soit.

## 1. Architecture

```
engine/                    package auto-suffisant, réutilisable tel quel
  window.py                  fenêtre GLFW + contexte GL + état clavier/souris
  clock.py                    delta-time + limiteur de framerate
  transform.py                 position/rotation/échelle + compteur de version GPU
  mesh.py                      classe Mesh (VBO/IBO immuables)
  primitives.py                 générateurs cube/plan/sphère/cylindre/cône
  camera.py                     caméra libre (déplacement + regard, delta-time)
  scene.py                       Scene / SceneObject / Material / RTPrimitive / lumières
  shader.py                       chargement/compilation des programmes GLSL
  gl_math.py                       conversions glm -> bytes explicites
  capabilities.py                  détection matérielle (fabricant/tier GPU)
  settings.py                       QualitySettings (niveaux 0-3) + presets par tier
  settings_ui.py                     panneau TAB (Dear ImGui)
  frame_ubo.py                        UBO par-frame (caméra, lumières, matrice d'ombre, toggles)
  rt_primitives.py                     UBO des primitives analytiques pour le RT hybride
  render_targets.py                     cible HDR (couleur linéaire + profondeur)
  prepass.py                             prepass profondeur + normales (pour la SSAO)
  shadow_pass.py                          shadow map directionnelle, résolution adaptative
  ssao_pass.py                             SSAO à noyau hémisphérique + flou
  volumetric_pass.py                        rayons volumétriques directionnels (god rays)
  fullscreen.py                              triangle plein écran sans VBO (post-process)
  gpu_timing.py                               mesure GPU non bloquante (ring buffer)
  capture.py                                   capture d'écran pour le développement/QA
  renderer.py                                  orchestration des passes, render() par frame
  shaders/                    shaders GLSL, à l'intérieur du package (portable)
    forward.vert/frag            PBR + ombres + SSAO + RT hybride
    shadow.vert/frag, prepass.vert/frag, ssao.frag, ssao_blur.frag,
    volumetric.frag, tonemap.frag, fullscreen.vert
main.py                     démo interactive (exemple d'usage du moteur)
benchmark.py                mesures objectives avant/après chaque fonctionnalité
```

Chaque module porte en tête un docstring expliquant son rôle (cf. code source).

## 2. Adaptabilité matérielle

### 2.1 Détection (`engine/capabilities.py`)

Au lancement, `detect(ctx)` lit `GL_VENDOR`/`GL_RENDERER`/`GL_VERSION` (les seules informations standard et portables qu'expose un driver OpenGL — il n'existe pas d'API GL native pour demander "quel est le niveau de performance de ce GPU") et classe la carte en un tier `low` / `medium` / `high` par reconnaissance de sous-chaînes :
- **NVIDIA** : RTX/Titan → `high` ; GTX et autres → `medium`.
- **AMD** : RX 6000/7000/9000 (RDNA2+) → `high` ; autres → `medium`.
- **Apple Silicon** (M1 à M5 et au-delà, détecté génériquement par `"apple"` dans le vendor/renderer, sans lister chaque génération) → `medium` : le GPU est très capable, mais le plafond OpenGL 4.1 de la plateforme (§0.1) signifie qu'aucun chemin compute-shader ne sera jamais disponible ici, quelle que soit la puce.
- **Intel intégré** → `low`.
- Fabricant inconnu → `medium` (repli prudent plutôt qu'un extrême).

### 2.2 Niveaux de qualité adaptatifs (`engine/settings.py`)

Chaque effet majeur (ombres, SSAO, RT hybride, lumière volumétrique) est un **niveau unique 0-3** plutôt qu'un simple booléen — 0 signifie désactivé, pas besoin d'un interrupteur séparé. `settings.py` est le seul endroit qui traduit un niveau en paramètres concrets :

| Niveau | Ombres (résolution / PCF) | SSAO (échantillons) | RT (rayons de réflexion) | Volumétrique (pas de marche) |
|---|---|---|---|---|
| 0 | désactivé | désactivé | désactivé | désactivé |
| 1 | 512² / 3×3 | 8 | 1 (net) | 12 |
| 2 | 1024² / 3×3 | 16 | 2 | 24 |
| 3 | 2048² / 5×5 | 24 | 4 (doux) | 48 |

`preset_for_tier()` choisit un jeu de niveaux par défaut selon le tier détecté (`high` → tout à 3, `low` → tout à 1 avec le volumétrique désactivé, `medium`/inconnu → tout à 2) — l'utilisateur peut ensuite tout modifier librement depuis le panneau TAB (§3), ce preset n'est qu'un point de départ raisonnable, jamais une limite imposée.

### 2.3 Ce que ça donne concrètement

Sur cette machine (Apple M1 Max, tier `medium`), les niveaux par défaut sont 2 partout (sauf volumétrique à 1). Sur la RTX 4070 Ti mentionnée, le tier serait détecté `high` et le moteur démarrerait directement à qualité maximale (ombres 2048², SSAO 24 échantillons, RT à 4 rayons doux, volumétrique à 48 pas) — sans configuration manuelle. Le §6 montre que même au niveau 3 partout, cette scène reste largement au-dessus de 60 FPS sur du matériel bien plus modeste que la carte visée.

## 3. Panneau de réglages (TAB)

`engine/settings_ui.py` construit une interface Dear ImGui : un slider 0-3 par effet (avec un libellé Off/Low/Medium/High affiché en direct), un slider d'exposition, une case V-Sync, un slider de plafond FPS, un bouton "revenir au preset détecté", et l'identité du GPU détecté (fabricant, tier, avertissement spécifique si Apple Silicon). Toutes les anciennes touches de debug (1/2/3 pour ombres/SSAO/RT dans une version antérieure de ce projet) ont été retirées : le panneau TAB est désormais le seul point de contrôle de ces réglages.

**Choix de bibliothèque** : Dear ImGui via `pyimgui` (paquet PyPI `imgui`), pas `imgui_bundle`. `imgui_bundle` embarque sa propre copie native de GLFW, chargée *en plus* du paquet `glfw` déjà utilisé pour la fenêtre — sur macOS, cela déclenche un avertissement du runtime Objective-C ("Class X is implemented in both ... this may cause spurious casting failures and mysterious crashes"), un vrai risque de fiabilité, pas juste un message cosmétique. Testé, confirmé, puis évité en utilisant `pyimgui`, qui n'embarque aucune dépendance native propre.

**Intégration** : le rendu ImGui n'attache pas ses propres callbacks GLFW (`attach_callbacks=False`) — `engine.window.Window`/`Input` restent la seule source de vérité pour l'entrée clavier/souris, et `settings_ui.py` se contente de relire cet état chaque frame pour nourrir ImGui. Ce choix garde le panneau réutilisable tel quel par le futur outil "Editing" sans que deux systèmes d'entrée ne se disputent la même fenêtre.

**Comportement pendant que le panneau est ouvert** : le curseur est libéré (plus de capture FPS-style), et les entrées de gameplay (regard caméra, ZQSD) sont suspendues — sinon glisser un slider ferait aussi tourner la caméra, et ZQSD ferait s'envoler la caméra sous le panneau. `Échap` ferme le panneau s'il est ouvert, sinon quitte l'application.

## 4. Disposition clavier (AZERTY / QWERTY)

Les touches GLFW (`glfw.KEY_W`, `KEY_A`, etc.) désignent des **positions physiques**, pas des caractères imprimés — c'est un choix de conception documenté de GLFW, valable sur toutes les plateformes qu'il supporte (macOS, Windows, Linux). Sur un clavier AZERTY français, la touche physiquement en position "W" (rangée du haut, 2ᵉ touche) porte l'inscription "Z" — donc `glfw.KEY_W` se déclenche en appuyant sur la touche marquée Z, exactement la convention ZQSD standard des joueurs français, **sans code spécifique à écrire** : c'est le comportement par défaut de GLFW, pas quelque chose que ce moteur a dû ajouter. La même ligne de code produit un contrôle WASD correct sur un clavier QWERTY (Windows/Nvidia inclus). Les indications à l'écran/dans le code mentionnent "ZQSD (WASD)" pour éviter toute confusion, mais aucune branche de code ne distingue les deux dispositions.

## 5. Ray tracing hybride : réflexions douces adaptatives

Toujours une intersection analytique dans `forward.frag` (voir §0.1) contre une petite liste de primitives (plan, boîte, **sphère** — ajoutée dans cette itération, voir `intersect_sphere_local`) transmise par un UBO dédié. Nouveauté : les réflexions sont maintenant **échantillonnées plusieurs fois avec une direction perturbée** (`jitter_reflection`, un cône dont l'angle croît avec la rugosité du matériau), moyennées — ce qui donne de vraies réflexions douces (floues) plutôt qu'un simple miroir nette, avec un nombre d'échantillons piloté par le niveau de qualité (1 rayon net au niveau 1, jusqu'à 4 rayons doux au niveau 3). Le rayon n°0 est toujours la direction miroir exacte, donc le niveau 1 reproduit exactement le comportement (plus simple) de la version précédente de ce moteur.

Les cylindres et cônes ne participent pas encore au RT (pas de routine d'intersection écrite pour ces formes) — limite documentée, pas un oubli (voir `RTPrimitive` dans `scene.py`).

## 6. Éclairage volumétrique (nouveau)

`engine/volumetric_pass.py` + `shaders/volumetric.frag` : rayons de lumière volumétrique ("god rays") pour la lumière directionnelle, par marche de rayons en espace écran. Pour chaque pixel, on avance depuis la caméra jusqu'à la profondeur réelle de la scène (lue depuis le depth buffer HDR du forward pass), en accumulant la longueur de trajet **éclairée** (interrogée via la shadow map à chaque pas), puis composée additivement dans le buffer HDR avant le tonemapping.

**Bug trouvé et corrigé pendant le développement** : la première version accumulait `longueur_du_segment × densité` de façon linéaire et sans plafond — n'importe quel pixel regardant loin dans une direction non occultée (le sol, par exemple) accumulait une contribution énorme, noyant toute l'image de blanc. Corrigé avec un modèle à saturation façon Beer-Lambert : `1 - exp(-longueur_éclairée × densité)`, borné dans [0, 1) quelle que soit la distance parcourue — c'est le mécanisme standard pour ce genre d'intégration, et sa nécessité n'était pas évidente avant de voir le résultat à l'écran (l'image blanche saturée est ce qui a révélé le problème). Exactement le genre d'erreur que la vérification visuelle systématique (§8) est censée attraper.

Le nombre de pas de marche suit le niveau de qualité (12/24/48). L'effet nécessite les ombres (niveau > 0) pour avoir un sens — sans shadow map, il n'y a rien à occulter et l'effet dégénère en un voile plat au lieu de rayons localisés ; le panneau de réglages l'indique (`(needs shadows)`).

## 7. Davantage de primitives, pour la flexibilité de création

`engine/primitives.py` ajoute sphère (UV sphere, 32×16 segments — les valeurs par défaut de Blender), cylindre et cône (32 segments, également la valeur par défaut de Blender) au plan et au cube déjà présents. Sphère et faces latérales du cylindre/cône sont en **normales lissées** (la vraie normale d'une sphère est sa direction radiale — pas besoin de moyennage pour la justifier), cube/plan/capuchons restent en facettes plates, cohérent avec l'esthétique "primitive non encore lissée" d'un outil de modélisation avant qu'un·e artiste ne l'ait explicitement lissée. Chaque générateur est vérifié individuellement (bornes d'indices valides, géométrie finie, rendu visuel correct — voir les captures produites pendant le développement).

## 8. Passage réalisme / matériaux

Sur retour explicite ("no flashy colors — like that red cube — [...] ruin the understated, realistic atmosphere"), tous les matériaux de la scène de démonstration ont été remplacés par des teintes neutres et désaturées : sol gris béton, cube gris pierre, sphère métal brossé, cylindre résine/caoutchouc sombre. La lumière ponctuelle, auparavant un bleu saturé (0.2, 0.5, 1.0), est devenue un blanc froid discret (0.7, 0.78, 0.95) - un contraste chaud/froid crédible avec la lumière directionnelle plutôt qu'une teinte "néon". Les formes se distinguent par silhouette et par rugosité/métallicité (PBR), pas par une couleur criarde.

## 9. Résultats de benchmark

Mesures produites par `benchmark.py` (1280×720 fenêtre / 2560×1440 framebuffer Retina, 240 frames par configuration, plafond de framerate désactivé pour la mesure, chaque fonctionnalité testée à son niveau de qualité maximal, 3) :

| Configuration | FPS | CPU ms/frame | GPU ms/frame |
|---|---|---|---|
| Base (PBR seul) | 306.9 | 0.419 | 0.459 |
| + shadow mapping (niveau 3) | 257.7 | 0.450 | 1.636 |
| + SSAO (niveau 3) | 235.9 | 0.597 | 2.700 |
| + ray tracing hybride (niveau 3, réflexions douces) | 135.8 | 0.629 | 4.604 |
| + lumière volumétrique (niveau 3) | 120.0 | 0.673 | 5.685 |

Vérification du limiteur de framerate : configuration de base avec le plafond à 60 FPS réactivé → **59.8 FPS** (contre 306.9 FPS non plafonnée).

Lecture : même cumulées à leur niveau de qualité *maximal*, toutes les fonctionnalités tiennent à 120 FPS sur un GPU classé `medium` par ce moteur lui-même — largement au-dessus de la fréquence d'affichage courante, avec une marge confortable avant d'atteindre un GPU `high` (RTX 4070 Ti). Le RT hybride est devenu plus coûteux que dans une version antérieure de ce rapport du fait de l'échantillonnage multiple pour les réflexions douces (§5) - un compromis qualité/coût désormais explicite et réglable par niveau plutôt qu'implicite.

## 10. Limites connues

- **OpenGL 4.1 partout, pas de chemin compute-shader** (§0.1) : décision délibérée pour garder un seul chemin de rendu vérifiable, y compris depuis une machine de développement Apple Silicon. Sur Nvidia/Windows avec GL 4.3+, le moteur fonctionne à l'identique (rien ne dépend d'une absence de 4.3) mais n'exploite pas de compute shader pour autant.
- **RT limité aux primitives analytiques** (plan, boîte, sphère), pas à des maillages arbitraires ; cylindres/cônes n'y participent pas encore (§5).
- **Réflexions RT à un seul rebond**, éclairées par la lumière directionnelle directe uniquement au point d'impact (pas de second appel à la shadow map/SSAO à cet endroit).
- **Volumétrique dépendant des ombres** : sans shadow map active, l'effet n'a pas de sens géométrique et est désactivé de fait (indiqué dans l'UI).
- **Classification matérielle par sous-chaînes** (`capabilities.py`) : heuristique fondée sur `GL_VENDOR`/`GL_RENDERER`, pas une base de données exhaustive de GPU — un GPU très récent ou obscur peut retomber sur le tier `medium` par défaut plutôt que sa vraie catégorie.
- **Mesure GPU non-bloquante "en pratique" plutôt que garantie par l'API** : ModernGL n'expose pas de sondage `GL_QUERY_RESULT_AVAILABLE` (ring buffer de profondeur 4 utilisé à la place, voir `gpu_timing.py`).
- **Shadow map à frustum fixe**, pas ajusté aux bornes réelles de la scène courante.
- **Pas de culling de face** : négligeable au nombre d'objets de cette démo.

## 11. Comment lancer le projet

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python main.py                 # démo interactive
./.venv/bin/python benchmark.py            # mesures avant/après
```

Contrôles : ZQSD/WASD + souris pour la caméra, Espace/Ctrl pour monter/descendre, Shift pour sprinter, **TAB** pour ouvrir/fermer le panneau de réglages (ombres/SSAO/RT/volumétrique, niveaux 0-3, exposition, V-Sync), Échap pour fermer le panneau ou quitter.

Options de développement de `main.py` (utilisées pour produire les captures de vérification pendant le développement, cf. `engine/capture.py`) : `--frames N` (quitte automatiquement après N frames), `--screenshot chemin.png`, `--shadows`/`--ssao`/`--rt`/`--volumetric N` (force un niveau 0-3 au démarrage), `--settings-open`, `--width`/`--height`.
