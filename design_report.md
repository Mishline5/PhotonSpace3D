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

## 10. Passe de réalisme (post-rapport initial)

Six extensions au pipeline forward, chacune vérifiée indépendamment (`main.py --screenshot` avant/après, `benchmark.py` pour le coût GPU) :

- **Compensation d'énergie GGX** (`forward.frag`) : fit fermé de Karis (`env_brdf_approx`/`energy_compensation`) corrigeant la perte d'énergie du GGX simple-scatter à forte rugosité - toujours actif, pas de niveau de qualité (corrige un terme, ne coûte presque rien).
- **Ambiant hémisphérique** (`Scene.ambient_light`, deux couleurs ciel/sol mélangées par `N.y`) remplaçant l'ancienne constante plate `vec3(0.05)*albedo`, dans `forward.frag` et dans `trace_reflection`. `FrameUBO` passe de 368 à 400 octets (deux `vec4` ajoutés strictement après `u_flags`, jamais avant, pour que `prepass.vert`/`volumetric.frag` - qui déclarent le même bloc `Frame` mais s'arrêtent à `u_flags` - restent corrects sans être modifiés).
- **Brouillard atmosphérique** (extinction Beer-Lambert sur la distance caméra-fragment, teinte dérivée de l'ambiant + lueur solaire), niveau 0-3 (`FOG_PARAMS`), algébriquement neutre à densité 0.
- **RT hybride enrichi** : intersection cylindre analytique (`intersect_cylinder_local`, kind=3 dans `RTPrimitive`) - le cylindre de la démo est maintenant réfléchissant/réfléchi ; la réflexion consulte désormais `sample_shadow` au point d'impact (`trace_reflection`) au lieu de lumière directe seule.
- **Ombres PCSS** remplaçant le PCF à rayon fixe : recherche de bloqueurs + rayon de pénombre proportionnel à la distance bloqueur/récepteur (exact, pas approximatif, car cette shadow map orthographique a une profondeur déjà linéaire en distance monde). Frustum ajusté chaque frame à la sphère englobante de la scène (`Scene.world_bounds`, `ShadowPass.fit_frustum`) plutôt qu'une boîte fixe de 8 unités.
- **MSAA** (`MSAATarget`, renderbuffers multi-échantillons résolus vers `HDRTarget` avant les passes en aval) : niveau 0/2x/4x/8x, clampé au `GL_MAX_SAMPLES` réel du GPU (4 sur la machine de développement M1 Max - un GPU annoncé "8x" ne l'obtiendrait donc pas forcément partout).

**Bug préexistant trouvé et corrigé en vérifiant le brouillard** : un pixel de ciel vide (sans géométrie, profondeur nettoyée au plan lointain) faisait marcher le rayon volumétrique jusqu'à ~94 unités sans occultation trouvée, saturant le terme de Beer-Lambert à ~1 et peignant tout le ciel vide à la couleur de la lumière directionnelle en pleine intensité dès que le volumétrique était actif - `volumetric.frag` retourne maintenant une contribution nulle quand la profondeur lue est au plan lointain (`>= 0.9999`), puisqu'il n'y a rien là pour diffuser la lumière (pas de skybox dans cette démo).

Benchmark (mêmes conditions que §9, tout au niveau 3, cumulatif) :

| Configuration | FPS | GPU ms/frame |
|---|---|---|
| Base (PBR seul) | 450.7 | 0.538 |
| + ombres (PCSS) | 234.0 | 2.743 |
| + SSAO | 189.5 | 3.549 |
| + RT hybride (avec ombre au point d'impact) | 118.5 | 6.651 |
| + volumétrique | 112.7 | 7.481 |
| + brouillard | 113.0 | 7.479 |
| + MSAA | 101.2 | 8.140 |

Même avec absolument tout au niveau 3 simultanément, la scène reste au-dessus de 100 FPS sur ce GPU `medium`. Le coût le plus notable est l'ombre au point d'impact de la réflexion RT (§ci-dessus) : elle ajoute une recherche PCSS complète par échantillon de réflexion (jusqu'à 4× au niveau 3), identifié comme le risque de performance principal de cette passe et confirmé mesurable (RT hybride : 4.6ms auparavant, 6.65ms maintenant) mais encore largement acceptable.

## 10bis. Textures PBR procédurales et scène enrichie

**Textures** (`engine/procedural_textures.py`, `engine/material_textures.py`) : bruit de valeur/Worley/fBm en numpy (jamais de fichier lu ni téléchargé), quatre présets (béton, métal brossé, caoutchouc, plâtre) donnant albedo/roughness/(metallic)/normal. `Material` (`scene.py`) gagne des champs `*_map` optionnels ; sans texture assignée, `MaterialTextures` fournit une texture 1x1 encodant le scalaire existant, pour que `forward.frag` garde un seul chemin de code (toujours un sampler, jamais de branche). Normal mapping par dérivées d'écran (`dFdx`/`dFdy`, `cotangent_frame`) plutôt qu'un attribut tangente sur les vertices - zéro changement à `mesh.py`/`primitives.py`. **Bug trouvé en vérifiant** : `forward.vert` déclarait `in_uv` mais ne le transmettait jamais au fragment shader - corrigé (`out vec2 v_uv`).

**Scène** (`main.py: build_scene()`) : sol 20×20 (était 12×12), deux murs formant un angle (plâtre), plateforme basse, second cylindre ("tambour", échelle non-uniforme radiale-symétrique X=Z pour rester correct sous transformation de normale), cône (visible, pas encore réfléchissant - voir §11). Chaque objet reçoit un vrai jeu de textures procédurales. Nouveau générateur `primitives.make_box(half_extents)` : contrairement à `make_cube` + `Transform.scale`, ses UV sont mises à l'échelle par la taille réelle de chaque face (même convention que `make_plane`) - nécessaire pour que les murs (très étirés) tuilent leur texture à une densité raisonnable au lieu d'étaler un unique cycle de texture sur toute la surface.

**Deux bugs de calibration trouvés en vérifiant visuellement** (la texture du mur restait invisible malgré une génération correcte) :
- Les murs apparaissaient presque blancs, sans variation visible. Cause réelle : la passe volumétrique (`u_density`, jusqu'ici 0.02, jamais recalibrée) intègre la "quantité de lumière" sur tout le segment caméra→surface sans notion de proximité à un occultant - un mur à ~15-20 unités, avec de l'air dégagé devant, accumulait un `lit_path_length` énorme et saturait le terme de Beer-Lambert près de son maximum, ajoutant une contribution additive massive. Diagnostiqué en lisant directement le buffer HDR linéaire (avant tonemap) et en isolant chaque passe une à une plutôt qu'en devinant. `u_density` recalibré à 0.003.
- L'ambiant hémisphérique (§10) était plus lumineux en moyenne que l'ancienne constante `0.05` qu'il remplaçait, ce qui poussait plus de surfaces vers la partie compressive de la courbe ACES (où la variation d'albédo devient visuellement imperceptible). `AmbientLight.sky_color`/`ground_color` recalibrés pour retrouver une luminosité totale proche de l'ancienne constante, tout en gardant la variation directionnelle ciel/sol qui est le vrai apport de ce changement.

**Coût mesuré** : la scène enrichie (8 primitives RT au lieu de 3-4, sol/murs réfléchissants plus grands) a fait chuter le RT hybride + ombre au point d'impact sous les 60 FPS non plafonnés à qualité 3 partout (35.7 FPS). Repli déjà anticipé au §10 appliqué : l'ombre PCSS au point d'impact ne s'applique plus qu'à l'échantillon miroir (index 0), pas aux échantillons flous jitterés, ramenant la config "tout au niveau 3" à ~49 FPS. **Le préréglage par défaut (tier medium, niveaux 2) reste à ~82 FPS non plafonné** - le cas "tout au maximum" est désormais un vrai compromis qualité/coût pour du matériel plus modeste, pas une garantie universelle comme au §9.

## 10ter. Auto-exposition

`engine/exposure_pass.py` + `shaders/luminance.frag`/`luminance_adapt.frag` : adaptation à la luminance moyenne de la scène, entièrement sur GPU (pas de lecture CPU, contrairement à `gpu_timing.py` dont la contrainte de blocage est réelle - ici rien ne l'impose). Luminance log2 capturée dans une petite texture (résolution pilotée par niveau de qualité), chaîne de mip générée (`Texture.build_mipmaps()`), lue à son niveau le plus grossier par une passe 1×1 qui mélange avec la valeur adaptée de la frame précédente (ping-pong, lissage exponentiel indépendant du framerate). Le slider d'exposition manuel devient une compensation EV appliquée par-dessus, activable/désactivable (`u_use_auto_exposure`). Limite connue et attendue (pas un bug) : comme tout système d'auto-exposition moyennant la luminance, une scène globalement claire (murs/sol clairs de cette démo) est tirée vers le gris moyen 0.18 et apparaît plus lumineuse qu'en exposition manuelle fixe - le même compromis qu'un appareil photo réel, corrigible via la compensation EV manuelle.

## 11. Limites connues

- **OpenGL 4.1 partout, pas de chemin compute-shader** (§0.1) : décision délibérée pour garder un seul chemin de rendu vérifiable, y compris depuis une machine de développement Apple Silicon. Sur Nvidia/Windows avec GL 4.3+, le moteur fonctionne à l'identique (rien ne dépend d'une absence de 4.3) mais n'exploite pas de compute shader pour autant.
- **RT limité aux primitives analytiques** (plan, boîte, sphère), pas à des maillages arbitraires ; cylindres/cônes n'y participent pas encore (§5).
- **Réflexions RT à un seul rebond**, éclairées par la lumière directionnelle directe uniquement au point d'impact (pas de second appel à la shadow map/SSAO à cet endroit).
- **Volumétrique dépendant des ombres** : sans shadow map active, l'effet n'a pas de sens géométrique et est désactivé de fait (indiqué dans l'UI).
- **Classification matérielle par sous-chaînes** (`capabilities.py`) : heuristique fondée sur `GL_VENDOR`/`GL_RENDERER`, pas une base de données exhaustive de GPU — un GPU très récent ou obscur peut retomber sur le tier `medium` par défaut plutôt que sa vraie catégorie.
- **Mesure GPU non-bloquante "en pratique" plutôt que garantie par l'API** : ModernGL n'expose pas de sondage `GL_QUERY_RESULT_AVAILABLE` (ring buffer de profondeur 4 utilisé à la place, voir `gpu_timing.py`).
- **Shadow map à frustum fixe**, pas ajusté aux bornes réelles de la scène courante.
- **Pas de culling de face** : négligeable au nombre d'objets de cette démo.

## 12. Comment lancer le projet

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python main.py                 # démo interactive
./.venv/bin/python benchmark.py            # mesures avant/après
```

`main.py` : ZQSD/WASD + souris pour la caméra, Espace/Ctrl pour monter/descendre, Shift pour sprinter, Échap pour quitter - un exemple d'usage minimal du moteur, sans aucune UI d'édition (voir §13).

Options de développement de `main.py`/`edit.py` (utilisées pour produire les captures de vérification pendant le développement, cf. `engine/capture.py`) : `--frames N` (quitte automatiquement après N frames), `--screenshot chemin.png`, `--shadows`/`--ssao`/`--rt`/`--volumetric`/`--fog`/`--msaa`/`--auto-exposure N` (force un niveau 0-3 au démarrage, `main.py` seulement), `--width`/`--height`.

## 13. Module `editing/` : outil interactif façon Blender

Second consommateur du package `engine/`, exactement comme anticipé au §0.2 - `engine/` ne garde que le rendu/la gestion (plus aucune dépendance à `pyimgui` : `engine/settings_ui.py` a été supprimé, sa logique déménagée dans `editing/ui.py`). Point d'entrée : `edit.py` à la racine, miroir structurel de `main.py`.

- **Sélection** (`editing/picking.py`) : rayon caméra→pixel, intersections analytiques dupliquées volontairement en Python/PyGLM (miroir de celles de `forward.frag` - aucun partage de code GLSL↔Python n'étant possible). `ensure_pickable()` donne une boîte englobante aux objets sans `RTPrimitive`.
- **Gizmos** (`editing/gizmo_geometry.py`, `gizmo.py`, `gizmo_interaction.py`) : flèches (translation), tores triangulés (rotation - pas de `GL_LINE_STRIP`, `glLineWidth` n'étant pas fiable en core profile), poignées cubiques (échelle), taille apparente constante à l'écran (formule exacte via le FOV, pas une approximation). Survol par distance au segment/polyligne projeté en 2D, pas un raycast 3D contre la géométrie du gizmo. Drag contraint à un axe via intersection rayon/plan (plan figé au début du drag). Déplacement/rotation en axes monde, échelle en axes locaux de l'objet (consequence de l'ordre `T*R*S`, pas une incohérence).
- **UI** (`editing/ui.py`) : panneau Quality (déplacé de `engine/settings_ui.py`) + panneau Properties (transform, matériau, boutons de préset de texture procédurale) toujours visible, un seul contexte Dear ImGui partagé.
- **Contrôles** (`edit.py`) : curseur jamais capturé (nécessaire pour cliquer précisément) ; clic droit maintenu = regard caméra (WASD reste actif sans condition) ; clic gauche = sélection/drag de gizmo ; G/R/S = mode translate/rotate/scale (convention Blender) ; TAB = panneau Quality.

**Vérification** : comportementale (pas seulement visuelle) - simulation de clic/drag via manipulation directe de `Window.input` puis lecture de l'état résultant (`session.selection`, `obj.transform.*`), confirmant que la sélection et les trois modes de drag modifient exactement l'axe/la propriété attendue et rien d'autre.

**Trois bugs trouvés en vérifiant** (aucun n'était visible par simple inspection du code) :
- `engine/capture.py` : `ctx.screen.read()` laisse une erreur GL fantôme (reproduite avec un contexte ModernGL minimal, sans rien de ce projet - caractéristique ModernGL/pilote sur cette plateforme, pas un bug de ce moteur) qui restait silencieuse jusqu'à ce qu'un appel PyOpenGL avec vérification stricte (le rendu Dear ImGui) la fasse remonter en exception, sur n'importe quelle frame suivant une capture d'écran dans le même processus. Corrigé en purgeant l'erreur après lecture.
- `editing/gizmo.py` : le premier jet du gizmo activait le test de profondeur contre `ctx.screen`, dont le tampon de profondeur n'est en réalité jamais rempli par le pipeline (tout le rendu 3D se fait hors-écran ; seul le triangle plein écran du tonemap touche `ctx.screen`, sans profondeur) - le gizmo échouait donc silencieusement son test de profondeur partout. Corrigé en désactivant le test de profondeur pour le gizmo (trois poignées qui partent d'une origine commune se chevauchent rarement à l'écran, simplification acceptable).
- Le mécanisme initialement prévu pour corriger le point précédent (masquer les canaux couleur puis vider seulement la profondeur) cassait silencieusement le rendu Dear ImGui de la frame suivante (aucune exception, panneau simplement invisible) - retiré en même temps que le test de profondeur lui-même, qui n'en avait plus besoin.

**Hors scope à l'époque** (voir §14, la plupart désormais livrés) : plusieurs lumières, ajout/suppression d'objets depuis l'éditeur, etc.

## 14. Grande passe : lumières multiples, éditeur complet, mode jeu

### 14.1 Rendu (`engine/`)
- **Lumières multiples** (`engine/lights.py`, `LightsUBO`, binding 2) : tableau de lumières point + spot séparé du soleil (qui reste dans le FrameUBO). Chaque lumière est portée par un `SceneObject` (`LightComponent`) → sélectionnable/déplaçable comme tout objet, avec un marqueur émissif (`Material.emissive`, ampoule/cône) masqué en mode jeu (`SceneObject.marker`). Spot : direction depuis la rotation, cône élargi par l'échelle. `forward.frag` boucle sur ces lumières (falloff windowed + cône lisse pour les spots).
- **Textures tuilables** : le bruit de valeur/fBm échantillonne désormais un réseau périodique (indices modulo) → couture parfaite là où les UV se répètent (corrige "les textures ne se répètent pas parfaitement"). Nouveau `make_noise_textures` générique (grain/force réglables) piloté depuis les propriétés d'objet (None/Noise).
- **Poussière** (`engine/dust_pass.py`) : pool de points sprites minuscules dérivant et wrappés autour de la caméra, réglables (densité niveau 0-3, taille, mouvement, opacité, couleur), profondeur testée contre la scène, alpha-blend en HDR.
- **Filtres** (`tonemap.frag`) : noir & blanc, et bodycam (vignette circulaire + FOV élargi côté caméra + légère aberration chromatique) réservé au jeu.
- **Ombres plus profondes** (résolutions 1024/2048/4096, pénombre resserrée, soleil un peu plus intense, ambiant abaissé). **Auto-exposition adoucie et clampée** (multiplicateur borné [0.6, 1.8], plancher de luminance) - fini le "tout blanc quand on fixe une lumière" ; **désactivée par défaut dans l'éditeur, active ailleurs**.
- **Plein écran** par défaut (`Window(fullscreen=True)`), fenêtré via `--windowed`/`--frames`. Correction HiDPI : le curseur est rapporté en pixels framebuffer (clics ImGui/picking corrects sur Retina).

### 14.2 Éditeur (`editing/`)
- **Barre d'outils** haute centrée à icônes vectorielles procédurales (`editing/icons.py`, aucun asset) : réglages, ajout d'objet (menu déroulant géométrie + lumières), soleil, move/rotate/scale/global-scale, lancer, quitter.
- **Menu contextuel** au clic droit *immobile* (un clic droit qui bouge = orbite caméra) : propriétés, les 4 modes de transform, copier, coller (grisé si presse-papier vide), supprimer.
- **Propriétés** (fenêtre à croix de fermeture) : couleur en cercle cliquable, metallic/roughness/reflectivity, texture None/Noise + paramètres, case collision ; pour une lumière : couleur + intensité (+ rotation/échelle pour un spot).
- **Contour de sélection** orange sobre (fil de fer), **global scale**, **copier/coller/supprimer** (raccourcis Ctrl+C/V, X/Suppr), **panneau Soleil** (rotation/hauteur/intensité/couleur).

### 14.3 Mode jeu (`game/`)
- **Lancer** (bouton play) passe éditeur→jeu ; **Échap** revient. Personnage FPS (souris + ZQSD, gravité, saut, **pas de vol**), **collisions exactes au maillage** (`game/collision.py` : sphère vs triangles vectorisé numpy, glisse le long des faces → on monte une rampe/cône sans être bloqué), respectant la case collision par objet.
- **Réglages graphiques du jeu séparés** (instance dédiée, TAB) : les modifier ne touche pas ceux de l'éditeur. Marqueurs de lumière masqués, filtre bodycam disponible ici.
- `game/` est le troisième consommateur de `engine/` (après `main.py` et `editing/`) : moteur + environnement construit dans l'éditeur + code de jeu, prêt à recevoir une logique de jeu spécifique.
