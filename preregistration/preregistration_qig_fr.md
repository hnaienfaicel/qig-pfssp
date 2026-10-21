# Pré-enregistrement : Iterated Greedy quantique (Q-IG) contre son jumeau classique

*Rédigé le 19 octobre 2026, avant tout calcul de réglage ou de test.*

## Question

Le même algorithme glouton itéré (IG, Ruiz & Stützle 2007, d = 4, T = 0,4), avec ses sous-programmes de recherche remplacés par leurs versions quantiques, fait-il mieux que sa version classique, au même nombre d'appels à l'oracle (évaluations du makespan) ?

## Bras

| Bras | Destruction | Construction (meilleure des n positions) | Recherche locale | Acceptation |
|---|---|---|---|---|
| **IG** (jumeau classique de Q-IG-min) | classique | balayage classique (n évaluations, règle de Taillard) | IG : chaque travail retiré puis réinséré au meilleur endroit (balayage) | Ruiz & Stützle |
| **Q-IG-min** | classique | minimum quantique (Dürr–Høyer) | IG, chaque réinsertion par minimum quantique | idem |
| **IG-R** (jumeau classique de Q-IG-G) | classique | balayage classique | descente sur tout le voisinage d'insertion par tirages uniformes sans remise (jumeau de même loi) | idem |
| **Q-IG-G** | classique | minimum quantique (Dürr–Høyer) | descente sur tout le voisinage d'insertion par amplification de Grover (BBHT) | idem |

Départ NEH pour tous (évaluations de NEH comptées). Budget 46 080 appels. Graines 1000 + 7919·r, identiques pour tous les bras.

## Modèle de coût (identique aux campagnes Grover précédentes)

- Balayage classique des n positions d'un travail : n évaluations.
- Minimum quantique, sur un registre de w = ⌈log₂ n⌉ qubits (N = 2^w codes) :
  - une évaluation pour la position initiale tirée au hasard ;
  - puis des recherches BBHT successives au seuil « makespan courant » (λ = 1,2) ; chaque tour coûte k appels à l'oracle, plus une évaluation pour vérifier le candidat mesuré (le code mesuré est valide avec la probabilité exacte) ;
  - arrêt dès qu'une recherche dépasse cq_min·√N appels sans succès.
- Descente de Grover : la fonction `ils_amp` des campagnes précédentes, sans changement (registre de mouvement, BBHT, arrêt après cq_ls·√N).
- Les makespans de toutes les branches sont calculés en classique pour l'émulation. Ils ne sont pas facturés.
- La loi de sortie émulée est celle du circuit, vérifiée porte par porte sur Aer à 4 × 3 et à 8 × 2.

## Réglage (TUNE+ : tai20_20, tai50_10, tai50_20 n° 0–4 et tai100_20 n° 0–4, soit 20 instances × 5 graines)

- Q-IG-min : cq_min ∈ {1, 2, 3, 4}.
- Q-IG-G : cq_min ∈ {2, 3} × cq_ls ∈ {1, 3, 6}.
- IG-R : cq_ls ∈ {1, 3, 6}.
- IG : aucun paramètre (valeurs publiées).
- Règle de choix : écart moyen best-of-5 minimal sur TUNE+. Si le meilleur réglage tombe au bord de la grille, la grille est étendue une fois, avec un amendement daté.

## Test confirmatoire

- **Jeux** : TEST (100 instances : 10 instances de chacun des groupes tai20_5, tai20_10, tai50_5, tai100_5, tai100_10, VRF10_10, VRF20_5, VRF50_5, VRF50_15, VRF100_20 ; liste dans `data/instances.csv`, 30 exécutions) et HARD (40 instances neuves, 30 exécutions). *[Version publique, 20/10/2026 : le renvoi à un manuscrit non publié a été remplacé par la liste des instances ; rien d'autre n'a changé.]*
- **Unité** : l'instance. Wilcoxon apparié, IC bootstrap à 95 % (10 000 tirages), correction de Holm sur la famille.
- **Critère principal** : best-of-30. Critère secondaire : moyenne des 30.

**Famille de tests principale** (6 tests, Holm) :
- **H1, apport quantique du glouton** : Q-IG-min − IG < 0, sur TEST et sur HARD.
- **H2, apport quantique de la recherche locale** : Q-IG-G − IG-R < 0, sur TEST et sur HARD.
- **H3, battre l'IG classique** : Q-IG-G − IG < 0, sur TEST et sur HARD.

**Exploratoire** (sans correction) :
- *[Version publique, 20/10/2026 : une comparaison exploratoire avec des algorithmes non publiés a été retirée ; elle n'est pas rapportée dans l'article.]*
- Effet de H1 selon n.

**Prédiction déclarée.** Le minimum quantique ne gagne en appels qu'au-delà de n ≈ 50 (×0,79 à n = 20, ×1,21 à 50, ×1,70 à 100 pour cq = 3). H1 devrait donc être nul ou faible sur TEST et sur HARD. H2 et H3 devraient être nettement négatifs.

## Limites déclarées

- Il s'agit d'un avantage en appels à l'oracle (modèle de requêtes), pas en temps.
- L'oracle cohérent suppose une machine tolérante aux fautes (44 782 qubits et 6,9 M CX-équivalents par appel à 100 × 20).

## Amendement 1 (19 octobre 2026, après le réglage, avant tout test sur TEST et HARD)

**Résultat du réglage** (écart moyen best-of-5 sur TUNE+) :

| Bras | Réglage | Écart |
|---|---|---|
| IG | — | 1,672 |
| Q-IG-min | cq_min = 1 | 1,343 |
| IG-R | cq_ls = 6 | 1,279 |
| Q-IG-G | cq_min = 2, cq_ls = 1 | 0,642 |

Les trois choix tombent au bord de leur grille.

**A1. Extension unique des grilles**, conformément à la règle :
- Q-IG-min : cq_min ∈ {0,25 ; 0,5} ;
- Q-IG-G : (cq_min, cq_ls) ∈ {(1 ; 0,5), (1 ; 1), (1 ; 3), (2 ; 0,5)} ;
- IG-R : cq_ls ∈ {10 ; 15}.

**A2. Jumeaux supplémentaires de même coût.** Le réglage montre qu'un minimum quantique *inexact* (cq_min = 1) fait mieux qu'un minimum exact. Une partie du gain pourrait donc venir de réinsertions moins coûteuses (plus d'itérations de l'IG), et non de l'interférence. L'IG à balayage complet ne contrôle pas cet effet. J'ajoute donc deux jumeaux, avec la même règle d'arrêt que le minimum quantique :
- **IG-Rmin** : chaque meilleure insertion devient une recherche classique du minimum par tirages uniformes sans remise des positions (1 évaluation par tirage). La recherche s'arrête après cq·√N tirages sans amélioration. Grille cq ∈ {0,5 ; 1 ; 2 ; 3}.
- **IG-RR** : la construction d'IG-Rmin, suivie de la descente par tirages uniformes (jumeau de Grover). Grille cq_min ∈ {0,5 ; 1 ; 2} × cq_ls ∈ {3 ; 6 ; 10}.
- Ces grilles peuvent aussi être étendues une fois.

**Famille secondaire** (4 tests, Holm), déclarée avant le test :
- H1b : Q-IG-min − IG-Rmin < 0, sur TEST et HARD ;
- H2b : Q-IG-G − IG-RR < 0, sur TEST et HARD.

Seules H1b et H2b isolent l'effet de l'interférence à coût égal. H1 et H2 restent rapportées telles que pré-enregistrées.

## Amendement 2 (19 octobre 2026, avant tout test)

Après l'extension A1–A2, les choix sont intérieurs à leur grille pour Q-IG-min (cq_min = 0,5 ; 1,329), Q-IG-G (cq_min = 2, cq_ls = 1 ; 0,642), IG-R (cq_ls = 6 ; 1,279) et IG-Rmin (cq = 1 ; 1,457).

IG-RR choisit (2 ; 10), soit 1,371, au bord de ses deux axes. J'étends sa grille une fois avec (cq_min, cq_ls) ∈ {(2 ; 15), (3 ; 6), (3 ; 10), (3 ; 15)}. Le meilleur réglage obtenu est retenu, même s'il reste au bord.

## Amendement 3 (20 octobre 2026, après tous les tests)

- **Référence des écarts.** Le pré-enregistrement calculait les écarts avec les bornes supérieures livrées avec les instances (fichiers de Taillard, fichier de Vallada et al. 2015). Elles sont dépassées pour 64 des 160 instances (jusqu'à 1,98 %). Les écarts sont désormais calculés contre les meilleures solutions connues de Brunner (2026, Zenodo 20529962, v10), recalculées sur nos fichiers (160 sur 160 identiques).
- **Effet.** Seuls les niveaux changent. Chaque différence appariée bouge de moins de 0,01 point, aucune conclusion de test ne change et le réglage retient les mêmes paramètres. Les résultats contre les bornes d'origine sont conservés (`results/tests_instance_file_bounds.csv`, `results/levels_instance_file_bounds.csv`).
- **Hors pré-enregistrement.** Les exécutions à budget ×10, ×100 et ×1000 (`experiments/run_long.py`, `run_record.py`, `analyse_long.py`) sont exploratoires.

## Amendement 4 (21 octobre 2026, 07:39 UTC, avant toute exécution sur EXT)

- **Jeu de réplication EXT.** 190 instances jamais utilisées : toutes les instances Taillard avec n ≤ 100 et toutes les instances VRF Small (Vallada et al. 2015) qui ne sont ni dans TEST, ni dans HARD, ni dans TUNE+. Cela fait 5 Taillard (tai50_10 n° 5–9) et 185 VRF Small (n = 10 à 60, m = 5 à 20). La liste figure dans `data/instances.csv` (set = EXT). Les meilleures solutions connues (Brunner 2026, v10) ont été recalculées à partir de leurs permutations (190 sur 190 identiques).
- **Protocole figé.** Mêmes bras et réglages (`experiments/arms.json`), 46 080 appels, 30 exécutions, graines 1000 + 7919 r, écart à la meilleure solution connue.
- **Hypothèses confirmatoires.** H1, H2, H3 (famille primaire) et H1b, H2b (famille secondaire), sur le meilleur des 30 exécutions et sur la moyenne des 30. Test de Wilcoxon apparié par instance, IC bootstrap à 95 %, correction de Holm dans chaque famille, comme pour TEST et HARD.
- **Analyse descriptive.** Ensemble poolé TEST ∪ HARD ∪ EXT (330 instances) : niveaux et différences par groupe, sans nouveau test confirmatoire.
- **Études exploratoires, hors confirmatoire.**
  - Robustesse au bruit : Grover imparfait ; avec une probabilité (1 − ε)^k que les k itérations restent cohérentes, sinon la mesure est uniforme.
  - Courbes de convergence : meilleur makespan en fonction des appels, mêmes exécutions que TEST et HARD.
  - Estimation des ressources tolérantes aux fautes.

## Amendement 5 (21 octobre 2026, 13:08 UTC, avant toute exécution de ces études)

Quatre études **exploratoires** (hors confirmatoire, sans correction de Holm commune avec les familles H1–H3). Mêmes graines (1000 + 7919 r), même budget de base (46 080 appels), écart à la meilleure solution connue (Brunner 2026, v10). Script : `experiments/run_amend5.py`.

- **Ablation 2×2.** Nouveau bras Q-IG-LS : construction par scan complet (comme IG et IG-R) et descente de Grover (BBHT, c_ls = 1, réglage repris de Q-IG-G, non re-réglé). Son jumeau est IG-R. TEST, HARD et EXT, 30 exécutions. Analyse : niveaux, différences appariées Q-IG-LS − IG-R et Q-IG-LS − IG (Wilcoxon apparié, IC bootstrap 95 %), décomposition du gain de Q-IG-G sur IG en part « descente » (Q-IG-LS − IG) et part « minimum quantique » (Q-IG-G − Q-IG-LS).
- **Robustesse aux réglages.** TEST et HARD, 10 exécutions (r = 0–9). Grilles : Q-IG-G c_min ∈ {0,5 ; 1 ; 2 ; 3} × c_ls ∈ {0,5 ; 1 ; 2 ; 3} (16 réglages) ; IG-RR r_min ∈ {1 ; 2 ; 3 ; 4 ; 6} × c_ls ∈ {3 ; 6 ; 10 ; 15} (20 réglages, étendue au-delà du bord r_min = 3) ; Q-IG-min c_min ∈ {0,25 ; 0,5 ; 1 ; 2} ; IG-Rmin r_min ∈ {0,5 ; 1 ; 2 ; 3}. Les réglages retenus réutilisent les exécutions 0–9 de la campagne (déterministes). Critère principal : le **pire** réglage quantique contre le **meilleur** réglage du jumeau, choisi a posteriori sur les mêmes instances (choix favorable au jumeau), sur la moyenne par instance du meilleur des 10.
- **Équivalence de budget.** Sur les 38 instances ouvertes de TEST et HARD (`results/raw_long/open_instances.txt`), 5 exécutions (r = 0–4), réglages retenus : IG à ×15, ×20, ×30 et ×50 ; IG-RR à ×5, ×7 et ×15 (×1, ×10 et ×100 existent déjà). On estime par interpolation en log du budget le facteur k* pour lequel le meilleur des 5 de IG (resp. IG-RR) égale celui de Q-IG-G à ×1, puis le temps maximal d'une itération de Grover qui donnerait un avantage en temps réel, comparé aux estimations tolérantes aux fautes.
- **Tableaux par groupe.** Écarts (meilleur des 30, moyenne des 30) par groupe n × m et par bras sur TEST, HARD et EXT, à partir des exécutions existantes. Aucune nouvelle exécution.
