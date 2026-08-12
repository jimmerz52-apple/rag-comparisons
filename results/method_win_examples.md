# Clear wins: Semantic vs GraphRAG vs LazyGraph

A **clear win** means the method beats *both* rivals in the trio by at least the margin threshold on the chosen score column.

## Semantic (vector)
### 2014 S/S is the debut album of a South Korean boy group that was formed by who?
- **Gold:** `YG Entertainment` (bridge)
- **composite_score:** 1.00 vs GraphRAG fast/basic 0.46 (margin **0.54**)
- **Breakdown:** judge=1.00, contains=1, F1=1.00, EM=1, generative=1.00

### Are Local H and For Against both from the United States?
- **Gold:** `yes` (comparison)
- **composite_score:** 0.88 vs GraphRAG fast/basic 0.39 (margin **0.49**)
- **Breakdown:** judge=0.50, contains=1, F1=1.00, EM=1, generative=0.75

### Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
- **Gold:** `yes` (comparison)
- **composite_score:** 0.88 vs GraphRAG fast/basic 0.46 (margin **0.42**)
- **Breakdown:** judge=0.50, contains=1, F1=1.00, EM=1, generative=0.75

### Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
- **Gold:** `no` (comparison)
- **composite_score:** 0.62 vs GraphRAG global 0.25 (margin **0.38**)
- **Breakdown:** judge=0.50, contains=1, F1=0.00, EM=1, generative=0.75

### Are Random House Tower and 888 7th Avenue both used for real estate?
- **Gold:** `no` (comparison)
- **composite_score:** 0.62 vs GraphRAG global 0.45 (margin **0.17**)
- **Breakdown:** judge=0.50, contains=1, F1=0.00, EM=1, generative=0.75

## GraphRAG global
_No clear wins on this slice at this threshold._

## GraphRAG fast/basic
### What science fantasy young adult series, told in first person, has a set of comp…
- **Gold:** `Animorphs` (bridge)
- **composite_score:** 0.46 vs Semantic (vector) 0.00 (margin **0.46**)
- **Breakdown:** judge=0.80, contains=1, F1=0.03, EM=0, generative=0.90

### Who was known by his stage name Aladin and helped organizations improve their pe…
- **Gold:** `Eenasul Fateh` (bridge)
- **composite_score:** 0.40 vs Semantic (vector) 0.12 (margin **0.28**)
- **Breakdown:** judge=0.50, contains=1, F1=0.10, EM=0, generative=0.75

### The arena where the Lewiston Maineiacs played their home games can seat how many…
- **Gold:** `3,677 seated` (bridge)
- **composite_score:** 0.26 vs Semantic (vector) 0.00 (margin **0.26**)
- **Breakdown:** judge=1.00, contains=0, F1=0.04, EM=0, generative=0.50

### Were Scott Derrickson and Ed Wood of the same nationality?
- **Gold:** `yes` (comparison)
- **composite_score:** 0.40 vs GraphRAG global 0.20 (margin **0.20**)
- **Breakdown:** judge=0.50, contains=1, F1=0.11, EM=0, generative=0.75


---

## Same filter on generative score (judge + contains)

# Clear wins: Semantic vs GraphRAG vs LazyGraph

A **clear win** means the method beats *both* rivals in the trio by at least the margin threshold on the chosen score column.

## Semantic (vector)
### Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
- **Gold:** `no` (comparison)
- **generative_score:** 0.75 vs GraphRAG global 0.50 (margin **0.25**)
- **Breakdown:** judge=0.50, contains=1, F1=0.00, EM=1, generative=0.75

## GraphRAG global
### Are Random House Tower and 888 7th Avenue both used for real estate?
- **Gold:** `no` (comparison)
- **generative_score:** 0.90 vs Semantic (vector) 0.75 (margin **0.15**)
- **Breakdown:** judge=0.80, contains=1, F1=0.00, EM=0, generative=0.90

## GraphRAG fast/basic
### What science fantasy young adult series, told in first person, has a set of comp…
- **Gold:** `Animorphs` (bridge)
- **generative_score:** 0.90 vs Semantic (vector) 0.00 (margin **0.90**)
- **Breakdown:** judge=0.80, contains=1, F1=0.03, EM=0, generative=0.90

### Who was known by his stage name Aladin and helped organizations improve their pe…
- **Gold:** `Eenasul Fateh` (bridge)
- **generative_score:** 0.75 vs Semantic (vector) 0.25 (margin **0.50**)
- **Breakdown:** judge=0.50, contains=1, F1=0.10, EM=0, generative=0.75

### The arena where the Lewiston Maineiacs played their home games can seat how many…
- **Gold:** `3,677 seated` (bridge)
- **generative_score:** 0.50 vs Semantic (vector) 0.00 (margin **0.50**)
- **Breakdown:** judge=1.00, contains=0, F1=0.04, EM=0, generative=0.50

### Were Scott Derrickson and Ed Wood of the same nationality?
- **Gold:** `yes` (comparison)
- **generative_score:** 0.75 vs GraphRAG global 0.40 (margin **0.35**)
- **Breakdown:** judge=0.50, contains=1, F1=0.11, EM=0, generative=0.75

### Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?
- **Gold:** `yes` (comparison)
- **generative_score:** 0.90 vs Semantic (vector) 0.75 (margin **0.15**)
- **Breakdown:** judge=0.80, contains=1, F1=0.03, EM=0, generative=0.90
