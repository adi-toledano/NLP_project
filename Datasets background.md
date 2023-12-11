## Regression 

### FreeSolv
The Free Solvation Database (FreeSolv) provides experimental and calculated hydration free energy of small molecules in water.
16 A subset of the compounds in the dataset are also used in the SAMPL blind prediction challenge.
15 The calculated values are derived from alchemical free energy calculations using molecular dynamics simulations.
We include the experimental values in the benchmark collection, and use calculated values for comparison.

### Lipophilicity
Lipophilicity is an important feature of drug molecules that affects both membrane permeability and solubility.
This dataset, curated from ChEMBL database,45 provides experimental results of octanol/water distribution coefficient (log D at pH 7.4)
of 4200 compounds.

## Classification

### BACE
BACE The BACE dataset provides quantitative (IC50) and qualitative (binary label) binding results for a set of 
inhibitors of human β-secretase 1 (BACE-1).51 All data are experimental values reported in scientific literature over 
the past decade, some with detailed crystal structures available. We merged a collection of 1522 compounds with their 
2D structures and binary labels in MoleculeNet, built as a classification task. Similarly, regarding a single protein 
target, scaffold splitting will be more practically useful.

### BBBP
BBBP The Blood–brain barrier penetration (BBBP) dataset comes from a recent study52 on the modeling and
prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid,
the blood–brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a 
long-standing issue in development of drugs targeting central nervous system. This dataset includes binary labels 
for over 2000 compounds on their permeability properties. Scaffold splitting is also recommended for this well-defined
target.

### HIV
HIV The HIV dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the
ability to inhibit HIV replication for over 40 000 compounds.47 Screening results were evaluated and placed into 
three categories: confirmed inactive (CI), confirmed active (CA) and confirmed moderately active (CM).
We further combine the latter two labels, making it a classification task between inactive (CI) and active (CA and CM).
As we are more interested in discover new categories of HIV inhibitors, scaffold splitting (introduced in the next subsection)
is recommended for this dataset.
