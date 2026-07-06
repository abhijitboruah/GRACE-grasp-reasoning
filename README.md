# GRACE: Constraint-Based Ontological Reasoning for Explainable Grasp Interpretation

This repository contains the dataset, ontology, and experiment code for the
GRACE framework, an ontology-driven approach to interpreting, explaining, and
validating human grasps captured from a wearable data glove.

GRACE maps per-finger flexion and object-context descriptors onto a
biomechanically grounded grasp taxonomy (seven power and two precision grasp
types), infers grasp types through SWRL rules, and checks each inference for
cross-modal consistency through a set of constraint rules. The repository
allows the reported results to be reproduced and the ontology to be inspected
in a description-logic reasoner.

## Repository structure

```
GRACE-grasp-reasoning/
├── data/
│   └── grace_grasp_dataset.csv      Grasp dataset (40,462 instances)
├── ontology/
│   └── OntoGrasp-ext-GRACE.owl       Ontology with SWRL rules (RDF/XML)
├── code/
│   ├── run_experiments.py            Main evaluation pipeline
│   ├── ablation.py                   Descriptor-group ablation study
│   
├── requirements.txt
├── LICENSE                           MIT (code)
└── LICENSE-data                      CC BY 4.0 (data and ontology)
```

## Dataset

`data/grace_grasp_dataset.csv` contains 40,462 stable grasp instances
collected from a customized wearable data glove instrumented with ten ADXL335
tri-axial accelerometers. Each row is a single grasp execution recorded at a
stable hand configuration.

Key columns:

| Column group | Description |
|---|---|
| `*_MCP(ab/ad)`, `*_MCP(f/e)`, `*_PIP(f/e)`, `*_DIP(f/e)`, `Thumb_IP(f/e)`, `Thumb_TMC(*)` | Joint angles for the five fingers (abduction/adduction and flexion/extension at the metacarpophalangeal, proximal interphalangeal, distal interphalangeal, and thumb interphalangeal/trapeziometacarpal joints) |
| `Shape` | Object shape (cylinder, sphere) |
| `Object` | Object identifier |
| `Material`, `Tactility`, `Curvature` | Object physical attributes |
| `Grip_Aperature` | Grip aperture category (Minimal, Intermediate, Maximal), annotated independently from object geometry |
| `Grasp_Type` | Target label: one of nine grasp types (Large_Diameter, Small_Diameter, Medium_Wrap, Thumb_Adducted, Power_Sphere, Sphere_3_Finger, Sphere_4_Finger, Tripod, Quadpod) |

The grasp-type distribution is imbalanced, from 9,976 instances for
`Thumb_Adducted` to 1,869 for `Sphere_4_Finger`.

## Ontology

`ontology/OntoGrasp-ext-GRACE.owl` is an RDF/XML OWL file containing the grasp
class hierarchy, descriptor classes, object properties, family- and leaf-level
disjointness axioms, eleven SWRL inference rules (R1–R11), the scoped
constraint rule C2, and three example individuals.

It can be opened in Protege 5.x and reasoned over with HermiT. The SWRL rules
execute through the SWRLTab Drools engine. 

## Reproducing the experiments

Install the dependencies:

```
pip install -r requirements.txt
```

Run the main evaluation (predictive performance, ontology diagnostics,
constraint-violation breakdown, runtime, and significance tests):

```
cd code
python run_experiments.py
```

Run the descriptor-group ablation study:

```
python ablation.py
```

Both scripts read the dataset from the working directory; place
`grace_grasp_dataset.csv` alongside the scripts or adjust the `DATA_PATH`
variable. The evaluation uses 5-fold stratified cross-validation with a fixed
random seed (42); flexion discretization thresholds (33rd and 66th
percentiles) are computed on the training fold only.

## Licensing

The code in `code/` is released under the MIT License (see `LICENSE`). The
dataset in `data/` and the ontology in `ontology/` are released under the
Creative Commons Attribution 4.0 International License (see `LICENSE-data`).

## Citation

If you use this dataset, ontology, or code, please cite the associated
publication:

```
Abhijit Boruah, Nayan M. Kakoty, Debajit Sarma.
"GRACE: Constraint-Based Ontological Reasoning for Explainable Grasp
Interpretation." (under review).
```

## Contact

Abhijit Boruah — Dibrugarh University Institute of Engineering and Technology
(DUIET), Dibrugarh University, Assam, India.
