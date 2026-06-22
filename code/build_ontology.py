#!/usr/bin/env python3
"""
Generate the OntoGrasp-ext (GRACE) ontology as RDF/XML.

Produces the seven-power / two-precision grasp taxonomy, supporting descriptor
classes, object properties (including hasObjectShape), the scoped C2 constraint
via CylindricalEnvelopingGrasp, full family- and leaf-level disjointness, the
eleven SWRL inference rules, and three example individuals for inspection in
Protege / HermiT.
"""

EXT = "http://example.org/ontograsp-extension#"
OG  = "http://example.org/ontograsp#"

# ---- taxonomy ----
POWER = ["LargeDiameter","SmallDiameter","MediumWrap","ThumbAdducted",
         "PowerSphere","Sphere3Finger","Sphere4Finger"]
PRECISION = ["Tripod","Quadpod"]
LEAVES = POWER + PRECISION

CYL_ENV = ["LargeDiameter","SmallDiameter","MediumWrap"]   # scoped C2 subfamily

SUPPORT_CLASSES = [
    "OppositionType","PalmOpposition","PadOpposition",
    "ThumbConfiguration","Abducted","Adducted",
    "FlexionLevel","LowFlexion","MediumFlexion","HighFlexion",
    "FlexionPattern","HighWrapPattern","SphericalPattern","TripodPattern",
    "ObjectShape",
    "ConstraintViolation",
]

# ---- helpers to build SWRL atoms ----
def cls_atom(cls_iri, arg):
    return f'''<rdf:Description>
  <rdf:type rdf:resource="http://www.w3.org/2003/11/swrl#ClassAtom"/>
  <swrl:classPredicate rdf:resource="{cls_iri}"/>
  <swrl:argument1 rdf:resource="{arg}"/>
</rdf:Description>'''

def prop_atom(prop_iri, a1, a2, a2_is_var=False):
    a2tag = f'<swrl:argument2 rdf:resource="{a2}"/>'
    return f'''<rdf:Description>
  <rdf:type rdf:resource="http://www.w3.org/2003/11/swrl#IndividualPropertyAtom"/>
  <swrl:propertyPredicate rdf:resource="{prop_iri}"/>
  <swrl:argument1 rdf:resource="{a1}"/>
  {a2tag}
</rdf:Description>'''

def atomlist(atoms):
    """Build nested rdf:first/rdf:rest AtomList from a list of atom XML strings."""
    if not atoms:
        return '<rdf:Description rdf:about="http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"/>'
    head = atoms[0]
    rest = atomlist(atoms[1:]) if len(atoms) > 1 else \
           '<rdf:rest rdf:resource="http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"/>'
    inner_rest = rest if len(atoms) > 1 else rest
    if len(atoms) > 1:
        rest_block = f"<rdf:rest>\n{atomlist(atoms[1:])}\n</rdf:rest>"
    else:
        rest_block = '<rdf:rest rdf:resource="http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"/>'
    return f'''<rdf:Description>
  <rdf:type rdf:resource="http://www.w3.org/2003/11/swrl#AtomList"/>
  <rdf:first>
{head}
  </rdf:first>
  {rest_block}
</rdf:Description>'''

def swrl_rule(name, body_atoms, head_atoms, comment=""):
    return f'''
<swrl:Imp rdf:about="{EXT}{name}">
  <swrla:isRuleEnabled rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</swrla:isRuleEnabled>
  <rdfs:comment xml:lang="en">{comment}</rdfs:comment>
  <swrl:body>
{atomlist(body_atoms)}
  </swrl:body>
  <swrl:head>
{atomlist(head_atoms)}
  </swrl:head>
</swrl:Imp>'''

# variables
g = EXT+"g"; p = EXT+"p"

def V(n): return EXT+n   # variable iri

# ============ build the rules ============
rules = []

# R1: HighWrapPattern detection (all four fingers HighFlexion)
rules.append(swrl_rule("R1_HighWrap",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasFlexionLevel_Index",  g, V("li")),
     cls_atom(EXT+"HighFlexion", V("li")),
     prop_atom(EXT+"hasFlexionLevel_Middle", g, V("lm")),
     cls_atom(EXT+"HighFlexion", V("lm")),
     prop_atom(EXT+"hasFlexionLevel_Ring",   g, V("lr")),
     cls_atom(EXT+"HighFlexion", V("lr")),
     prop_atom(EXT+"hasFlexionLevel_Little", g, V("ll")),
     cls_atom(EXT+"HighFlexion", V("ll"))],
    [prop_atom(EXT+"hasFlexionPattern", g, EXT+"HighWrapPattern_ind")],
    "R1: four-finger high flexion -> HighWrapPattern (cylindrical envelopment)"))

# R4: LargeDiameter  (cylinder + HighWrapPattern + Maximal)
rules.append(swrl_rule("R4_LargeDiameter",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Cylinder"),
     prop_atom(EXT+"hasFlexionPattern", g, p),
     cls_atom(EXT+"HighWrapPattern", p),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Maximal")],
    [cls_atom(EXT+"LargeDiameter", g)],
    "R4: cylinder + high wrap + maximal aperture -> LargeDiameter (PowerGrasp)"))

# R5: SmallDiameter (cylinder + HighWrapPattern + Minimal)
rules.append(swrl_rule("R5_SmallDiameter",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Cylinder"),
     prop_atom(EXT+"hasFlexionPattern", g, p),
     cls_atom(EXT+"HighWrapPattern", p),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Minimal")],
    [cls_atom(EXT+"SmallDiameter", g)],
    "R5: cylinder + high wrap + minimal aperture -> SmallDiameter (PowerGrasp)"))

# R6: MediumWrap (cylinder + index&middle non-low + Intermediate)
rules.append(swrl_rule("R6_MediumWrap",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Cylinder"),
     prop_atom(EXT+"hasFlexionLevel_Index",  g, V("li")),
     cls_atom(EXT+"MediumFlexion", V("li")),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Intermediate")],
    [cls_atom(EXT+"MediumWrap", g)],
    "R6: cylinder + medium index flexion + intermediate -> MediumWrap (PowerGrasp)"))

# R7: ThumbAdducted (cylinder + Adducted thumb + Minimal)
rules.append(swrl_rule("R7_ThumbAdducted",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Cylinder"),
     prop_atom(EXT+"hasThumbConfiguration", g, EXT+"Adducted_ind"),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Minimal"),
     prop_atom(EXT+"hasFlexionLevel_Ring", g, V("lr")),
     cls_atom(EXT+"LowFlexion", V("lr"))],
    [cls_atom(EXT+"ThumbAdducted", g)],
    "R7: cylinder + adducted thumb + minimal + ring extended -> ThumbAdducted (PowerGrasp)"))

# R2: SphericalPattern detection (sphere + three non-low fingers)
rules.append(swrl_rule("R2_Spherical",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Sphere"),
     prop_atom(EXT+"hasFlexionLevel_Index",  g, V("li")),
     cls_atom(EXT+"HighFlexion", V("li")),
     prop_atom(EXT+"hasFlexionLevel_Middle", g, V("lm")),
     cls_atom(EXT+"HighFlexion", V("lm")),
     prop_atom(EXT+"hasFlexionLevel_Ring",   g, V("lr")),
     cls_atom(EXT+"HighFlexion", V("lr"))],
    [prop_atom(EXT+"hasFlexionPattern", g, EXT+"SphericalPattern_ind")],
    "R2: sphere + 3 fingers high flexion -> SphericalPattern"))

# R8: PowerSphere (sphere + SphericalPattern + little non-low + Intermediate)
rules.append(swrl_rule("R8_PowerSphere",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Sphere"),
     prop_atom(EXT+"hasFlexionPattern", g, p),
     cls_atom(EXT+"SphericalPattern", p),
     prop_atom(EXT+"hasFlexionLevel_Little", g, V("ll")),
     cls_atom(EXT+"HighFlexion", V("ll")),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Intermediate")],
    [cls_atom(EXT+"PowerSphere", g)],
    "R8: sphere + spherical pattern + little flexed + intermediate -> PowerSphere"))

# R9: Sphere4Finger (sphere + SphericalPattern + little low + Intermediate)
rules.append(swrl_rule("R9_Sphere4Finger",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Sphere"),
     prop_atom(EXT+"hasFlexionPattern", g, p),
     cls_atom(EXT+"SphericalPattern", p),
     prop_atom(EXT+"hasFlexionLevel_Little", g, V("ll")),
     cls_atom(EXT+"LowFlexion", V("ll")),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Intermediate")],
    [cls_atom(EXT+"Sphere4Finger", g)],
    "R9: sphere + spherical pattern + little extended + intermediate -> Sphere4Finger"))

# R10: Sphere3Finger (sphere + index&middle non-low + ring low + Intermediate)
rules.append(swrl_rule("R10_Sphere3Finger",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Sphere"),
     prop_atom(EXT+"hasFlexionLevel_Index",  g, V("li")),
     cls_atom(EXT+"HighFlexion", V("li")),
     prop_atom(EXT+"hasFlexionLevel_Ring", g, V("lr")),
     cls_atom(EXT+"LowFlexion", V("lr")),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Intermediate")],
    [cls_atom(EXT+"Sphere3Finger", g)],
    "R10: sphere + index flexed + ring extended + intermediate -> Sphere3Finger"))

# R3: TripodPattern detection
rules.append(swrl_rule("R3_TripodPattern",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasFlexionLevel_Index",  g, V("li")),
     cls_atom(EXT+"MediumFlexion", V("li")),
     prop_atom(EXT+"hasFlexionLevel_Middle", g, V("lm")),
     cls_atom(EXT+"MediumFlexion", V("lm")),
     prop_atom(EXT+"hasFlexionLevel_Ring",   g, V("lr")),
     cls_atom(EXT+"LowFlexion", V("lr")),
     prop_atom(EXT+"hasFlexionLevel_Little", g, V("ll")),
     cls_atom(EXT+"LowFlexion", V("ll"))],
    [prop_atom(EXT+"hasFlexionPattern", g, EXT+"TripodPattern_ind")],
    "R3: index+middle medium, ring+little low -> TripodPattern (fingertip)"))

# R11: Tripod (sphere + TripodPattern + Minimal) -> PrecisionGrasp
rules.append(swrl_rule("R11_Tripod",
    [cls_atom(OG+"Grasp", g),
     prop_atom(EXT+"hasObjectShape", g, EXT+"Sphere"),
     prop_atom(EXT+"hasFlexionPattern", g, p),
     cls_atom(EXT+"TripodPattern", p),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Minimal")],
    [cls_atom(EXT+"Tripod", g)],
    "R11: sphere + tripod pattern + minimal -> Tripod (PrecisionGrasp)"))

# C2 constraint: CylindricalEnvelopingGrasp + Minimal -> ConstraintViolation
rules.append(swrl_rule("C2_Constraint",
    [cls_atom(EXT+"CylindricalEnvelopingGrasp", g),
     prop_atom(EXT+"requiresGripAperture", g, OG+"Minimal")],
    [cls_atom(EXT+"ConstraintViolation", g)],
    "C2: cylindrical enveloping grasp under minimal aperture -> ConstraintViolation"))

# ============ class declarations ============
def class_decl(name, parent=None, comment=""):
    sub = f'\n  <rdfs:subClassOf rdf:resource="{parent}"/>' if parent else ""
    com = f'\n  <rdfs:comment xml:lang="en">{comment}</rdfs:comment>' if comment else ""
    return f'''<owl:Class rdf:about="{EXT}{name}">{sub}{com}
  <rdfs:label xml:lang="en">{name}</rdfs:label>
</owl:Class>'''

classes = []
classes.append(class_decl("PowerGrasp", OG+"Grasp", "Power (whole-hand / palmar enveloping) grasps."))
classes.append(class_decl("PrecisionGrasp", OG+"Grasp", "Precision (fingertip / pad) grasps."))
for c in POWER:
    classes.append(class_decl(c, EXT+"PowerGrasp"))
for c in PRECISION:
    classes.append(class_decl(c, EXT+"PrecisionGrasp"))

# CylindricalEnvelopingGrasp = LargeDiameter OR SmallDiameter OR MediumWrap (for scoped C2)
cyl_union = "".join(f'<rdf:Description rdf:about="{EXT}{c}"/>' for c in CYL_ENV)
classes.append(f'''<owl:Class rdf:about="{EXT}CylindricalEnvelopingGrasp">
  <owl:equivalentClass>
    <owl:Class>
      <owl:unionOf rdf:parseType="Collection">
        {cyl_union}
      </owl:unionOf>
    </owl:Class>
  </owl:equivalentClass>
  <rdfs:comment xml:lang="en">Union of the large-span cylindrical power grasps; scope of constraint C2.</rdfs:comment>
  <rdfs:label xml:lang="en">CylindricalEnvelopingGrasp</rdfs:label>
</owl:Class>''')

# FlexedLevel = MediumFlexion OR HighFlexion  (convenience class for "not low")
classes.append(f'''<owl:Class rdf:about="{EXT}FlexedLevel">
  <owl:equivalentClass>
    <owl:Class>
      <owl:unionOf rdf:parseType="Collection">
        <rdf:Description rdf:about="{EXT}MediumFlexion"/>
        <rdf:Description rdf:about="{EXT}HighFlexion"/>
      </owl:unionOf>
    </owl:Class>
  </owl:equivalentClass>
  <rdfs:comment xml:lang="en">A flexion level that is not low (medium or high); used by rules to express that a finger is flexed.</rdfs:comment>
  <rdfs:label xml:lang="en">FlexedLevel</rdfs:label>
</owl:Class>''')

# supporting classes
sup_parent = {
  "PalmOpposition":"OppositionType","PadOpposition":"OppositionType",
  "Abducted":"ThumbConfiguration","Adducted":"ThumbConfiguration",
  "LowFlexion":"FlexionLevel","MediumFlexion":"FlexionLevel","HighFlexion":"FlexionLevel",
  "HighWrapPattern":"FlexionPattern","SphericalPattern":"FlexionPattern","TripodPattern":"FlexionPattern",
}
for c in SUPPORT_CLASSES:
    parent = EXT+sup_parent[c] if c in sup_parent else None
    classes.append(class_decl(c, parent))

# ============ object properties ============
OPROPS = ["hasOppositionType","hasThumbConfiguration","requiresGripAperture",
          "hasObjectShape","hasFlexionPattern",
          "hasFlexionLevel_Index","hasFlexionLevel_Middle",
          "hasFlexionLevel_Ring","hasFlexionLevel_Little","hasFlexionLevel_Thumb"]
oprops = "\n".join(
    f'<owl:ObjectProperty rdf:about="{EXT}{p}"><rdfs:domain rdf:resource="{OG}Grasp"/></owl:ObjectProperty>'
    for p in OPROPS)

# ============ disjointness ============
# family-level
disjoint = [f'''<owl:AllDisjointClasses>
  <owl:members rdf:parseType="Collection">
    <rdf:Description rdf:about="{EXT}PowerGrasp"/>
    <rdf:Description rdf:about="{EXT}PrecisionGrasp"/>
  </owl:members>
</owl:AllDisjointClasses>''']
# leaf-level (all 9 mutually disjoint)
leaf_members = "".join(f'<rdf:Description rdf:about="{EXT}{c}"/>' for c in LEAVES)
disjoint.append(f'''<owl:AllDisjointClasses>
  <owl:members rdf:parseType="Collection">
    {leaf_members}
  </owl:members>
</owl:AllDisjointClasses>''')
# supporting disjoint sets
for grp in [["PalmOpposition","PadOpposition"],
            ["Abducted","Adducted"],
            ["LowFlexion","MediumFlexion","HighFlexion"],
            ["HighWrapPattern","SphericalPattern","TripodPattern"]]:
    mem = "".join(f'<rdf:Description rdf:about="{EXT}{c}"/>' for c in grp)
    disjoint.append(f'''<owl:AllDisjointClasses>
  <owl:members rdf:parseType="Collection">{mem}</owl:members>
</owl:AllDisjointClasses>''')

# ============ named individuals ============
def indiv(name, types=None, props=None):
    types = types or []; props = props or []
    t = "".join(f'\n  <rdf:type rdf:resource="{ti}"/>' for ti in types)
    pr = "".join(f'\n  <{pn} rdf:resource="{pv}"/>' for pn,pv in props)
    return f'<owl:NamedIndividual rdf:about="{EXT}{name}">{t}{pr}\n</owl:NamedIndividual>'

individuals = []
# pattern + descriptor individuals (typed instances of pattern classes)
individuals.append(indiv("HighWrapPattern_ind", [EXT+"HighWrapPattern"]))
individuals.append(indiv("SphericalPattern_ind", [EXT+"SphericalPattern"]))
individuals.append(indiv("TripodPattern_ind", [EXT+"TripodPattern"]))
individuals.append(indiv("Adducted_ind", [EXT+"Adducted"]))
individuals.append(indiv("Cylinder", [EXT+"ObjectShape"]))
individuals.append(indiv("Sphere", [EXT+"ObjectShape"]))
# flexion level individuals
individuals.append(indiv("Low_ind",[EXT+"LowFlexion"]))
individuals.append(indiv("Medium_ind",[EXT+"MediumFlexion"]))
individuals.append(indiv("High_ind",[EXT+"HighFlexion"]))

# --- Example 1: valid precision grasp (Tripod) ---
individuals.append(indiv("Grasp_Precision_OK",
    [OG+"Grasp"],
    [("hasObjectShape", EXT+"Sphere"),
     ("hasFlexionLevel_Index", EXT+"Medium_ind"),
     ("hasFlexionLevel_Middle", EXT+"Medium_ind"),
     ("hasFlexionLevel_Ring", EXT+"Low_ind"),
     ("hasFlexionLevel_Little", EXT+"Low_ind"),
     ("hasFlexionPattern", EXT+"TripodPattern_ind"),
     ("requiresGripAperture", OG+"Minimal")]))

# --- Example 2: C2 violation (cylindrical enveloping under Minimal) ---
# Declared as MediumWrap (a CylindricalEnvelopingGrasp) under Minimal aperture.
individuals.append(indiv("Grasp_Power_C2",
    [OG+"Grasp", EXT+"MediumWrap"],
    [("hasObjectShape", EXT+"Cylinder"),
     ("hasFlexionPattern", EXT+"HighWrapPattern_ind"),
     ("requiresGripAperture", OG+"Minimal")]))

# --- Example 3: contradiction (both PowerGrasp and PrecisionGrasp) ---
individuals.append(indiv("Grasp_Contradict",
    [OG+"Grasp", EXT+"PowerGrasp", EXT+"PrecisionGrasp"],
    [("hasObjectShape", EXT+"Cylinder"),
     ("requiresGripAperture", OG+"Maximal")]))

# grip aperture individuals are from base og: namespace; declare them so Protege sees types
individuals.append(indiv2 := f'<owl:NamedIndividual rdf:about="{OG}Minimal"/>')
individuals.append(f'<owl:NamedIndividual rdf:about="{OG}Intermediate"/>')
individuals.append(f'<owl:NamedIndividual rdf:about="{OG}Maximal"/>')

# ============ assemble document ============
doc = f'''<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ontograsp-extension#"
     xml:base="http://example.org/ontograsp-extension"
     xmlns:og="http://example.org/ontograsp#"
     xmlns:ogx="http://example.org/ontograsp-extension#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:swrl="http://www.w3.org/2003/11/swrl#"
     xmlns:swrla="http://swrl.stanford.edu/ontologies/3.3/swrla.owl#"
     xmlns:swrlb="http://www.w3.org/2003/11/swrlb#">

  <owl:Ontology rdf:about="http://example.org/ontograsp-extension">
    <rdfs:comment xml:lang="en">GRACE ontology: OntoGrasp extension with a biomechanically grounded seven-power / two-precision grasp taxonomy, supporting descriptor classes, SWRL rules (R1-R11), and the scoped constraint C2.</rdfs:comment>
    <rdfs:label xml:lang="en">OntoGrasp Extension (GRACE) for Rule-Based Grasp Reasoning</rdfs:label>
  </owl:Ontology>

  <!-- base-namespace anchor class -->
  <owl:Class rdf:about="{OG}Grasp"/>
  <owl:Class rdf:about="{OG}GripAperture"/>

  <!-- ============ CLASSES ============ -->
  {chr(10).join(classes)}

  <!-- ============ OBJECT PROPERTIES ============ -->
  {oprops}

  <!-- ============ DISJOINTNESS ============ -->
  {chr(10).join(disjoint)}

  <!-- ============ INDIVIDUALS ============ -->
  {chr(10).join(individuals)}

  <!-- ============ SWRL RULES (R1-R11 + C2) ============ -->
  {chr(10).join(rules)}

</rdf:RDF>
'''

open("OntoGrasp-ext-GRACE.owl","w",encoding="utf-8").write(doc)
print("Wrote OntoGrasp-ext-GRACE.owl")
print("Lines:", doc.count(chr(10)))
print("Rules:", len(rules))
print("Leaf classes:", len(LEAVES))
