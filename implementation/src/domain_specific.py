"""
Domain-Specific Diversity Analysis.

Implements diversity analysis tailored to professional domains including
legal, medical, financial, scientific, and engineering contexts.  Each
domain function applies curated keyword/phrase lists, categorises matched
terms, and computes entropy-based diversity scores over the resulting
distributions.

Helper classes provide reusable machinery: terminology matching, a generic
analysis framework, and hierarchical taxonomy coverage scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalised_entropy(counter: Counter) -> float:
    """Return Shannon entropy normalised to [0, 1].

    A uniform distribution yields 1.0; a single-category distribution
    yields 0.0.  Returns 0.0 when *counter* is empty.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    n_categories = len(counter)
    if n_categories <= 1:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = math.log2(n_categories)
    if max_entropy == 0.0:
        return 0.0
    return min(entropy / max_entropy, 1.0)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _pairwise_dissimilarity(sets: List[Set[str]]) -> float:
    """Average pairwise Jaccard *distance* across a list of sets."""
    if len(sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            total += 1.0 - _jaccard(sets[i], sets[j])
            count += 1
    return total / count


def _lower_text(text: str) -> str:
    return text.lower()


# ---------------------------------------------------------------------------
# Data-classes for domain reports
# ---------------------------------------------------------------------------


@dataclass
class LegalDiversityReport:
    """Report on diversity within a collection of legal texts."""

    argument_types: Dict[str, int] = field(default_factory=dict)
    jurisdiction_coverage: Dict[str, int] = field(default_factory=dict)
    precedent_diversity: float = 0.0
    reasoning_diversity: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicalDiversityReport:
    """Report on diversity within a collection of medical texts."""

    diagnosis_breadth: Dict[str, int] = field(default_factory=dict)
    treatment_diversity: Dict[str, int] = field(default_factory=dict)
    evidence_levels: Dict[str, int] = field(default_factory=dict)
    specialty_coverage: Dict[str, int] = field(default_factory=dict)
    risk_coverage: Dict[str, int] = field(default_factory=dict)
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinancialDiversityReport:
    """Report on diversity within a collection of financial analyses."""

    market_views: Dict[str, int] = field(default_factory=dict)
    sector_coverage: Dict[str, int] = field(default_factory=dict)
    time_horizon_diversity: Dict[str, int] = field(default_factory=dict)
    risk_perspective_diversity: Dict[str, int] = field(default_factory=dict)
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientificDiversityReport:
    """Report on diversity within scientific hypotheses or papers."""

    hypothesis_diversity: float = 0.0
    methodology_diversity: Dict[str, int] = field(default_factory=dict)
    field_coverage: Dict[str, int] = field(default_factory=dict)
    evidence_types: Dict[str, int] = field(default_factory=dict)
    novelty_score: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineeringDiversityReport:
    """Report on diversity within engineering design proposals."""

    design_approach_diversity: Dict[str, int] = field(default_factory=dict)
    material_diversity: Dict[str, int] = field(default_factory=dict)
    constraint_coverage: Dict[str, int] = field(default_factory=dict)
    trade_off_coverage: Dict[str, int] = field(default_factory=dict)
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TaxonomyTree — hierarchical concept taxonomy
# ---------------------------------------------------------------------------


class TaxonomyTree:
    """Hierarchical taxonomy for domain concepts.

    Nodes are strings.  Each node may have zero or more children.
    Coverage is computed as the fraction of leaf nodes matched,
    optionally weighted by depth.
    """

    def __init__(self) -> None:
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parent: Dict[str, Optional[str]] = {}

    # -- mutation --------------------------------------------------------

    def add_node(self, parent: Optional[str], child: str) -> None:
        """Add *child* under *parent*.  If *parent* is ``None`` the child
        becomes a root node."""
        if parent is not None:
            if child not in self._children[parent]:
                self._children[parent].append(child)
        self._parent[child] = parent
        if child not in self._children:
            self._children[child] = []

    # -- queries ---------------------------------------------------------

    def _all_nodes(self) -> Set[str]:
        nodes: Set[str] = set()
        for k, vs in self._children.items():
            nodes.add(k)
            nodes.update(vs)
        return nodes

    def _leaves(self, root: Optional[str] = None) -> Set[str]:
        """Return leaf nodes under *root* (or globally when *root* is None)."""
        if root is not None:
            subtree = self.get_subtree(root)
        else:
            subtree = self._all_nodes()
        return {n for n in subtree if not self._children.get(n)}

    def get_subtree(self, node: str) -> Set[str]:
        """Return all nodes in the subtree rooted at *node* (inclusive)."""
        result: Set[str] = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            for child in self._children.get(current, []):
                if child not in result:
                    result.add(child)
                    stack.append(child)
        return result

    def _depth(self, node: str) -> int:
        depth = 0
        cur: Optional[str] = node
        while cur is not None:
            par = self._parent.get(cur)
            if par is None:
                break
            depth += 1
            cur = par
        return depth

    def coverage(self, matched_nodes: Set[str]) -> float:
        """Fraction of leaf nodes that are contained in *matched_nodes*."""
        leaves = self._leaves()
        if not leaves:
            return 0.0
        return len(matched_nodes & leaves) / len(leaves)

    def depth_weighted_coverage(self, matched_nodes: Set[str]) -> float:
        """Coverage weighted by node depth (deeper = higher weight)."""
        leaves = self._leaves()
        if not leaves:
            return 0.0
        total_weight = 0.0
        matched_weight = 0.0
        for leaf in leaves:
            w = 1.0 + self._depth(leaf)
            total_weight += w
            if leaf in matched_nodes:
                matched_weight += w
        if total_weight == 0.0:
            return 0.0
        return matched_weight / total_weight


# ---------------------------------------------------------------------------
# DomainTerminologyMatcher
# ---------------------------------------------------------------------------

# Curated keyword lists keyed by ``(domain, category)``.
_TERMINOLOGY: Dict[Tuple[str, str], List[str]] = {
    # -- legal -----------------------------------------------------------
    ("legal", "statutory"): [
        "statute", "legislation", "enacted", "codified", "statutory provision",
        "legislative intent", "code section", "public law", "act of congress",
        "regulatory framework", "ordinance", "municipal code", "federal register",
        "rule-making", "statutory interpretation", "legislative history",
        "enabling statute", "sunset clause", "preemption", "delegation doctrine",
        "administrative rule", "promulgated", "gazetted", "bylaw",
    ],
    ("legal", "constitutional"): [
        "constitution", "amendment", "due process", "equal protection",
        "first amendment", "fourth amendment", "fifth amendment", "fourteenth amendment",
        "bill of rights", "constitutional right", "fundamental right",
        "strict scrutiny", "rational basis", "intermediate scrutiny",
        "separation of powers", "federalism", "supremacy clause",
        "commerce clause", "dormant commerce clause", "privileges and immunities",
        "substantive due process", "procedural due process", "incorporation doctrine",
        "judicial review", "constitutional interpretation",
    ],
    ("legal", "precedent-based"): [
        "stare decisis", "precedent", "holding", "ratio decidendi",
        "obiter dictum", "distinguishable", "overruled", "affirmed",
        "reversed", "binding authority", "persuasive authority",
        "landmark case", "case law", "common law", "judicial opinion",
        "concurring opinion", "dissenting opinion", "plurality opinion",
        "en banc", "certiorari", "appellate review", "trial court",
    ],
    ("legal", "policy"): [
        "public policy", "policy rationale", "societal interest",
        "balancing test", "cost-benefit", "deterrence", "efficiency",
        "fairness", "equity", "justice", "welfare", "public interest",
        "regulatory purpose", "legislative purpose", "social contract",
        "proportionality", "necessity", "least restrictive means",
        "compelling interest", "legitimate purpose", "instrumentalism",
    ],
    ("legal", "procedural"): [
        "standing", "jurisdiction", "mootness", "ripeness",
        "statute of limitations", "burden of proof", "standard of review",
        "motion to dismiss", "summary judgment", "discovery", "pleading",
        "class action", "intervention", "joinder", "venue", "forum selection",
        "res judicata", "collateral estoppel", "exhaustion of remedies",
        "injunction", "preliminary injunction", "temporary restraining order",
    ],
    # -- legal reasoning patterns ----------------------------------------
    ("legal", "deductive"): [
        "therefore", "it follows that", "syllogism", "deductive",
        "logically", "necessarily", "by definition", "axiomatically",
        "if and only if", "formal logic", "modus ponens", "conclusion follows",
        "given that", "since", "inasmuch as", "consequently",
        "deductively", "from the premise", "inferring", "entails",
    ],
    ("legal", "analogical"): [
        "analogous", "similarly", "by analogy", "comparable",
        "parallel", "akin to", "resembles", "like the case of",
        "on all fours", "factually similar", "reasoning by analogy",
        "same rationale", "extending the logic", "applying the same principle",
        "mutatis mutandis", "in pari materia", "pari passu",
        "draws a parallel", "comparable circumstances", "analogical reasoning",
    ],
    ("legal", "policy-based"): [
        "policy consideration", "weighing interests", "societal impact",
        "pragmatic", "practical consequence", "real-world effect",
        "institutional competence", "administrability", "rule of law",
        "chilling effect", "slippery slope", "flood of litigation",
        "bright-line rule", "totality of circumstances", "case-by-case",
        "standard vs rule", "over-inclusive", "under-inclusive",
        "policy-driven", "consequentialist", "teleological",
    ],
    # -- legal domains ---------------------------------------------------
    ("legal", "criminal"): [
        "criminal", "felony", "misdemeanor", "indictment", "prosecution",
        "defendant", "beyond reasonable doubt", "plea", "sentencing",
        "incarceration", "probation", "parole", "mens rea", "actus reus",
        "homicide", "assault", "fraud", "conspiracy", "accomplice",
        "grand jury", "arraignment", "bail",
    ],
    ("legal", "civil"): [
        "civil", "plaintiff", "tort", "breach of contract", "negligence",
        "damages", "compensatory", "punitive damages", "liability",
        "fiduciary duty", "duty of care", "proximate cause", "preponderance",
        "restitution", "unjust enrichment", "specific performance",
        "civil procedure", "complaint", "answer", "counterclaim",
        "cross-claim", "interpleader",
    ],
    ("legal", "administrative"): [
        "administrative", "agency", "rulemaking", "adjudication",
        "chevron deference", "arbitrary and capricious", "notice and comment",
        "administrative hearing", "regulatory", "compliance", "enforcement",
        "permit", "license", "inspection", "administrative law judge",
        "final agency action", "exhaustion", "primary jurisdiction",
        "delegation", "non-delegation doctrine", "auer deference",
    ],
    # -- legal jurisdictions ---------------------------------------------
    ("legal", "federal"): [
        "federal", "united states", "u.s. supreme court", "circuit court",
        "district court", "federal question", "diversity jurisdiction",
        "federal statute", "cfr", "usc", "federal regulation",
        "interstate", "federal agency", "federal government",
        "congressional", "senate", "house of representatives",
        "executive order", "federal register", "code of federal regulations",
    ],
    ("legal", "state"): [
        "state law", "state court", "state constitution", "state statute",
        "state regulation", "governor", "state legislature", "county",
        "municipal", "local ordinance", "state agency", "state supreme court",
        "court of appeals", "trial court", "state bar", "uniform state law",
        "intrastate", "police power", "zoning", "state administrative code",
    ],
    ("legal", "international"): [
        "international law", "treaty", "convention", "united nations",
        "international court", "hague", "geneva convention", "extradition",
        "sovereign immunity", "comity", "customary international law",
        "jus cogens", "bilateral", "multilateral", "international tribunal",
        "arbitration", "world trade organization", "wto", "nafta",
        "european union", "eu directive", "eu regulation",
    ],
    # -- medical ---------------------------------------------------------
    ("medical", "cardiovascular"): [
        "cardiac", "heart", "cardiovascular", "myocardial", "arrhythmia",
        "hypertension", "atherosclerosis", "angina", "coronary",
        "aortic", "valvular", "cardiomyopathy", "heart failure",
        "atrial fibrillation", "ventricular", "pericarditis",
        "endocarditis", "aneurysm", "thrombosis", "embolism",
        "echocardiogram", "electrocardiogram", "stent", "bypass",
    ],
    ("medical", "neurological"): [
        "neurological", "brain", "neurology", "stroke", "seizure",
        "epilepsy", "parkinson", "alzheimer", "dementia", "neuropathy",
        "multiple sclerosis", "meningitis", "encephalitis", "migraine",
        "headache", "concussion", "traumatic brain injury", "spinal cord",
        "neurodegenerative", "cerebral", "cranial", "neurotransmitter",
        "dopamine", "serotonin",
    ],
    ("medical", "respiratory"): [
        "respiratory", "pulmonary", "lung", "asthma", "copd",
        "pneumonia", "bronchitis", "tuberculosis", "pleural",
        "emphysema", "bronchoscopy", "ventilator", "intubation",
        "oxygen therapy", "spirometry", "cystic fibrosis", "pulmonary embolism",
        "dyspnea", "tachypnea", "respiratory failure", "ards",
        "pulmonary fibrosis", "sleep apnea", "tracheostomy",
    ],
    ("medical", "musculoskeletal"): [
        "orthopedic", "fracture", "osteoporosis", "arthritis",
        "joint", "bone", "tendon", "ligament", "cartilage",
        "spine", "scoliosis", "herniated disc", "musculoskeletal",
        "rheumatoid", "osteoarthritis", "fibromyalgia", "gout",
        "bursitis", "tendinitis", "rotator cuff", "meniscus",
        "prosthesis", "arthroplasty", "osteomyelitis",
    ],
    ("medical", "gastrointestinal"): [
        "gastrointestinal", "liver", "hepatic", "gastric", "intestinal",
        "colon", "pancreatic", "esophageal", "gallbladder", "appendicitis",
        "crohn", "ulcerative colitis", "celiac", "cirrhosis", "hepatitis",
        "pancreatitis", "gastritis", "peptic ulcer", "diverticulitis",
        "colonoscopy", "endoscopy", "bowel", "gi tract", "ibs",
    ],
    ("medical", "endocrine"): [
        "endocrine", "diabetes", "thyroid", "adrenal", "pituitary",
        "insulin", "glucose", "metabolic", "hormonal", "hypothyroidism",
        "hyperthyroidism", "cushing", "addison", "growth hormone",
        "testosterone", "estrogen", "cortisol", "prolactin",
        "polycystic ovary", "diabetic ketoacidosis", "hyperglycemia",
        "hypoglycemia", "glycated hemoglobin", "hemoglobin a1c",
    ],
    ("medical", "pharmaceutical"): [
        "medication", "drug", "prescription", "dosage", "pharmaceutical",
        "pharmacology", "pharmacokinetics", "adverse effect", "side effect",
        "contraindication", "interaction", "monotherapy", "combination therapy",
        "antibiotic", "antiviral", "analgesic", "antipyretic",
        "anti-inflammatory", "statin", "beta blocker", "ace inhibitor",
        "immunosuppressant", "chemotherapy", "biologic",
    ],
    ("medical", "surgical"): [
        "surgery", "surgical", "incision", "excision", "resection",
        "transplant", "graft", "laparoscopic", "minimally invasive",
        "open surgery", "robotic surgery", "biopsy", "debridement",
        "anastomosis", "catheter", "suture", "drainage",
        "anesthesia", "postoperative", "preoperative", "intraoperative",
        "sterile", "operating room", "surgical site",
    ],
    ("medical", "therapeutic"): [
        "therapy", "rehabilitation", "physical therapy", "occupational therapy",
        "cognitive behavioral therapy", "psychotherapy", "radiation therapy",
        "dialysis", "transfusion", "infusion", "immunotherapy",
        "phototherapy", "speech therapy", "behavioral therapy",
        "gene therapy", "stem cell therapy", "palliative care",
        "supportive care", "counseling", "intervention",
        "lifestyle modification", "diet", "exercise program",
    ],
    ("medical", "preventive"): [
        "prevention", "screening", "vaccination", "immunization",
        "prophylaxis", "early detection", "risk factor", "lifestyle",
        "preventive care", "health promotion", "public health",
        "epidemiology", "surveillance", "quarantine", "contact tracing",
        "wellness", "check-up", "mammography", "colonoscopy screening",
        "pap smear", "blood pressure screening", "cholesterol screening",
    ],
    ("medical", "meta-analysis"): [
        "meta-analysis", "systematic review", "pooled analysis",
        "forest plot", "heterogeneity", "publication bias",
        "effect size", "confidence interval", "funnel plot",
        "cochrane", "prisma", "risk of bias", "sensitivity analysis",
        "subgroup analysis", "fixed effect", "random effect",
        "quality assessment", "grade", "evidence synthesis",
        "quantitative synthesis", "narrative synthesis",
    ],
    ("medical", "rct"): [
        "randomized controlled trial", "rct", "randomization",
        "double-blind", "placebo-controlled", "single-blind",
        "control group", "treatment group", "intention to treat",
        "per protocol", "crossover", "parallel group",
        "block randomization", "stratified randomization",
        "allocation concealment", "blinding", "washout period",
        "primary endpoint", "secondary endpoint", "interim analysis",
    ],
    ("medical", "cohort"): [
        "cohort study", "prospective", "retrospective", "longitudinal",
        "follow-up", "incidence", "prevalence", "relative risk",
        "hazard ratio", "survival analysis", "kaplan-meier",
        "cox regression", "person-years", "exposure", "outcome",
        "confounder", "adjustment", "propensity score",
        "observational study", "registry", "database study",
    ],
    ("medical", "case_study"): [
        "case report", "case series", "case study", "clinical presentation",
        "patient history", "chief complaint", "differential diagnosis",
        "clinical findings", "imaging findings", "laboratory findings",
        "pathology report", "hospital course", "discharge summary",
        "follow-up visit", "clinical outcome", "unusual presentation",
        "rare condition", "atypical", "anecdotal", "single patient",
    ],
    ("medical", "expert_opinion"): [
        "expert opinion", "clinical experience", "expert consensus",
        "guideline", "recommendation", "best practice", "clinical judgment",
        "standard of care", "clinical expertise", "professional opinion",
        "consensus statement", "position paper", "white paper",
        "editorial", "commentary", "perspective", "narrative review",
        "clinical reasoning", "expert panel", "delphi method",
    ],
    # -- financial -------------------------------------------------------
    ("financial", "bullish"): [
        "bullish", "upside", "growth", "rally", "outperform",
        "buy", "accumulate", "overweight", "positive outlook",
        "uptrend", "breakout", "momentum", "expansion",
        "recovery", "boom", "appreciation", "strong buy",
        "target price increase", "upgrade", "optimistic",
        "bullish divergence", "golden cross", "bull market",
    ],
    ("financial", "bearish"): [
        "bearish", "downside", "decline", "sell", "underperform",
        "underweight", "negative outlook", "downtrend", "correction",
        "recession", "contraction", "depreciation", "headwind",
        "risk-off", "strong sell", "target price decrease", "downgrade",
        "pessimistic", "death cross", "bear market", "capitulation",
        "bearish divergence", "breakdown",
    ],
    ("financial", "neutral"): [
        "neutral", "hold", "market perform", "equal weight",
        "range-bound", "consolidation", "sideways", "mixed signals",
        "balanced", "wait and see", "fair value", "appropriately valued",
        "in line", "market weight", "sector perform", "peer perform",
        "stable outlook", "maintain", "unchanged", "consensus",
    ],
    ("financial", "technology"): [
        "technology", "tech", "software", "semiconductor", "saas",
        "cloud computing", "artificial intelligence", "machine learning",
        "cybersecurity", "fintech", "e-commerce", "digital transformation",
        "tech sector", "nasdaq", "platform", "api", "data center",
        "internet of things", "blockchain", "quantum computing",
    ],
    ("financial", "healthcare"): [
        "healthcare", "biotech", "pharmaceutical", "medical device",
        "health insurance", "hospital", "clinical trial", "fda approval",
        "drug pipeline", "generic drug", "patent cliff", "biosimilar",
        "telemedicine", "digital health", "health sector", "medicare",
        "medicaid", "managed care", "health services",
        "pharmaceutical sector", "biotech index",
    ],
    ("financial", "energy"): [
        "energy", "oil", "natural gas", "petroleum", "crude",
        "renewable energy", "solar", "wind power", "utilities",
        "pipeline", "refinery", "opec", "drilling", "upstream",
        "downstream", "midstream", "lng", "fossil fuel",
        "clean energy", "energy transition", "carbon credit",
        "nuclear energy", "geothermal", "hydroelectric",
    ],
    ("financial", "financial_sector"): [
        "banking", "bank", "insurance", "asset management",
        "investment banking", "retail banking", "commercial banking",
        "mortgage", "lending", "credit", "interest rate",
        "net interest margin", "non-performing loan", "capital adequacy",
        "stress test", "financial services", "wealth management",
        "brokerage", "custodian", "payment processing",
    ],
    ("financial", "consumer"): [
        "consumer", "retail", "consumer discretionary", "consumer staples",
        "luxury", "fast-moving consumer goods", "fmcg", "brand",
        "e-commerce", "direct-to-consumer", "franchise",
        "consumer sentiment", "consumer spending", "disposable income",
        "consumer confidence", "same-store sales", "footfall",
        "omnichannel", "private label", "consumer electronics",
    ],
    ("financial", "industrials"): [
        "industrial", "manufacturing", "aerospace", "defense",
        "construction", "infrastructure", "transportation", "logistics",
        "supply chain", "capital goods", "machinery", "automation",
        "robotics", "industrials sector", "backlog", "capacity utilization",
        "factory output", "purchasing managers index", "pmi",
        "industrial production", "heavy equipment",
    ],
    ("financial", "real_estate"): [
        "real estate", "reit", "property", "commercial real estate",
        "residential", "vacancy rate", "cap rate", "occupancy",
        "rental income", "mortgage rate", "housing market",
        "property value", "land use", "development", "zoning",
        "landlord", "tenant", "lease", "square footage",
        "real estate investment trust", "property management",
    ],
    ("financial", "short_term"): [
        "short-term", "day trading", "intraday", "weekly",
        "near-term", "swing trade", "scalp", "overnight",
        "short-dated", "front-month", "spot", "immediate",
        "quick", "rapid", "fast", "next quarter",
        "current quarter", "tactical", "opportunistic", "momentum trade",
    ],
    ("financial", "medium_term"): [
        "medium-term", "quarterly", "semi-annual", "six-month",
        "one-year", "annual", "fiscal year", "cyclical",
        "intermediate", "mid-cycle", "next year", "twelve-month",
        "year-ahead", "forward-looking", "rolling", "calendar year",
        "medium-dated", "strategic allocation", "position trade",
        "medium horizon", "two-year",
    ],
    ("financial", "long_term"): [
        "long-term", "multi-year", "decade", "secular",
        "structural", "generational", "buy and hold", "compounding",
        "retirement", "pension", "endowment", "long-dated",
        "five-year", "ten-year", "thirty-year", "perpetual",
        "strategic", "thematic", "macro trend", "demographic shift",
        "long-horizon", "patient capital",
    ],
    ("financial", "fundamental"): [
        "fundamental analysis", "earnings", "revenue", "profit margin",
        "price-to-earnings", "p/e ratio", "book value", "cash flow",
        "discounted cash flow", "dcf", "intrinsic value", "valuation",
        "balance sheet", "income statement", "financial statement",
        "dividend", "free cash flow", "return on equity", "roe",
        "earnings per share", "eps", "debt-to-equity",
    ],
    ("financial", "technical"): [
        "technical analysis", "moving average", "rsi",
        "relative strength", "macd", "bollinger band", "support",
        "resistance", "trend line", "chart pattern", "candlestick",
        "fibonacci", "volume", "stochastic", "oscillator",
        "breakout", "pullback", "head and shoulders", "double top",
        "double bottom", "elliott wave", "ichimoku",
    ],
    ("financial", "quantitative"): [
        "quantitative", "algorithmic", "quant", "factor model",
        "regression", "monte carlo", "backtesting", "alpha",
        "beta", "sharpe ratio", "sortino ratio", "var",
        "value at risk", "optimization", "portfolio theory",
        "mean-variance", "risk-adjusted", "correlation matrix",
        "covariance", "statistical arbitrage", "systematic",
    ],
    ("financial", "macro"): [
        "macroeconomic", "gdp", "inflation", "interest rate",
        "central bank", "federal reserve", "monetary policy",
        "fiscal policy", "unemployment", "trade balance",
        "current account", "geopolitical", "global macro",
        "yield curve", "bond market", "currency", "forex",
        "sovereign debt", "quantitative easing", "tapering",
        "rate hike", "rate cut",
    ],
    # -- scientific ------------------------------------------------------
    ("scientific", "experimental"): [
        "experiment", "experimental", "controlled experiment",
        "laboratory", "lab", "bench", "wet lab", "dry lab",
        "clinical trial", "field experiment", "randomized",
        "treatment group", "control group", "variable",
        "independent variable", "dependent variable", "manipulation",
        "measurement", "protocol", "replication", "reproducibility",
        "sample", "specimen", "apparatus",
    ],
    ("scientific", "observational"): [
        "observational", "observation", "survey", "field study",
        "naturalistic", "longitudinal", "cross-sectional",
        "epidemiological", "cohort", "case-control", "descriptive",
        "correlational", "census", "ethnography", "field observation",
        "remote sensing", "monitoring", "telemetry", "citizen science",
        "ecological study", "behavioral observation",
    ],
    ("scientific", "computational"): [
        "computational", "simulation", "model", "numerical",
        "algorithm", "machine learning", "deep learning",
        "neural network", "finite element", "molecular dynamics",
        "monte carlo simulation", "agent-based model", "in silico",
        "high-performance computing", "gpu", "parallel computing",
        "bioinformatics", "data mining", "large-scale simulation",
        "computational fluid dynamics", "cfd",
    ],
    ("scientific", "theoretical"): [
        "theoretical", "theory", "mathematical model", "analytical",
        "proof", "theorem", "conjecture", "hypothesis",
        "framework", "formalism", "axiomatic", "derivation",
        "first principles", "dimensional analysis", "perturbation theory",
        "group theory", "topology", "quantum field theory",
        "general relativity", "string theory", "renormalization",
        "phase transition", "critical phenomenon",
    ],
    ("scientific", "physics"): [
        "physics", "quantum", "particle", "photon", "electron",
        "magnetism", "electromagnetism", "thermodynamics", "entropy",
        "mechanics", "relativity", "gravity", "nuclear",
        "optics", "condensed matter", "plasma", "cosmology",
        "astrophysics", "dark matter", "dark energy", "higgs",
        "superconductor", "superfluidity", "laser",
    ],
    ("scientific", "biology"): [
        "biology", "cell", "gene", "protein", "dna", "rna",
        "evolution", "ecology", "organism", "species", "genome",
        "mutation", "natural selection", "biodiversity", "ecosystem",
        "microbiology", "immunology", "virology", "bacteriology",
        "molecular biology", "genetics", "epigenetics",
        "transcription", "translation",
    ],
    ("scientific", "chemistry"): [
        "chemistry", "chemical", "molecule", "reaction", "catalyst",
        "bond", "compound", "element", "synthesis", "polymer",
        "organic", "inorganic", "biochemistry", "electrochemistry",
        "spectroscopy", "chromatography", "titration", "oxidation",
        "reduction", "equilibrium", "kinetics", "stoichiometry",
        "crystallography", "nanochemistry",
    ],
    ("scientific", "earth_science"): [
        "geology", "climate", "atmosphere", "ocean", "seismology",
        "volcano", "earthquake", "plate tectonics", "meteorology",
        "hydrology", "geophysics", "paleontology", "fossil",
        "sediment", "erosion", "glaciology", "oceanography",
        "geochemistry", "mineralogy", "petrology", "stratigraphy",
        "remote sensing", "gis",
    ],
    ("scientific", "computer_science"): [
        "computer science", "algorithm", "data structure", "complexity",
        "software", "programming", "database", "network",
        "distributed system", "operating system", "compiler",
        "cryptography", "information theory", "artificial intelligence",
        "natural language processing", "computer vision",
        "human-computer interaction", "robotics", "cybersecurity",
        "cloud computing", "edge computing",
    ],
    ("scientific", "primary_literature"): [
        "original research", "primary data", "raw data", "first report",
        "novel finding", "new discovery", "empirical evidence",
        "experimental evidence", "direct observation", "measurement",
        "laboratory result", "field data", "collected data",
        "original contribution", "primary source",
        "first-hand evidence", "bench data", "pilot study",
        "preliminary result", "initial finding",
    ],
    ("scientific", "review"): [
        "review", "literature review", "survey", "overview",
        "state of the art", "comprehensive review", "critical review",
        "scoping review", "narrative review", "synthesis",
        "summary", "compilation", "aggregation", "meta-review",
        "umbrella review", "bibliometric", "citation analysis",
        "knowledge gap", "future direction", "research agenda",
    ],
    ("scientific", "preprint"): [
        "preprint", "arxiv", "biorxiv", "medrxiv", "ssrn",
        "working paper", "manuscript", "draft", "unpublished",
        "submitted", "under review", "not peer-reviewed",
        "preliminary", "early access", "ahead of print",
        "pre-publication", "non-refereed", "self-published",
        "repository", "open access preprint",
    ],
    # -- engineering -----------------------------------------------------
    ("engineering", "modular"): [
        "modular", "module", "plug-in", "interchangeable",
        "component-based", "decoupled", "loosely coupled",
        "interface", "api", "encapsulation", "separation of concerns",
        "microservice", "composable", "reusable", "extensible",
        "hot-swappable", "modular architecture", "building block",
        "plug-and-play", "standardized interface", "abstraction layer",
    ],
    ("engineering", "monolithic"): [
        "monolithic", "single unit", "integrated", "tightly coupled",
        "all-in-one", "unified", "single codebase", "centralized",
        "self-contained", "standalone", "one-piece", "solid block",
        "integral", "non-decomposable", "indivisible",
        "monolithic architecture", "single deployment",
        "vertical integration", "single process", "unified platform",
    ],
    ("engineering", "distributed"): [
        "distributed", "decentralized", "peer-to-peer", "cluster",
        "node", "replication", "sharding", "partitioning",
        "load balancing", "fault tolerance", "redundancy",
        "consensus", "eventual consistency", "cap theorem",
        "distributed system", "mesh", "federation",
        "horizontal scaling", "microservices", "service mesh",
        "distributed computing", "edge computing",
    ],
    ("engineering", "hybrid"): [
        "hybrid", "mixed", "combination", "blend",
        "best of both", "multi-paradigm", "composite",
        "layered", "tiered", "heterogeneous", "multi-modal",
        "hybrid architecture", "hybrid approach", "integrated approach",
        "combined strategy", "mixed-mode", "dual",
        "hybrid cloud", "multi-cloud", "hybrid system",
        "converged", "hyper-converged",
    ],
    ("engineering", "metal"): [
        "steel", "aluminum", "aluminium", "titanium", "copper",
        "iron", "alloy", "stainless steel", "carbon steel",
        "brass", "bronze", "zinc", "nickel", "chromium",
        "tungsten", "magnesium", "metallic", "ferrous",
        "non-ferrous", "sheet metal", "casting", "forging",
        "welding", "machining",
    ],
    ("engineering", "polymer"): [
        "polymer", "plastic", "composite", "fiberglass",
        "carbon fiber", "kevlar", "epoxy", "resin",
        "thermoplastic", "thermoset", "polyethylene", "polypropylene",
        "nylon", "pvc", "abs", "polycarbonate",
        "polyester", "polyurethane", "silicone", "rubber",
        "elastomer", "laminate", "injection molding",
    ],
    ("engineering", "ceramic"): [
        "ceramic", "glass", "porcelain", "concrete", "cement",
        "brick", "tile", "refractory", "alumina", "zirconia",
        "silicon carbide", "boron nitride", "piezoelectric",
        "oxide ceramic", "non-oxide ceramic", "sintering",
        "vitrification", "glazing", "kiln", "fired",
        "calcium silicate", "calcium carbonate",
    ],
    ("engineering", "natural"): [
        "wood", "timber", "bamboo", "stone", "granite",
        "marble", "sandstone", "limestone", "leather",
        "cotton", "wool", "silk", "hemp", "natural fiber",
        "bioplastic", "biomaterial", "biodegradable",
        "sustainable material", "renewable material",
        "cork", "natural rubber", "cellulose",
    ],
    ("engineering", "cost"): [
        "cost", "budget", "expense", "price", "affordable",
        "cost-effective", "economical", "capital expenditure",
        "operating cost", "total cost of ownership", "roi",
        "return on investment", "payback period", "cost reduction",
        "cost optimization", "value engineering", "cost analysis",
        "bill of materials", "unit cost", "marginal cost",
        "fixed cost", "variable cost", "cost constraint",
    ],
    ("engineering", "weight"): [
        "weight", "mass", "lightweight", "heavy", "density",
        "load", "payload", "dead weight", "gross weight",
        "net weight", "weight reduction", "weight optimization",
        "strength-to-weight", "power-to-weight", "weight constraint",
        "structural weight", "curb weight", "takeoff weight",
        "gravitational load", "specific gravity", "buoyancy",
    ],
    ("engineering", "safety"): [
        "safety", "hazard", "risk assessment", "failure mode",
        "fmea", "safety factor", "factor of safety", "redundancy",
        "failsafe", "fail-safe", "protective", "guard",
        "warning", "emergency", "evacuation", "fire safety",
        "electrical safety", "mechanical safety", "chemical safety",
        "osha", "safety standard", "safety regulation",
        "safety analysis", "incident", "accident prevention",
    ],
    ("engineering", "efficiency"): [
        "efficiency", "performance", "throughput", "utilization",
        "optimization", "energy efficiency", "thermal efficiency",
        "fuel efficiency", "power consumption", "latency",
        "bandwidth", "cycle time", "uptime", "availability",
        "mean time between failures", "mtbf", "yield",
        "productivity", "waste reduction", "lean",
        "six sigma", "process improvement", "streamline",
    ],
    ("engineering", "trade_off_performance_cost"): [
        "trade-off", "tradeoff", "performance vs cost",
        "cost-performance", "value proposition", "diminishing returns",
        "optimal balance", "compromise", "acceptable trade-off",
        "marginal benefit", "marginal cost", "pareto",
        "cost-benefit analysis", "break-even", "sweet spot",
        "goldilocks", "balanced approach", "multi-objective",
        "constraint satisfaction", "design space exploration",
    ],
    ("engineering", "trade_off_speed_accuracy"): [
        "speed vs accuracy", "latency vs precision",
        "real-time constraint", "approximate", "heuristic",
        "good enough", "precision vs recall", "accuracy trade-off",
        "time constraint", "deadline", "response time",
        "processing speed", "computational cost", "numerical precision",
        "floating point", "rounding error", "approximation algorithm",
        "lossy", "lossless", "sampling rate",
    ],
    ("engineering", "trade_off_scalability_simplicity"): [
        "scalability vs simplicity", "complexity", "scalable",
        "simple design", "over-engineering", "premature optimization",
        "kiss principle", "yagni", "technical debt",
        "maintainability", "extensibility", "complexity budget",
        "architectural complexity", "cognitive load", "learning curve",
        "scale", "horizontal scaling", "vertical scaling",
        "capacity planning", "growth",
    ],
}


class DomainTerminologyMatcher:
    """Match domain-specific terminology in text.

    Uses curated keyword/phrase lists organised by ``(domain, category)``.
    Matching is case-insensitive and uses word-boundary-aware regex so that
    partial matches inside longer words are avoided for multi-word phrases.
    """

    def __init__(
        self,
        extra_terms: Optional[Dict[Tuple[str, str], List[str]]] = None,
    ) -> None:
        self._terms: Dict[Tuple[str, str], List[str]] = dict(_TERMINOLOGY)
        if extra_terms:
            for key, phrases in extra_terms.items():
                self._terms.setdefault(key, []).extend(phrases)

        # Pre-compile patterns keyed by (domain, category)
        self._patterns: Dict[Tuple[str, str], List[re.Pattern[str]]] = {}
        for key, phrases in self._terms.items():
            pats: List[re.Pattern[str]] = []
            for phrase in phrases:
                escaped = re.escape(phrase.lower())
                pats.append(re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE))
            self._patterns[key] = pats

    def match(self, text: str, domain: str) -> Dict[str, List[str]]:
        """Return ``{category: [matched_phrases]}`` for *domain*."""
        lower = _lower_text(text)
        result: Dict[str, List[str]] = {}
        for (d, cat), pats in self._patterns.items():
            if d != domain:
                continue
            matched: List[str] = []
            for pat, phrase in zip(pats, self._terms[(d, cat)]):
                if pat.search(lower):
                    matched.append(phrase)
            if matched:
                result[cat] = matched
        return result

    def categorize(self, text: str, domain: str) -> Counter:
        """Return a ``Counter`` mapping category -> number of matched terms."""
        matches = self.match(text, domain)
        return Counter({cat: len(ms) for cat, ms in matches.items()})

    def coverage(self, texts: List[str], domain: str) -> Dict[str, float]:
        """Fraction of texts that mention at least one term per category."""
        if not texts:
            return {}
        cats: Dict[str, int] = defaultdict(int)
        for text in texts:
            matched_cats = set(self.match(text, domain).keys())
            for c in matched_cats:
                cats[c] += 1
        return {c: count / len(texts) for c, count in cats.items()}


# ---------------------------------------------------------------------------
# DomainDiversityAnalyzer — generic framework
# ---------------------------------------------------------------------------


class DomainDiversityAnalyzer:
    """Generic domain diversity analysis framework.

    A *domain_config* is a dict with:
    - ``"domain"``: str — the domain name used in the terminology matcher
    - ``"category_groups"``: dict mapping group-name -> list of categories
    - ``"weights"``: optional dict mapping group-name -> float weight
    """

    def __init__(self, matcher: Optional[DomainTerminologyMatcher] = None) -> None:
        self._matcher = matcher or DomainTerminologyMatcher()

    def analyze(
        self,
        texts: List[str],
        domain_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run diversity analysis on *texts* according to *domain_config*.

        Returns a dictionary with per-group entropy scores, overall coverage,
        and a weighted overall score.
        """
        domain = domain_config["domain"]
        groups: Dict[str, List[str]] = domain_config.get("category_groups", {})
        weights: Dict[str, float] = domain_config.get("weights", {})

        group_counters: Dict[str, Counter] = {g: Counter() for g in groups}
        for text in texts:
            cats = self._matcher.categorize(text, domain)
            for g, g_cats in groups.items():
                for c in g_cats:
                    if c in cats:
                        group_counters[g][c] += cats[c]

        entropies: Dict[str, float] = {}
        for g, counter in group_counters.items():
            entropies[g] = _normalised_entropy(counter)

        coverage = self._matcher.coverage(texts, domain)

        # Weighted overall score
        total_weight = sum(weights.get(g, 1.0) for g in groups)
        if total_weight == 0:
            overall = 0.0
        else:
            overall = sum(
                entropies.get(g, 0.0) * weights.get(g, 1.0) for g in groups
            ) / total_weight

        return {
            "entropies": entropies,
            "group_counts": {g: dict(c) for g, c in group_counters.items()},
            "coverage": coverage,
            "overall_score": overall,
        }

    def compare_domains(
        self,
        texts: List[str],
        domains: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse the same *texts* across multiple domain configs."""
        return {cfg["domain"]: self.analyze(texts, cfg) for cfg in domains}

    def recommend_gaps(self, report: Dict[str, Any]) -> List[str]:
        """Suggest categories that are under-represented in *report*."""
        gaps: List[str] = []
        group_counts: Dict[str, Dict[str, int]] = report.get("group_counts", {})
        for group, counts in group_counts.items():
            if not counts:
                gaps.append(f"No matches found in group '{group}'; consider adding content for this area.")
                continue
            total = sum(counts.values())
            n = len(counts)
            if n < 2:
                continue
            mean_share = 1.0 / n
            for cat, cnt in counts.items():
                share = cnt / total
                if share < mean_share * 0.3:
                    gaps.append(
                        f"Category '{cat}' in group '{group}' is under-represented "
                        f"({share:.1%} vs expected {mean_share:.1%})."
                    )
        coverage: Dict[str, float] = report.get("coverage", {})
        for cat, cov in coverage.items():
            if cov < 0.15:
                gaps.append(
                    f"Category '{cat}' appears in only {cov:.0%} of texts; "
                    "more coverage would improve diversity."
                )
        return gaps


# ---------------------------------------------------------------------------
# Domain analysis functions
# ---------------------------------------------------------------------------

_MATCHER = DomainTerminologyMatcher()

_ARGUMENT_TYPE_CATS = ["statutory", "constitutional", "precedent-based", "policy", "procedural"]
_JURISDICTION_CATS = ["federal", "state", "international"]
_REASONING_CATS = ["deductive", "analogical", "policy-based"]
_LEGAL_DOMAIN_CATS = ["criminal", "civil", "administrative"]


def legal_diversity(legal_texts: List[str]) -> LegalDiversityReport:
    """Analyse diversity of a collection of legal texts.

    Detects argument types (statutory, constitutional, precedent-based,
    policy, procedural), jurisdictions mentioned (federal, state,
    international), reasoning patterns (deductive, analogical, policy-based),
    and legal domains (criminal, civil, administrative, constitutional).
    Computes normalised entropy over the resulting distributions.
    """
    if not legal_texts:
        return LegalDiversityReport()

    arg_counter: Counter = Counter()
    juris_counter: Counter = Counter()
    reasoning_counter: Counter = Counter()
    domain_counter: Counter = Counter()

    per_text_terms: List[Set[str]] = []

    for text in legal_texts:
        cats = _MATCHER.categorize(text, "legal")
        text_terms: Set[str] = set()
        for cat in _ARGUMENT_TYPE_CATS:
            if cat in cats:
                arg_counter[cat] += cats[cat]
                text_terms.add(cat)
        for cat in _JURISDICTION_CATS:
            if cat in cats:
                juris_counter[cat] += cats[cat]
                text_terms.add(cat)
        for cat in _REASONING_CATS:
            if cat in cats:
                reasoning_counter[cat] += cats[cat]
                text_terms.add(cat)
        for cat in _LEGAL_DOMAIN_CATS + ["constitutional"]:
            if cat in cats:
                domain_counter[cat] += cats[cat]
                text_terms.add(cat)
        per_text_terms.append(text_terms)

    arg_entropy = _normalised_entropy(arg_counter)
    juris_entropy = _normalised_entropy(juris_counter)
    reasoning_entropy = _normalised_entropy(reasoning_counter)
    domain_entropy = _normalised_entropy(domain_counter)

    precedent_diversity = _pairwise_dissimilarity(per_text_terms)

    overall = float(np.mean([arg_entropy, juris_entropy, reasoning_entropy, domain_entropy, precedent_diversity]))

    return LegalDiversityReport(
        argument_types=dict(arg_counter),
        jurisdiction_coverage=dict(juris_counter),
        precedent_diversity=precedent_diversity,
        reasoning_diversity=reasoning_entropy,
        overall_score=overall,
        details={
            "argument_entropy": arg_entropy,
            "jurisdiction_entropy": juris_entropy,
            "reasoning_entropy": reasoning_entropy,
            "domain_entropy": domain_entropy,
            "domain_counts": dict(domain_counter),
            "n_texts": len(legal_texts),
        },
    )


_DIAGNOSIS_CATS = [
    "cardiovascular", "neurological", "respiratory",
    "musculoskeletal", "gastrointestinal", "endocrine",
]
_TREATMENT_CATS = ["pharmaceutical", "surgical", "therapeutic", "preventive"]
_EVIDENCE_CATS = ["meta-analysis", "rct", "cohort", "case_study", "expert_opinion"]
_SPECIALTY_CATS = _DIAGNOSIS_CATS  # re-use body-system categories as specialties


def medical_diversity(medical_texts: List[str]) -> MedicalDiversityReport:
    """Analyse diversity of medical/clinical texts.

    Detects diagnosis categories (by body system), treatment modalities
    (pharmaceutical, surgical, therapeutic, preventive), evidence levels
    (meta-analysis, RCT, cohort, case study, expert opinion), and medical
    specialties.  Scores breadth and balance via normalised entropy.
    """
    if not medical_texts:
        return MedicalDiversityReport()

    diag_counter: Counter = Counter()
    treat_counter: Counter = Counter()
    evidence_counter: Counter = Counter()
    specialty_counter: Counter = Counter()
    risk_counter: Counter = Counter()

    for text in medical_texts:
        cats = _MATCHER.categorize(text, "medical")
        for cat in _DIAGNOSIS_CATS:
            if cat in cats:
                diag_counter[cat] += cats[cat]
        for cat in _TREATMENT_CATS:
            if cat in cats:
                treat_counter[cat] += cats[cat]
        for cat in _EVIDENCE_CATS:
            if cat in cats:
                evidence_counter[cat] += cats[cat]
        for cat in _SPECIALTY_CATS:
            if cat in cats:
                specialty_counter[cat] += cats[cat]
        # Risk coverage: flag texts mentioning adverse effects, contraindications
        risk_terms = ["adverse", "risk", "complication", "contraindication",
                      "side effect", "mortality", "morbidity", "toxicity",
                      "overdose", "allergic reaction", "interaction"]
        lower = _lower_text(text)
        for rt in risk_terms:
            if rt in lower:
                risk_counter[rt] += 1

    diag_entropy = _normalised_entropy(diag_counter)
    treat_entropy = _normalised_entropy(treat_counter)
    evidence_entropy = _normalised_entropy(evidence_counter)
    specialty_entropy = _normalised_entropy(specialty_counter)
    risk_entropy = _normalised_entropy(risk_counter)

    overall = float(np.mean([
        diag_entropy, treat_entropy, evidence_entropy,
        specialty_entropy, risk_entropy,
    ]))

    return MedicalDiversityReport(
        diagnosis_breadth=dict(diag_counter),
        treatment_diversity=dict(treat_counter),
        evidence_levels=dict(evidence_counter),
        specialty_coverage=dict(specialty_counter),
        risk_coverage=dict(risk_counter),
        overall_score=overall,
        details={
            "diagnosis_entropy": diag_entropy,
            "treatment_entropy": treat_entropy,
            "evidence_entropy": evidence_entropy,
            "specialty_entropy": specialty_entropy,
            "risk_entropy": risk_entropy,
            "n_texts": len(medical_texts),
        },
    )


_MARKET_VIEW_CATS = ["bullish", "bearish", "neutral"]
_SECTOR_CATS = [
    "technology", "healthcare", "energy", "financial_sector",
    "consumer", "industrials", "real_estate",
]
_TIME_HORIZON_CATS = ["short_term", "medium_term", "long_term"]
_ANALYSIS_TYPE_CATS = ["fundamental", "technical", "quantitative", "macro"]


def financial_diversity(analyses: List[str]) -> FinancialDiversityReport:
    """Analyse diversity of financial analyses.

    Detects market outlook (bullish, bearish, neutral), sectors mentioned,
    time horizons (short/medium/long-term), analysis types (fundamental,
    technical, quantitative, macro), and risk perspectives.  Computes
    normalised entropy over each distribution.
    """
    if not analyses:
        return FinancialDiversityReport()

    view_counter: Counter = Counter()
    sector_counter: Counter = Counter()
    horizon_counter: Counter = Counter()
    analysis_counter: Counter = Counter()
    risk_counter: Counter = Counter()

    risk_keywords = [
        "downside risk", "upside risk", "tail risk", "systemic risk",
        "credit risk", "market risk", "liquidity risk", "operational risk",
        "inflation risk", "currency risk", "interest rate risk",
        "geopolitical risk", "concentration risk", "counterparty risk",
        "volatility", "drawdown", "stress test", "scenario analysis",
        "risk management", "hedging", "diversification",
    ]

    for text in analyses:
        cats = _MATCHER.categorize(text, "financial")
        for cat in _MARKET_VIEW_CATS:
            if cat in cats:
                view_counter[cat] += cats[cat]
        for cat in _SECTOR_CATS:
            if cat in cats:
                sector_counter[cat] += cats[cat]
        for cat in _TIME_HORIZON_CATS:
            if cat in cats:
                horizon_counter[cat] += cats[cat]
        for cat in _ANALYSIS_TYPE_CATS:
            if cat in cats:
                analysis_counter[cat] += cats[cat]
        lower = _lower_text(text)
        for rk in risk_keywords:
            if rk in lower:
                risk_counter[rk] += 1

    view_entropy = _normalised_entropy(view_counter)
    sector_entropy = _normalised_entropy(sector_counter)
    horizon_entropy = _normalised_entropy(horizon_counter)
    analysis_entropy = _normalised_entropy(analysis_counter)
    risk_entropy = _normalised_entropy(risk_counter)

    overall = float(np.mean([
        view_entropy, sector_entropy, horizon_entropy,
        analysis_entropy, risk_entropy,
    ]))

    return FinancialDiversityReport(
        market_views=dict(view_counter),
        sector_coverage=dict(sector_counter),
        time_horizon_diversity=dict(horizon_counter),
        risk_perspective_diversity=dict(risk_counter),
        overall_score=overall,
        details={
            "view_entropy": view_entropy,
            "sector_entropy": sector_entropy,
            "horizon_entropy": horizon_entropy,
            "analysis_entropy": analysis_entropy,
            "risk_entropy": risk_entropy,
            "analysis_type_counts": dict(analysis_counter),
            "n_texts": len(analyses),
        },
    )


_METHODOLOGY_CATS = ["experimental", "observational", "computational", "theoretical"]
_FIELD_CATS = ["physics", "biology", "chemistry", "earth_science", "computer_science"]
_SCI_EVIDENCE_CATS = ["primary_literature", "review", "preprint"]


def scientific_diversity(hypotheses: List[str]) -> ScientificDiversityReport:
    """Analyse diversity of scientific hypotheses or papers.

    Detects research methodology (experimental, observational, computational,
    theoretical), scientific fields, evidence types, and novelty indicators.
    Novelty is scored as average pairwise Jaccard distance of per-text
    keyword sets combined with frequency of novelty-indicating phrases.
    """
    if not hypotheses:
        return ScientificDiversityReport()

    method_counter: Counter = Counter()
    field_counter: Counter = Counter()
    evidence_counter: Counter = Counter()

    novelty_phrases = [
        "novel", "new approach", "first time", "for the first time",
        "unprecedented", "pioneering", "breakthrough", "innovative",
        "never before", "original", "newly discovered", "cutting-edge",
        "state-of-the-art", "paradigm shift", "transformative",
        "ground-breaking", "groundbreaking", "first report",
        "first demonstration", "unique", "unexplored",
    ]

    per_text_keyword_sets: List[Set[str]] = []
    novelty_hits = 0
    total_novelty_checks = 0

    for text in hypotheses:
        cats = _MATCHER.categorize(text, "scientific")
        kw_set: Set[str] = set()
        for cat in _METHODOLOGY_CATS:
            if cat in cats:
                method_counter[cat] += cats[cat]
                kw_set.add(cat)
        for cat in _FIELD_CATS:
            if cat in cats:
                field_counter[cat] += cats[cat]
                kw_set.add(cat)
        for cat in _SCI_EVIDENCE_CATS:
            if cat in cats:
                evidence_counter[cat] += cats[cat]
                kw_set.add(cat)
        per_text_keyword_sets.append(kw_set)

        lower = _lower_text(text)
        for phrase in novelty_phrases:
            total_novelty_checks += 1
            if phrase in lower:
                novelty_hits += 1

    hypothesis_div = _pairwise_dissimilarity(per_text_keyword_sets)
    method_entropy = _normalised_entropy(method_counter)
    field_entropy = _normalised_entropy(field_counter)
    evidence_entropy = _normalised_entropy(evidence_counter)

    # Novelty score: blend of dissimilarity and novelty phrase frequency
    if total_novelty_checks > 0:
        novelty_phrase_rate = novelty_hits / total_novelty_checks
    else:
        novelty_phrase_rate = 0.0
    novelty_score = 0.6 * hypothesis_div + 0.4 * min(novelty_phrase_rate * 5.0, 1.0)

    overall = float(np.mean([
        method_entropy, field_entropy, evidence_entropy,
        hypothesis_div, novelty_score,
    ]))

    return ScientificDiversityReport(
        hypothesis_diversity=hypothesis_div,
        methodology_diversity=dict(method_counter),
        field_coverage=dict(field_counter),
        evidence_types=dict(evidence_counter),
        novelty_score=novelty_score,
        overall_score=overall,
        details={
            "method_entropy": method_entropy,
            "field_entropy": field_entropy,
            "evidence_entropy": evidence_entropy,
            "novelty_phrase_rate": novelty_phrase_rate,
            "n_texts": len(hypotheses),
        },
    )


_DESIGN_APPROACH_CATS = ["modular", "monolithic", "distributed", "hybrid"]
_MATERIAL_CATS = ["metal", "polymer", "ceramic", "natural"]
_CONSTRAINT_CATS = ["cost", "weight", "safety", "efficiency"]
_TRADEOFF_CATS = [
    "trade_off_performance_cost",
    "trade_off_speed_accuracy",
    "trade_off_scalability_simplicity",
]


def engineering_diversity(designs: List[str]) -> EngineeringDiversityReport:
    """Analyse diversity of engineering design proposals.

    Detects design approach (modular, monolithic, distributed, hybrid),
    materials mentioned (metal, polymer, ceramic, natural), constraints
    addressed (cost, weight, safety, efficiency), and trade-offs discussed.
    Scores coverage of the design space via normalised entropy and a
    taxonomy-based coverage metric.
    """
    if not designs:
        return EngineeringDiversityReport()

    approach_counter: Counter = Counter()
    material_counter: Counter = Counter()
    constraint_counter: Counter = Counter()
    tradeoff_counter: Counter = Counter()

    # Build a simple taxonomy for engineering design space
    taxonomy = TaxonomyTree()
    taxonomy.add_node(None, "design")
    for a in _DESIGN_APPROACH_CATS:
        taxonomy.add_node("design", a)
    taxonomy.add_node(None, "materials")
    for m in _MATERIAL_CATS:
        taxonomy.add_node("materials", m)
    taxonomy.add_node(None, "constraints")
    for c in _CONSTRAINT_CATS:
        taxonomy.add_node("constraints", c)
    taxonomy.add_node(None, "tradeoffs")
    for t in _TRADEOFF_CATS:
        taxonomy.add_node("tradeoffs", t)

    matched_taxonomy_nodes: Set[str] = set()

    for text in designs:
        cats = _MATCHER.categorize(text, "engineering")
        for cat in _DESIGN_APPROACH_CATS:
            if cat in cats:
                approach_counter[cat] += cats[cat]
                matched_taxonomy_nodes.add(cat)
        for cat in _MATERIAL_CATS:
            if cat in cats:
                material_counter[cat] += cats[cat]
                matched_taxonomy_nodes.add(cat)
        for cat in _CONSTRAINT_CATS:
            if cat in cats:
                constraint_counter[cat] += cats[cat]
                matched_taxonomy_nodes.add(cat)
        for cat in _TRADEOFF_CATS:
            if cat in cats:
                tradeoff_counter[cat] += cats[cat]
                matched_taxonomy_nodes.add(cat)

    approach_entropy = _normalised_entropy(approach_counter)
    material_entropy = _normalised_entropy(material_counter)
    constraint_entropy = _normalised_entropy(constraint_counter)
    tradeoff_entropy = _normalised_entropy(tradeoff_counter)

    tax_coverage = taxonomy.depth_weighted_coverage(matched_taxonomy_nodes)

    overall = float(np.mean([
        approach_entropy, material_entropy,
        constraint_entropy, tradeoff_entropy,
        tax_coverage,
    ]))

    return EngineeringDiversityReport(
        design_approach_diversity=dict(approach_counter),
        material_diversity=dict(material_counter),
        constraint_coverage=dict(constraint_counter),
        trade_off_coverage=dict(tradeoff_counter),
        overall_score=overall,
        details={
            "approach_entropy": approach_entropy,
            "material_entropy": material_entropy,
            "constraint_entropy": constraint_entropy,
            "tradeoff_entropy": tradeoff_entropy,
            "taxonomy_coverage": tax_coverage,
            "n_texts": len(designs),
        },
    )
