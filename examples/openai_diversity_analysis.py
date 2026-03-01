#!/usr/bin/env python3
"""
OpenAI Diversity Analysis — generate text at different temperatures
and discover which diversity metrics are redundant vs. complementary.

This example shows the unique value of the Diversity Decoding Arena:
given multiple sets of LLM outputs, it computes all diversity metrics
and builds a correlation taxonomy so you know exactly which metrics
to report in your paper or evaluation.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Run from the implementation/ directory:
    python examples/openai_diversity_analysis.py

If no API key is set, the script uses built-in sample data so you can
still see the analysis pipeline.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.metrics.diversity import (
    DistinctN,
    DiversityMetricSuite,
    EmbeddingPairwiseDistance,
    NGramEntropy,
    SelfBLEU,
    VendiScore,
)
from src.metrics.correlation import (
    MetricCorrelationAnalyzer,
    MetricRedundancyAnalyzer,
    effective_dimensionality,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

PROMPT = "Write a one-sentence description of an imaginary planet."
TEMPERATURES = [0.2, 0.5, 0.7, 1.0, 1.2, 1.5]
N_SAMPLES = 8  # generations per temperature
MODEL = "gpt-3.5-turbo"

# -----------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------


def generate_with_openai(
    prompt: str,
    temperatures: list[float],
    n_samples: int,
    model: str,
) -> dict[str, list[str]]:
    """Generate texts at each temperature using the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed — using sample data.", file=sys.stderr)
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — using sample data.", file=sys.stderr)
        return {}

    client = OpenAI(api_key=api_key)
    groups: dict[str, list[str]] = {}

    for temp in temperatures:
        label = f"temp_{temp}"
        texts: list[str] = []
        print(f"  Generating {n_samples} samples at temperature={temp} ...", end=" ")
        for _ in range(n_samples):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=80,
                n=1,
            )
            texts.append(response.choices[0].message.content.strip())
        groups[label] = texts
        print("done")

    return groups


def sample_data() -> dict[str, list[str]]:
    """Realistic stand-in data simulating temperature-based diversity."""
    return {
        "temp_0.2": [
            "Zephyria is a small, rocky planet with a thin nitrogen atmosphere and vast deserts of crimson sand.",
            "Zephyria is a small, rocky world with a thin atmosphere and expansive deserts of red sand.",
            "Zephyria is a compact rocky planet featuring a thin nitrogen atmosphere and wide crimson deserts.",
            "Zephyria is a small rocky planet with a thin nitrogen atmosphere and endless red sand deserts.",
            "Zephyria is a small, rocky planet with a thin atmosphere and deserts of crimson-colored sand.",
            "Zephyria is a small rocky world with a thin nitrogen atmosphere and vast red sand deserts.",
            "Zephyria is a compact, rocky planet with a thin atmosphere and expansive crimson sand deserts.",
            "Zephyria is a small, rocky planet with a thin nitrogen atmosphere and broad deserts of red sand.",
        ],
        "temp_0.5": [
            "Verdania is a lush, tropical planet covered in dense bioluminescent forests and warm shallow oceans.",
            "Crystara orbits twin suns and features vast plains of naturally forming quartz crystal formations.",
            "Zephyria is a small rocky planet with crimson deserts and a thin but breathable atmosphere.",
            "Verdania is a vibrant tropical world with glowing forests and oceans teeming with aquatic life.",
            "Nebulon is a gas giant surrounded by rings of frozen methane and dozens of icy moons.",
            "Crystara is a shimmering world where massive crystal spires grow from the planet's silicon-rich crust.",
            "Verdania is a lush planet with bioluminescent vegetation and warm, shallow seas full of coral.",
            "Zephyria features vast crimson deserts under a pale orange sky with two small moons.",
        ],
        "temp_0.7": [
            "Aurelion is a tidally locked world where one hemisphere blazes while the other lies in perpetual frost.",
            "Mycosia is a fungal planet where towering mushroom forests exhale spores that light up the violet sky.",
            "Glimmertide orbits a pulsar and experiences electromagnetic storms that paint its skies in neon ribbons.",
            "Petralune is a hollow planet whose interior contains a luminous ocean heated by geothermal vents.",
            "Verdania's triple canopy of bioluminescent trees creates a permanent twilight on the forest floor below.",
            "Cryovex is an ice world with subsurface ammonia rivers that carve labyrinthine caves through glaciers.",
            "Solanthea drifts through a nebula, collecting stardust that settles as iridescent snow on its mountain peaks.",
            "Terranova is an ocean planet where floating islands of pumice support entire civilizations above the waves.",
        ],
        "temp_1.0": [
            "Quillthorn is a planet of sentient cacti whose electromagnetic hums compose an ever-changing planetary symphony.",
            "Duskweave has an atmosphere of suspended silk-like fibers that filter starlight into perpetual golden twilight.",
            "On Echomere, sound travels as visible color waves, creating a world where every whisper paints the air.",
            "Ferroglide is a molten-core planet where magnetic lifeforms surf rivers of liquid iron beneath aurora skies.",
            "Voidpetal exists at the edge of a black hole, its flowers blooming with captured gravitational energy.",
            "Mirrordeep is an ocean world whose waters are perfectly reflective, making the sea and sky indistinguishable.",
            "Chronalis experiences time at different rates across its surface, creating temporal archipelagos of past and future.",
            "Tessellara is covered in hexagonal basalt columns that resonate with infrasound during its frequent moonquakes.",
        ],
        "temp_1.2": [
            "Whimsicora breathes confetti-pollen storms through sentient cloud-mouths that giggle in subsonic frequencies above magenta tundra.",
            "Paradoxium exists simultaneously as a solid and gas, its inhabitants phasing between states with each heartbeat.",
            "On Synesthex, gravity tastes like cinnamon and the aurora smells of rain-soaked copper and old books.",
            "Fractalheim's coastlines recursively nest smaller coastlines infinitely, making cartography a branch of pure mathematics.",
            "Membrainia is a living cell the size of Jupiter, its organelles hosting distinct biomes of microscopic civilizations.",
            "Palindrova's history runs backward and forward simultaneously, its inhabitants remembering both their birth and unbirth.",
            "Glitchmere suffers rendering errors in its reality, causing trees to clip through mountains and skies to buffer.",
            "Umbraweave knits shadows into solid architecture, its cities standing only while the triple suns cast overlapping darkness.",
        ],
        "temp_1.5": [
            "Squonkleberry fizzes with electromagnetic jam-tides that hiccup mathematical theorems into being across crystallized thought-meadows of yonder.",
            "Blorbatron's seventeen-dimensional surface folds inward through recursive dreams, each layer tasting increasingly like forgotten Tuesdays in amber.",
            "On Wumblethax, the ground argues with itself about whether to be solid while rain falls upward diagonally into song.",
            "Noodlevex spirals through spacetime like a cosmic corkscrew, its inhabitants experiencing dinner before breakfast and gravity sideways.",
            "Snorkelplume's atmosphere is thick with liquid questions that condense into solid answers during the bi-annual philosophy monsoon.",
            "Quarklejazz resonates at the frequency of abstract gratitude, causing visiting astronauts to spontaneously compose operas in colors.",
            "Flimflammora is governed by a council of self-aware weather patterns who debate policy through interpretive thunderstorms.",
            "Zurblewhisk occupies the space between two colliding parallel universes, serving as reality's awkward middle seat at dinner.",
        ],
    }


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("OpenAI Diversity Analysis — Temperature vs. Diversity Metrics")
    print("=" * 65)

    # Generate or load data
    print("\nStep 1: Generating texts at different temperatures...")
    groups = generate_with_openai(PROMPT, TEMPERATURES, N_SAMPLES, MODEL)
    if not groups:
        print("  Using built-in sample data.\n")
        groups = sample_data()

    # Build metric suite
    suite = DiversityMetricSuite([
        SelfBLEU(max_order=4),
        DistinctN(n=1),
        DistinctN(n=2),
        NGramEntropy(n=2),
        EmbeddingPairwiseDistance(
            distance_metric="cosine", embedding_method="tfidf"
        ),
        VendiScore(kernel_type="cosine"),
    ])

    # Compute metrics per group
    print("Step 2: Computing diversity metrics per temperature...")
    all_results: dict[str, dict[str, float]] = {}
    for gname, texts in sorted(groups.items()):
        all_results[gname] = suite.compute_all(texts)
        print(f"  {gname}: {len(texts)} texts evaluated")

    # Display scores
    metric_names = suite.metric_names
    print("\n" + "-" * 65)
    col_w = 10
    header = f"{'Temperature':<15}" + "".join(f"{m[:col_w]:>{col_w}}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for gname in sorted(all_results.keys()):
        row = f"{gname:<15}" + "".join(
            f"{all_results[gname][m]:>{col_w}.4f}" for m in metric_names
        )
        print(row)

    # Correlation analysis
    print("\n" + "=" * 65)
    print("Step 3: Metric Correlation Taxonomy")
    print("=" * 65)
    print("Which metrics capture the same information?\n")

    group_list = sorted(all_results.keys())
    metric_values = {
        m: [all_results[g][m] for g in group_list] for m in metric_names
    }

    # Filter constant metrics
    active_metrics = [
        m for m in metric_names if np.std(metric_values[m]) > 1e-12
    ]

    if len(active_metrics) < 2 or len(group_list) < 3:
        print("Not enough variation for taxonomy analysis.")
        return

    analyzer = MetricCorrelationAnalyzer(metrics=active_metrics)
    active_values = {m: metric_values[m] for m in active_metrics}
    corr = analyzer.compute_correlation_matrix(active_values)

    # Print correlation matrix
    mc = max(len(n) for n in active_metrics) + 1
    hdr = " " * mc + "".join(f"{m[:9]:>10}" for m in active_metrics)
    print(hdr)
    for i, name in enumerate(active_metrics):
        vals = "".join(f"{corr[i, j]:>10.3f}" for j in range(len(active_metrics)))
        print(f"{name:<{mc}}{vals}")

    # Redundancy
    redundancy = MetricRedundancyAnalyzer(threshold=0.8)
    redundant = redundancy.find_redundant_pairs(corr, active_metrics)
    orthogonal = redundancy.find_orthogonal_pairs(corr, active_metrics)
    recommended = redundancy.select_representative_metrics(corr, active_metrics)

    print(f"\n{'─' * 65}")
    print("Redundant pairs (|τ| ≥ 0.8):")
    if redundant:
        for a, b, tau in redundant:
            print(f"  ⚠  {a}  ↔  {b}   τ = {tau:+.3f}")
        print("  → These pairs measure essentially the same thing.")
        print("    You only need one metric from each redundant pair.")
    else:
        print("  ✓ No redundant pairs — all metrics are independent!")

    print(f"\nComplementary pairs (|τ| ≤ 0.1):")
    if orthogonal:
        for a, b, tau in orthogonal:
            print(f"  ✓  {a}  ↔  {b}   τ = {tau:+.3f}")
        print("  → These pairs capture genuinely different aspects of diversity.")
    else:
        print("  None at threshold 0.1")

    eigenvalues = np.linalg.eigvalsh(corr)
    eff_dim = effective_dimensionality(np.maximum(eigenvalues, 0))
    independence = redundancy.metric_independence_score(corr)

    print(f"\n{'─' * 65}")
    print("Summary")
    print(f"  Metrics evaluated:        {len(active_metrics)}")
    print(f"  Effective dimensionality:  {eff_dim:.1f} / {len(active_metrics)}")
    print(f"  Independence score:        {independence:.3f}  (1.0 = all independent)")
    print(f"  Recommended subset:        {', '.join(recommended)}")
    print(
        "\n  → Report these metrics for a non-redundant diversity evaluation."
    )

    # Save results
    output = {
        "prompt": PROMPT,
        "model": MODEL,
        "temperatures": TEMPERATURES,
        "metric_scores": {
            g: {m: round(v, 6) for m, v in vals.items()}
            for g, vals in all_results.items()
        },
        "taxonomy": {
            "correlation_matrix": corr.tolist(),
            "metric_names": active_metrics,
            "redundant_pairs": [
                {"a": a, "b": b, "tau": round(t, 4)} for a, b, t in redundant
            ],
            "complementary_pairs": [
                {"a": a, "b": b, "tau": round(t, 4)} for a, b, t in orthogonal
            ],
            "recommended_subset": recommended,
            "effective_dimensionality": round(eff_dim, 2),
            "independence_score": round(independence, 4),
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "openai_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
