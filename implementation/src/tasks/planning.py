"""
Planning and reasoning task domain for the Diversity Decoding Arena.

Evaluates diversity-promoting decoding algorithms on planning and multi-step
reasoning tasks: logistics, project management, cooking recipes, travel
itineraries, problem solving, strategy/game-theory, constraint satisfaction,
and causal reasoning.  Measures plan completeness, step ordering, feasibility,
efficiency, constraint satisfaction, causal coherence, detail level, robustness,
creativity, and diversity of generated plans.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import textwrap
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from src.tasks.base import (
    ConstraintType,
    GenerationTask,
    PromptDataset,
    TaskConfig,
    TaskConstraint,
    TaskEvaluator,
    TaskPrompt,
    _TASK_REGISTRY,
)
from src.types import TaskDomain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlanningDomain(Enum):
    """Sub-domain of planning task."""

    LOGISTICS = auto()
    PROJECT_MANAGEMENT = auto()
    COOKING = auto()
    TRAVEL = auto()
    PROBLEM_SOLVING = auto()
    STRATEGY = auto()
    CONSTRAINT_SATISFACTION = auto()
    CAUSAL_REASONING = auto()

    def __repr__(self) -> str:
        return f"PlanningDomain.{self.name}"

    @property
    def description(self) -> str:
        _desc = {
            PlanningDomain.LOGISTICS: "Route planning, scheduling, and resource allocation",
            PlanningDomain.PROJECT_MANAGEMENT: "Project management and task decomposition",
            PlanningDomain.COOKING: "Recipe generation with ingredient and step management",
            PlanningDomain.TRAVEL: "Travel itinerary creation and optimization",
            PlanningDomain.PROBLEM_SOLVING: "Multi-step reasoning and problem decomposition",
            PlanningDomain.STRATEGY: "Game theory, decision making, and strategic planning",
            PlanningDomain.CONSTRAINT_SATISFACTION: "Satisfying multiple competing constraints",
            PlanningDomain.CAUSAL_REASONING: "Cause-effect chain analysis and prediction",
        }
        return _desc[self]


class PlanComplexity(Enum):
    """Complexity tier for a planning problem."""

    SIMPLE = auto()
    MODERATE = auto()
    COMPLEX = auto()
    EXPERT = auto()

    def __repr__(self) -> str:
        return f"PlanComplexity.{self.name}"

    @property
    def expected_steps(self) -> Tuple[int, int]:
        """Return (min, max) expected number of plan steps."""
        _ranges = {
            PlanComplexity.SIMPLE: (3, 7),
            PlanComplexity.MODERATE: (5, 15),
            PlanComplexity.COMPLEX: (10, 30),
            PlanComplexity.EXPERT: (20, 60),
        }
        return _ranges[self]


class ReasoningPattern(Enum):
    """Type of reasoning observed in a chain."""

    DEDUCTIVE = auto()
    INDUCTIVE = auto()
    ABDUCTIVE = auto()
    ANALOGICAL = auto()
    CAUSAL = auto()
    UNKNOWN = auto()

    def __repr__(self) -> str:
        return f"ReasoningPattern.{self.name}"


class PlanStructureType(Enum):
    """Structural topology of a plan."""

    LINEAR = auto()
    BRANCHING = auto()
    PARALLEL = auto()
    HIERARCHICAL = auto()
    CYCLIC = auto()
    MIXED = auto()

    def __repr__(self) -> str:
        return f"PlanStructureType.{self.name}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step in a plan.

    Parameters
    ----------
    action : str
        Description of what is done in this step.
    preconditions : List[str]
        Conditions that must hold before this step can execute.
    effects : List[str]
        Conditions that become true after this step executes.
    order : int
        Ordinal position of this step in the plan.
    dependencies : List[int]
        Indices of prior steps that must complete first.
    duration : float
        Estimated time for this step (arbitrary units).
    resources : List[str]
        Resources consumed or required by this step.
    metadata : Dict[str, Any]
        Arbitrary extra information.
    """

    action: str = ""
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    order: int = 0
    dependencies: List[int] = field(default_factory=list)
    duration: float = 1.0
    resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "order": self.order,
            "dependencies": list(self.dependencies),
            "duration": self.duration,
            "resources": list(self.resources),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            action=data.get("action", ""),
            preconditions=data.get("preconditions", []),
            effects=data.get("effects", []),
            order=data.get("order", 0),
            dependencies=data.get("dependencies", []),
            duration=data.get("duration", 1.0),
            resources=data.get("resources", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def action_words(self) -> List[str]:
        """Tokenise the action string into lower-case words."""
        return self.action.lower().split()

    @property
    def has_dependencies(self) -> bool:
        return len(self.dependencies) > 0


@dataclass
class PlanStructure:
    """A complete plan consisting of ordered steps.

    Parameters
    ----------
    steps : List[PlanStep]
        The ordered steps of the plan.
    goal : str
        The goal the plan is designed to achieve.
    constraints : List[str]
        Textual constraints that the plan must satisfy.
    metadata : Dict[str, Any]
        Arbitrary extra data (domain, source, etc.).
    """

    steps: List[PlanStep] = field(default_factory=list)
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "goal": self.goal,
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStructure":
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            steps=steps,
            goal=data.get("goal", ""),
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.steps)

    @property
    def all_resources(self) -> Set[str]:
        resources: Set[str] = set()
        for step in self.steps:
            resources.update(step.resources)
        return resources

    def dependency_graph(self) -> Dict[int, List[int]]:
        """Return adjacency list: step_order -> list of dependent step orders."""
        graph: Dict[int, List[int]] = {s.order: [] for s in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep in graph:
                    graph[dep].append(step.order)
        return graph


# ---------------------------------------------------------------------------
# PlanningConfig
# ---------------------------------------------------------------------------


@dataclass
class PlanningConfig(TaskConfig):
    """Configuration specific to planning / reasoning tasks.

    Extends :class:`TaskConfig` with planning-specific fields.
    """

    planning_domains: List[str] = field(
        default_factory=lambda: ["logistics", "project_management", "cooking", "travel"]
    )
    complexity_levels: List[str] = field(
        default_factory=lambda: ["simple", "moderate", "complex"]
    )
    max_plan_steps: int = 50
    min_plan_steps: int = 3
    require_dependencies: bool = True
    require_preconditions: bool = False
    enable_causal_reasoning: bool = True
    enable_strategy_tasks: bool = True
    diversity_weight: float = 0.3
    quality_weight: float = 0.7
    constraint_strictness: float = 0.8
    prompts_per_domain: int = 10

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "planning_domains": list(self.planning_domains),
            "complexity_levels": list(self.complexity_levels),
            "max_plan_steps": self.max_plan_steps,
            "min_plan_steps": self.min_plan_steps,
            "require_dependencies": self.require_dependencies,
            "require_preconditions": self.require_preconditions,
            "enable_causal_reasoning": self.enable_causal_reasoning,
            "enable_strategy_tasks": self.enable_strategy_tasks,
            "diversity_weight": self.diversity_weight,
            "quality_weight": self.quality_weight,
            "constraint_strictness": self.constraint_strictness,
            "prompts_per_domain": self.prompts_per_domain,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningConfig":
        constraints = [
            TaskConstraint.from_dict(c) for c in data.get("constraints", [])
        ]
        return cls(
            name=data.get("name", "planning"),
            domain=TaskDomain[data.get("domain", "REASONING")],
            num_prompts=data.get("num_prompts", 100),
            max_length=data.get("max_length", 1024),
            min_length=data.get("min_length", 50),
            temperature=data.get("temperature", 1.0),
            constraints=constraints,
            evaluation_metrics=data.get(
                "evaluation_metrics",
                ["completeness", "ordering", "feasibility", "efficiency", "diversity"],
            ),
            prompt_template=data.get("prompt_template", "{text}"),
            seed=data.get("seed", 42),
            planning_domains=data.get(
                "planning_domains",
                ["logistics", "project_management", "cooking", "travel"],
            ),
            complexity_levels=data.get(
                "complexity_levels", ["simple", "moderate", "complex"]
            ),
            max_plan_steps=data.get("max_plan_steps", 50),
            min_plan_steps=data.get("min_plan_steps", 3),
            require_dependencies=data.get("require_dependencies", True),
            require_preconditions=data.get("require_preconditions", False),
            enable_causal_reasoning=data.get("enable_causal_reasoning", True),
            enable_strategy_tasks=data.get("enable_strategy_tasks", True),
            diversity_weight=data.get("diversity_weight", 0.3),
            quality_weight=data.get("quality_weight", 0.7),
            constraint_strictness=data.get("constraint_strictness", 0.8),
            prompts_per_domain=data.get("prompts_per_domain", 10),
        )

    def validate(self) -> List[str]:
        errors = super().validate()
        if self.max_plan_steps < self.min_plan_steps:
            errors.append("max_plan_steps must be >= min_plan_steps")
        if not self.planning_domains:
            errors.append("At least one planning domain must be specified")
        if not (0.0 <= self.diversity_weight <= 1.0):
            errors.append("diversity_weight must be in [0, 1]")
        if not (0.0 <= self.quality_weight <= 1.0):
            errors.append("quality_weight must be in [0, 1]")
        if not (0.0 <= self.constraint_strictness <= 1.0):
            errors.append("constraint_strictness must be in [0, 1]")
        return errors


# ---------------------------------------------------------------------------
# Built-in prompt templates  (25+ scenarios)
# ---------------------------------------------------------------------------

_PLANNING_SCENARIOS: List[Dict[str, Any]] = [
    # --- Logistics (1-5) ---
    {
        "id": "logistics_delivery",
        "domain": "logistics",
        "complexity": "moderate",
        "title": "Multi-Stop Delivery Route",
        "prompt": (
            "Plan an optimal delivery route for a courier who must deliver "
            "packages to 8 locations in a city. The courier starts at a central "
            "warehouse at 8 AM and must complete all deliveries by 5 PM. Some "
            "deliveries have specific time windows. Consider traffic patterns, "
            "package sizes, and fuel efficiency."
        ),
        "constraints": [
            "All deliveries must be completed within business hours",
            "Perishable items must be delivered first",
            "No more than 3 packages at a time due to vehicle capacity",
        ],
        "goal": "Complete all deliveries efficiently within time constraints",
    },
    {
        "id": "logistics_warehouse",
        "domain": "logistics",
        "complexity": "complex",
        "title": "Warehouse Reorganisation",
        "prompt": (
            "Design a plan to reorganise a 50,000 sq ft warehouse that stores "
            "electronics, furniture, and food products. The warehouse must remain "
            "partially operational during the reorganisation. High-demand items "
            "should be placed nearest to the loading docks. The plan should "
            "minimise disruption to ongoing fulfillment operations."
        ),
        "constraints": [
            "Food products must be in climate-controlled zones",
            "At least 60% of orders must still be fulfillable during reorganisation",
            "Heavy furniture must be on ground level",
            "Electronics require anti-static shelving",
        ],
        "goal": "Reorganise warehouse for maximum pick efficiency",
    },
    {
        "id": "logistics_scheduling",
        "domain": "logistics",
        "complexity": "moderate",
        "title": "Employee Shift Scheduling",
        "prompt": (
            "Create a weekly shift schedule for a 24/7 call centre with 20 "
            "employees. Each shift is 8 hours. Employees have varying skill "
            "levels, availability constraints, and maximum weekly hour limits. "
            "Peak hours (9 AM-5 PM) require at least 6 staff members, while "
            "off-peak hours require at least 2."
        ),
        "constraints": [
            "No employee works more than 40 hours per week",
            "At least one senior employee per shift",
            "8-hour minimum rest between consecutive shifts",
            "Weekend shifts require volunteer-only staffing",
        ],
        "goal": "Create a fair and complete shift schedule",
    },
    {
        "id": "logistics_supply_chain",
        "domain": "logistics",
        "complexity": "complex",
        "title": "Supply Chain Contingency",
        "prompt": (
            "A manufacturing company relies on three overseas suppliers for "
            "critical components. One supplier has just experienced a natural "
            "disaster and will be offline for 3 months. Develop a plan to "
            "maintain production levels, considering alternative suppliers, "
            "inventory buffers, product redesign options, and customer "
            "communication strategies."
        ),
        "constraints": [
            "Production cannot drop below 70% of normal output",
            "Quality standards must be maintained",
            "Budget for contingency cannot exceed 20% premium",
            "Regulatory compliance for new suppliers required",
        ],
        "goal": "Maintain production continuity during supplier disruption",
    },
    {
        "id": "logistics_event",
        "domain": "logistics",
        "complexity": "simple",
        "title": "Event Resource Allocation",
        "prompt": (
            "Plan the resource allocation for a one-day outdoor music "
            "festival with 5,000 expected attendees. Allocate stages, food "
            "vendors, restrooms, parking, medical stations, and security "
            "personnel across the venue grounds."
        ),
        "constraints": [
            "Medical stations within 2-minute walk of any location",
            "Food vendors not adjacent to restroom facilities",
            "Main stage visible from at least 60% of venue",
        ],
        "goal": "Optimally allocate resources for attendee experience and safety",
    },
    # --- Project Management (6-9) ---
    {
        "id": "project_software",
        "domain": "project_management",
        "complexity": "complex",
        "title": "Software Product Launch",
        "prompt": (
            "Plan the launch of a new SaaS product from beta to general "
            "availability. The team has 12 engineers, 3 designers, 2 PMs, and "
            "a 4-month timeline. Key milestones include feature freeze, QA "
            "testing, documentation, marketing preparation, beta feedback "
            "integration, infrastructure scaling, and launch day operations."
        ),
        "constraints": [
            "Feature freeze must occur at least 6 weeks before launch",
            "All P0 bugs must be resolved before GA",
            "Load testing must demonstrate 10x current capacity",
            "Documentation must cover all user-facing features",
        ],
        "goal": "Successfully launch product on schedule with quality standards met",
    },
    {
        "id": "project_renovation",
        "domain": "project_management",
        "complexity": "moderate",
        "title": "Office Renovation Project",
        "prompt": (
            "Plan the renovation of a 3-floor office building while 200 "
            "employees continue working. The renovation includes new HVAC, "
            "open floor plan conversion, server room upgrade, and kitchen "
            "remodel. Budget is $2 million and the deadline is 6 months."
        ),
        "constraints": [
            "No more than one floor under construction at a time",
            "Server room must have zero downtime",
            "Noise-generating work limited to evenings and weekends",
            "Fire safety compliance at all times",
        ],
        "goal": "Complete renovation on time and budget with minimal disruption",
    },
    {
        "id": "project_research",
        "domain": "project_management",
        "complexity": "moderate",
        "title": "Research Grant Execution",
        "prompt": (
            "Plan the execution of a 2-year research grant studying urban "
            "microplastic pollution. The team includes 2 principal "
            "investigators, 3 graduate students, and 1 lab technician. "
            "Deliverables include quarterly reports, a mid-term review, "
            "3 journal publications, and a public presentation."
        ),
        "constraints": [
            "IRB approval required before field sampling",
            "Budget spending must follow institutional guidelines",
            "First publication submitted within 12 months",
            "All data must be archived in open-access repository",
        ],
        "goal": "Complete all deliverables and advance scientific knowledge",
    },
    {
        "id": "project_migration",
        "domain": "project_management",
        "complexity": "expert",
        "title": "Cloud Migration",
        "prompt": (
            "Plan the migration of a legacy on-premise enterprise system "
            "to cloud infrastructure. The system serves 50,000 daily active "
            "users, includes 15 microservices, 4 databases, and integrates "
            "with 8 third-party APIs. The migration must achieve zero "
            "downtime and maintain data integrity throughout."
        ),
        "constraints": [
            "Zero data loss during migration",
            "Maximum 5 minutes of degraded performance",
            "All compliance certifications must transfer",
            "Rollback capability required at each phase",
            "Cost must not exceed 150% of current annual infrastructure spend",
        ],
        "goal": "Migrate fully to cloud with zero downtime and data integrity",
    },
    # --- Cooking (10-12) ---
    {
        "id": "cooking_dinner_party",
        "domain": "cooking",
        "complexity": "moderate",
        "title": "Five-Course Dinner Party",
        "prompt": (
            "Plan a five-course dinner party for 8 guests. Design the menu "
            "with appetizer, soup, main course, salad, and dessert. Include "
            "a detailed preparation timeline starting 3 days before the "
            "event, ingredient shopping list, and day-of cooking schedule. "
            "One guest is vegan and another has a nut allergy."
        ),
        "constraints": [
            "All courses must be nut-free",
            "At least one vegan option per course",
            "All hot dishes served within 5 minutes of preparation",
            "Total food budget under $200",
        ],
        "goal": "Execute a seamless multi-course dinner with dietary accommodations",
    },
    {
        "id": "cooking_meal_prep",
        "domain": "cooking",
        "complexity": "simple",
        "title": "Weekly Meal Prep",
        "prompt": (
            "Plan a weekly meal prep session to prepare lunches and dinners "
            "for one person for 5 weekdays. Prioritise nutrition balance, "
            "variety, and ingredients that can be batch-cooked. The total "
            "prep time should not exceed 4 hours on Sunday."
        ),
        "constraints": [
            "Each meal must include protein, carbs, and vegetables",
            "No ingredient should repeat more than 3 times across meals",
            "Total grocery cost under $75",
            "All meals must be refrigerator-safe for 5 days",
        ],
        "goal": "Efficient weekly meal preparation with nutritional balance",
    },
    {
        "id": "cooking_baking_competition",
        "domain": "cooking",
        "complexity": "complex",
        "title": "Multi-Layer Cake Challenge",
        "prompt": (
            "Plan the creation of a three-tier celebration cake with "
            "different flavours per tier: chocolate, lemon, and strawberry. "
            "Include fondant decorations and a structural support system. "
            "The cake must be completed in 8 hours and serve 50 people. "
            "Plan the baking, assembly, and decoration sequence."
        ),
        "constraints": [
            "Tiers must cool completely before stacking",
            "Fondant must be applied at room temperature",
            "Structural dowels required between tiers",
            "Must be transportable to venue 30 minutes away",
        ],
        "goal": "Create a structurally sound and visually stunning celebration cake",
    },
    # --- Travel (13-15) ---
    {
        "id": "travel_europe",
        "domain": "travel",
        "complexity": "complex",
        "title": "Three-Week European Tour",
        "prompt": (
            "Plan a 21-day trip through Western Europe visiting Paris, "
            "Barcelona, Rome, and Amsterdam. Two travellers with a budget "
            "of $6,000 total (excluding flights to/from home). Include "
            "accommodation, transportation between cities, daily itineraries "
            "with key attractions, dining recommendations, and rest days."
        ),
        "constraints": [
            "At least 4 days in each city",
            "Use trains between cities for sustainability",
            "Include at least 2 rest days total",
            "All accommodations must have reviews above 4 stars",
            "Budget split roughly 40% accommodation, 30% food, 30% activities",
        ],
        "goal": "Create a memorable and well-paced European travel experience",
    },
    {
        "id": "travel_business",
        "domain": "travel",
        "complexity": "simple",
        "title": "Business Trip Optimisation",
        "prompt": (
            "Plan a 3-day business trip to New York City. The traveller has "
            "meetings at 9 AM and 2 PM on Day 1, a conference all day on Day "
            "2, and a client lunch on Day 3 before a 6 PM flight home. "
            "Include hotel selection, transport, meal planning, and one "
            "evening activity."
        ),
        "constraints": [
            "Hotel within 15 minutes of meeting locations",
            "Corporate travel policy limits hotel to $300/night",
            "All expenses must be receipt-documented",
            "Arrive at airport 2 hours before flight",
        ],
        "goal": "Maximise business productivity while maintaining comfort",
    },
    {
        "id": "travel_family",
        "domain": "travel",
        "complexity": "moderate",
        "title": "Family Vacation Planning",
        "prompt": (
            "Plan a 10-day family vacation for 2 adults and 3 children "
            "(ages 5, 8, and 13) to a beach destination. Include flights, "
            "accommodation, daily activities suitable for all ages, dining "
            "options, and a budget breakdown. The total budget is $5,000."
        ),
        "constraints": [
            "Activities must be suitable for the 5-year-old",
            "Include at least 2 educational excursions for the 13-year-old",
            "Accommodation must have kitchen facilities",
            "Maximum 4-hour flight time",
        ],
        "goal": "Create a fun and relaxing family vacation for all ages",
    },
    # --- Problem Solving (16-18) ---
    {
        "id": "problem_water",
        "domain": "problem_solving",
        "complexity": "moderate",
        "title": "Community Water Crisis",
        "prompt": (
            "A small town of 10,000 residents has discovered contamination "
            "in its primary water source. Develop a multi-step plan to "
            "address the immediate crisis (safe drinking water), medium-term "
            "solution (water treatment), and long-term prevention (source "
            "protection and infrastructure upgrade)."
        ),
        "constraints": [
            "Immediate safe water within 24 hours",
            "Solution must be affordable for the municipal budget",
            "Must comply with EPA water quality standards",
            "Plan must address communication with residents",
        ],
        "goal": "Resolve water contamination at all time horizons",
    },
    {
        "id": "problem_traffic",
        "domain": "problem_solving",
        "complexity": "complex",
        "title": "Urban Traffic Congestion",
        "prompt": (
            "A mid-size city of 500,000 people faces severe traffic "
            "congestion that adds an average of 45 minutes to daily "
            "commutes. Develop a comprehensive plan addressing "
            "infrastructure changes, public transit improvements, policy "
            "measures, technology integration, and behavioural incentives."
        ),
        "constraints": [
            "Infrastructure changes must be phased over 5 years",
            "Must not displace existing residents or businesses",
            "Public transit must serve all socioeconomic areas equally",
            "Plan must show measurable reduction targets",
        ],
        "goal": "Reduce average commute time by at least 30%",
    },
    {
        "id": "problem_energy",
        "domain": "problem_solving",
        "complexity": "expert",
        "title": "Campus Energy Transition",
        "prompt": (
            "Plan the transition of a university campus (30 buildings, "
            "15,000 occupants) from fossil fuels to 100% renewable energy "
            "within 10 years. Consider solar, wind, geothermal, energy "
            "storage, grid integration, building retrofits, and behavioural "
            "change programs."
        ),
        "constraints": [
            "No interruption to campus operations",
            "Return on investment within 15 years",
            "Meet all local building codes and regulations",
            "Reduce total energy consumption by 20% simultaneously",
            "Include student and faculty engagement component",
        ],
        "goal": "Achieve 100% renewable energy while reducing consumption",
    },
    # --- Strategy (19-21) ---
    {
        "id": "strategy_market_entry",
        "domain": "strategy",
        "complexity": "complex",
        "title": "New Market Entry Strategy",
        "prompt": (
            "A mid-size tech company wants to enter the Southeast Asian "
            "market with its project management SaaS product. Develop a "
            "market entry strategy covering market analysis, localisation, "
            "pricing, partnerships, marketing channels, hiring, and legal "
            "compliance. The company has $5M allocated for the first year."
        ),
        "constraints": [
            "Must launch in at least 3 countries within 12 months",
            "Product must support local languages",
            "Pricing must account for purchasing power parity",
            "Must comply with each country's data sovereignty laws",
        ],
        "goal": "Establish a profitable presence in Southeast Asian market",
    },
    {
        "id": "strategy_negotiation",
        "domain": "strategy",
        "complexity": "moderate",
        "title": "Multi-Party Negotiation",
        "prompt": (
            "Three neighbouring towns must negotiate shared use of a "
            "regional park. Town A wants to expand recreational facilities, "
            "Town B wants to preserve natural habitat, and Town C wants "
            "to develop commercial tourism. Develop a negotiation strategy "
            "that identifies interests, generates options, and proposes "
            "fair division mechanisms."
        ),
        "constraints": [
            "All three towns must agree to the final plan",
            "Environmental impact assessment required",
            "Budget contributions proportional to town population",
            "No single town controls more than 40% of decision-making",
        ],
        "goal": "Reach a mutually acceptable agreement on park usage",
    },
    {
        "id": "strategy_competitive",
        "domain": "strategy",
        "complexity": "moderate",
        "title": "Competitive Response Strategy",
        "prompt": (
            "A leading e-commerce company discovers that a well-funded "
            "competitor is about to launch a same-day delivery service in "
            "its core markets. Develop a competitive response strategy "
            "covering defensive moves, counter-offers, differentiation, "
            "and potential collaboration or acquisition scenarios."
        ),
        "constraints": [
            "Response must be implementable within 90 days",
            "Cannot engage in anti-competitive practices",
            "Must maintain current profit margins",
            "Customer experience must not degrade during transition",
        ],
        "goal": "Maintain market position against competitive threat",
    },
    # --- Constraint Satisfaction (22-24) ---
    {
        "id": "csp_course_scheduling",
        "domain": "constraint_satisfaction",
        "complexity": "complex",
        "title": "University Course Scheduling",
        "prompt": (
            "Schedule 40 courses across 10 classrooms over a 5-day week. "
            "Each course meets 2 or 3 times per week for 50 or 75 minutes. "
            "Professors have availability constraints, some courses require "
            "specific room types (labs, lecture halls), and popular courses "
            "must not conflict with each other."
        ),
        "constraints": [
            "No professor teaches overlapping time slots",
            "Room capacity must meet expected enrollment",
            "Lab courses only in rooms with appropriate equipment",
            "No student should have more than 3 consecutive hours of class",
            "At least one section of each required course offered before noon",
        ],
        "goal": "Create a conflict-free schedule satisfying all constraints",
    },
    {
        "id": "csp_seating",
        "domain": "constraint_satisfaction",
        "complexity": "moderate",
        "title": "Wedding Seating Arrangement",
        "prompt": (
            "Arrange 120 wedding guests across 15 round tables of 8 seats "
            "each. Couples must sit together, feuding family members must "
            "be separated, the bridal party has a head table, children "
            "should be grouped near parents, and there should be a mix of "
            "family and friends at each non-head table."
        ),
        "constraints": [
            "All couples seated at the same table",
            "Feuding pairs at least 3 tables apart",
            "Head table seats bridal party of 8",
            "Children within 2 tables of their parents",
            "Each table has at least 2 people who know each other",
        ],
        "goal": "Create a harmonious seating arrangement respecting all social constraints",
    },
    {
        "id": "csp_budget",
        "domain": "constraint_satisfaction",
        "complexity": "simple",
        "title": "Event Budget Allocation",
        "prompt": (
            "Allocate a $50,000 budget across 8 departments for a company "
            "annual retreat. Departments include: venue, catering, "
            "entertainment, transportation, accommodation, team-building "
            "activities, gifts, and contingency. Each department has "
            "minimum requirements and historical spending data."
        ),
        "constraints": [
            "Venue must receive at least 25% of budget",
            "Catering at least $80 per person for 100 attendees",
            "Contingency fund minimum 10%",
            "No department receives more than 30%",
            "Transportation cost depends on venue distance",
        ],
        "goal": "Optimally allocate budget satisfying all department needs",
    },
    # --- Causal Reasoning (25-27) ---
    {
        "id": "causal_ecosystem",
        "domain": "causal_reasoning",
        "complexity": "complex",
        "title": "Ecosystem Disruption Analysis",
        "prompt": (
            "Analyse the causal chain of effects when an invasive predatory "
            "fish species is introduced into a freshwater lake ecosystem. "
            "Trace the cascading impacts on native fish populations, "
            "aquatic plants, water quality, bird populations, local fishing "
            "industry, and recreational use. Propose intervention strategies."
        ),
        "constraints": [
            "Must identify at least 5 causal pathways",
            "Include both direct and indirect effects",
            "Consider feedback loops",
            "Interventions must be ecologically sound",
        ],
        "goal": "Map causal chains and propose effective ecosystem interventions",
    },
    {
        "id": "causal_economic",
        "domain": "causal_reasoning",
        "complexity": "moderate",
        "title": "Economic Policy Impact",
        "prompt": (
            "Trace the causal effects of raising the national minimum wage "
            "by 25%. Analyse impacts on employment, consumer spending, "
            "small businesses, inflation, income inequality, government "
            "tax revenue, and social welfare programs. Consider short-term "
            "vs long-term effects and regional variations."
        ),
        "constraints": [
            "Distinguish correlation from causation",
            "Account for time-delayed effects",
            "Include counter-arguments for each causal claim",
            "Reference plausible economic mechanisms",
        ],
        "goal": "Provide a balanced causal analysis of minimum wage increase",
    },
    {
        "id": "causal_technology",
        "domain": "causal_reasoning",
        "complexity": "moderate",
        "title": "Technology Adoption Cascade",
        "prompt": (
            "Analyse the causal chain of effects when a major city adopts "
            "autonomous public transit. Trace effects on employment, urban "
            "planning, real estate, energy consumption, traffic safety, "
            "disability access, social equity, and adjacent transportation "
            "industries."
        ),
        "constraints": [
            "Include at least 3 positive and 3 negative causal chains",
            "Identify potential unintended consequences",
            "Consider interactions between causal pathways",
            "Propose monitoring indicators for each effect",
        ],
        "goal": "Map comprehensive causal network of autonomous transit adoption",
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_action_verbs(text: str) -> List[str]:
    """Extract likely action verbs from a plan step description.

    Uses a curated set of common planning action verbs and returns those
    found in the text, preserving order of first occurrence.
    """
    action_verbs = {
        "plan", "schedule", "allocate", "assign", "build", "create",
        "design", "develop", "implement", "install", "launch", "manage",
        "monitor", "optimise", "optimize", "organise", "organize",
        "prepare", "prioritise", "prioritize", "procure", "purchase",
        "recruit", "remove", "replace", "review", "select", "set",
        "setup", "start", "stop", "test", "train", "transport",
        "update", "upgrade", "verify", "coordinate", "execute",
        "evaluate", "inspect", "maintain", "measure", "negotiate",
        "order", "pack", "deliver", "cook", "bake", "mix", "chop",
        "boil", "fry", "grill", "roast", "serve", "clean", "wash",
        "book", "reserve", "confirm", "cancel", "check", "travel",
        "drive", "fly", "walk", "navigate", "explore", "visit",
        "research", "analyse", "analyze", "calculate", "compare",
        "decide", "determine", "estimate", "forecast", "identify",
        "investigate", "solve", "study", "gather", "collect",
        "assemble", "construct", "deploy", "distribute", "establish",
        "facilitate", "integrate", "migrate", "prototype", "refine",
        "restructure", "consolidate", "communicate", "delegate",
        "document", "present", "report", "submit", "approve",
    }
    words = re.findall(r"\b[a-z]+\b", text.lower())
    seen: Set[str] = set()
    result: List[str] = []
    for w in words:
        if w in action_verbs and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def extract_temporal_ordering(text: str) -> List[Tuple[str, int]]:
    """Extract temporal markers and their approximate positions in the text.

    Returns a list of (marker, char_position) tuples sorted by position.
    """
    temporal_patterns = [
        r"\b(first|firstly)\b",
        r"\b(second|secondly)\b",
        r"\b(third|thirdly)\b",
        r"\b(fourth)\b",
        r"\b(fifth)\b",
        r"\b(then|next|after\s+that|afterwards|subsequently)\b",
        r"\b(finally|lastly|last)\b",
        r"\b(before|prior\s+to|preceding)\b",
        r"\b(after|following|once)\b",
        r"\b(meanwhile|simultaneously|concurrently|at\s+the\s+same\s+time)\b",
        r"\b(initially|to\s+begin|to\s+start)\b",
        r"\b(step\s+\d+)\b",
        r"\b(phase\s+\d+)\b",
        r"\b(day\s+\d+)\b",
        r"\b(week\s+\d+)\b",
        r"\b(month\s+\d+)\b",
    ]
    markers: List[Tuple[str, int]] = []
    text_lower = text.lower()
    for pattern in temporal_patterns:
        for m in re.finditer(pattern, text_lower):
            markers.append((m.group(0).strip(), m.start()))
    markers.sort(key=lambda x: x[1])
    return markers


def compute_plan_similarity(plan_a: str, plan_b: str) -> float:
    """Compute similarity between two plan texts using word overlap and
    structural comparison.

    Returns a score in [0, 1] where 1 means identical plans.
    """
    if not plan_a.strip() or not plan_b.strip():
        return 0.0

    # Word-level Jaccard similarity
    words_a = set(plan_a.lower().split())
    words_b = set(plan_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    jaccard = len(words_a & words_b) / len(union)

    # Action verb overlap
    verbs_a = set(extract_action_verbs(plan_a))
    verbs_b = set(extract_action_verbs(plan_b))
    verb_union = verbs_a | verbs_b
    verb_sim = len(verbs_a & verbs_b) / len(verb_union) if verb_union else 1.0

    # Structural similarity: compare number of numbered/bulleted items
    steps_a = len(re.findall(r"(?m)^\s*(?:\d+[.)]|\-|\*)\s+", plan_a))
    steps_b = len(re.findall(r"(?m)^\s*(?:\d+[.)]|\-|\*)\s+", plan_b))
    max_steps = max(steps_a, steps_b, 1)
    step_sim = 1.0 - abs(steps_a - steps_b) / max_steps

    # Temporal marker overlap
    temp_a = set(t[0] for t in extract_temporal_ordering(plan_a))
    temp_b = set(t[0] for t in extract_temporal_ordering(plan_b))
    temp_union = temp_a | temp_b
    temp_sim = len(temp_a & temp_b) / len(temp_union) if temp_union else 1.0

    # Weighted combination
    similarity = 0.4 * jaccard + 0.25 * verb_sim + 0.2 * step_sim + 0.15 * temp_sim
    return float(np.clip(similarity, 0.0, 1.0))


def detect_plan_structure(text: str) -> PlanStructureType:
    """Heuristically detect the structural topology of a plan from free text.

    Returns the most likely :class:`PlanStructureType`.
    """
    text_lower = text.lower()

    parallel_indicators = [
        "simultaneously", "in parallel", "at the same time",
        "concurrently", "meanwhile", "while also",
    ]
    branching_indicators = [
        "if ", "alternatively", "option a", "option b", "either",
        "depending on", "in case", "otherwise", "or else",
    ]
    hierarchical_indicators = [
        "sub-task", "subtask", "phase", "milestone", "work package",
        "sub-step", "decompose", "break down into",
    ]
    cyclic_indicators = [
        "repeat", "iterate", "loop", "cycle", "revisit", "go back to",
        "recur", "re-evaluate", "retry",
    ]

    scores = {
        PlanStructureType.PARALLEL: 0,
        PlanStructureType.BRANCHING: 0,
        PlanStructureType.HIERARCHICAL: 0,
        PlanStructureType.CYCLIC: 0,
        PlanStructureType.LINEAR: 0,
    }

    for ind in parallel_indicators:
        if ind in text_lower:
            scores[PlanStructureType.PARALLEL] += 1

    for ind in branching_indicators:
        if ind in text_lower:
            scores[PlanStructureType.BRANCHING] += 1

    for ind in hierarchical_indicators:
        if ind in text_lower:
            scores[PlanStructureType.HIERARCHICAL] += 1

    for ind in cyclic_indicators:
        if ind in text_lower:
            scores[PlanStructureType.CYCLIC] += 1

    # Check for linear indicators: numbered lists without branching
    numbered_steps = re.findall(r"(?m)^\s*\d+[.)]\s+", text)
    if len(numbered_steps) >= 3:
        scores[PlanStructureType.LINEAR] += 2

    # If multiple non-linear structures detected, call it mixed
    non_linear_scores = {
        k: v for k, v in scores.items()
        if k != PlanStructureType.LINEAR and v > 0
    }
    if len(non_linear_scores) >= 2:
        total_non_linear = sum(non_linear_scores.values())
        if total_non_linear >= 3:
            return PlanStructureType.MIXED

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return PlanStructureType.LINEAR
    return best


def validate_dependency_graph(steps: List[PlanStep]) -> Tuple[bool, List[str]]:
    """Check that the dependency graph among steps is a valid DAG.

    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []
    order_set = {s.order for s in steps}

    # Check that all dependencies reference existing steps
    for step in steps:
        for dep in step.dependencies:
            if dep not in order_set:
                errors.append(
                    f"Step {step.order} depends on non-existent step {dep}"
                )
            if dep == step.order:
                errors.append(f"Step {step.order} depends on itself")

    # Check for cycles using DFS
    adj: Dict[int, List[int]] = defaultdict(list)
    for step in steps:
        for dep in step.dependencies:
            adj[dep].append(step.order)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[int, int] = {s.order: WHITE for s in steps}

    def _dfs(node: int) -> bool:
        colour[node] = GREY
        for neighbour in adj.get(node, []):
            if colour.get(neighbour, WHITE) == GREY:
                errors.append(
                    f"Cycle detected involving steps {node} and {neighbour}"
                )
                return False
            if colour.get(neighbour, WHITE) == WHITE:
                if not _dfs(neighbour):
                    return False
        colour[node] = BLACK
        return True

    for step in steps:
        if colour[step.order] == WHITE:
            _dfs(step.order)

    return len(errors) == 0, errors


def topological_sort(steps: List[PlanStep]) -> List[PlanStep]:
    """Return a topological ordering of plan steps based on dependencies.

    Uses Kahn's algorithm.  If the graph has cycles the function returns
    as many steps as it can order and logs a warning.
    """
    in_degree: Dict[int, int] = {s.order: 0 for s in steps}
    adj: Dict[int, List[int]] = defaultdict(list)
    step_map: Dict[int, PlanStep] = {s.order: s for s in steps}

    for step in steps:
        for dep in step.dependencies:
            if dep in step_map:
                adj[dep].append(step.order)
                in_degree[step.order] += 1

    queue: deque = deque()
    for order, deg in in_degree.items():
        if deg == 0:
            queue.append(order)

    sorted_orders: List[int] = []
    while queue:
        node = queue.popleft()
        sorted_orders.append(node)
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(sorted_orders) < len(steps):
        logger.warning(
            "Dependency graph contains cycles; topological sort is partial "
            "(%d of %d steps ordered)", len(sorted_orders), len(steps),
        )
        remaining = [s for s in steps if s.order not in set(sorted_orders)]
        return [step_map[o] for o in sorted_orders] + remaining

    return [step_map[o] for o in sorted_orders]


def compute_critical_path(steps: List[PlanStep]) -> Tuple[float, List[int]]:
    """Compute the critical path length and the sequence of step orders.

    The critical path is the longest path through the dependency DAG where
    path length is the sum of step durations.

    Returns (total_duration, list_of_step_orders_on_critical_path).
    """
    step_map: Dict[int, PlanStep] = {s.order: s for s in steps}
    sorted_steps = topological_sort(steps)

    # dist[order] = (longest distance to this node, predecessor)
    dist: Dict[int, Tuple[float, Optional[int]]] = {}
    for step in sorted_steps:
        dist[step.order] = (step.duration, None)

    for step in sorted_steps:
        current_dist = dist[step.order][0]
        # Look at steps that depend on this one
        for other in steps:
            if step.order in other.dependencies:
                new_dist = current_dist + other.duration
                if new_dist > dist[other.order][0]:
                    dist[other.order] = (new_dist, step.order)

    if not dist:
        return 0.0, []

    # Find the end node with the longest path
    end_order = max(dist, key=lambda o: dist[o][0])
    total_duration = dist[end_order][0]

    # Trace back the path
    path: List[int] = []
    current: Optional[int] = end_order
    while current is not None:
        path.append(current)
        current = dist[current][1]

    path.reverse()
    return total_duration, path


def _text_to_bow_vector(text: str, vocabulary: Dict[str, int]) -> np.ndarray:
    """Convert text to a bag-of-words vector using the given vocabulary."""
    vec = np.zeros(len(vocabulary), dtype=np.float64)
    words = text.lower().split()
    for word in words:
        idx = vocabulary.get(word)
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _build_vocabulary(texts: List[str], max_vocab: int = 5000) -> Dict[str, int]:
    """Build a vocabulary mapping from the most frequent words."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab: Dict[str, int] = {}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab


def _extract_numbered_steps(text: str) -> List[str]:
    """Extract numbered or bulleted steps from plan text."""
    patterns = [
        r"(?m)^\s*(\d+)[.)]\s+(.+?)(?=\n\s*\d+[.)]|\n\s*$|\Z)",
        r"(?m)^\s*[-*]\s+(.+?)(?=\n\s*[-*]|\n\s*$|\Z)",
    ]
    steps: List[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            step_text = match[-1] if isinstance(match, tuple) else match
            step_text = step_text.strip()
            if step_text:
                steps.append(step_text)
        if steps:
            break
    if not steps:
        sentences = re.split(r"[.!?]+", text)
        steps = [s.strip() for s in sentences if len(s.strip()) > 10]
    return steps


# ---------------------------------------------------------------------------
# PlanningPromptGenerator
# ---------------------------------------------------------------------------


class PlanningPromptGenerator:
    """Generates diverse planning prompts across multiple domains.

    Uses the built-in scenario bank and can procedurally generate
    variations for any domain.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._scenarios = list(_PLANNING_SCENARIOS)

    # -----------------------------------------------------------------
    # Domain-specific generators
    # -----------------------------------------------------------------

    def generate_logistics_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate logistics planning prompts: routing, scheduling, resource allocation."""
        templates = [
            (
                "Plan the most efficient route for a fleet of {n_vehicles} delivery "
                "trucks departing from a central depot to deliver {n_packages} packages "
                "across {n_zones} city zones. Consider traffic, time windows, and fuel.",
                {"n_vehicles": [2, 3, 5], "n_packages": [15, 30, 50], "n_zones": [4, 6, 8]},
            ),
            (
                "Design a scheduling system for {n_workers} factory workers across "
                "{n_shifts} shifts to operate {n_machines} machines. Each machine "
                "requires a certified operator and must run at least {min_hours} hours daily.",
                {"n_workers": [12, 20, 30], "n_shifts": [2, 3], "n_machines": [5, 8, 12],
                 "min_hours": [6, 8, 10]},
            ),
            (
                "Allocate {n_ambulances} ambulances across {n_stations} stations in "
                "a city to minimise average emergency response time. Consider "
                "population density, historical call data, and hospital locations.",
                {"n_ambulances": [8, 12, 20], "n_stations": [4, 6, 10]},
            ),
            (
                "Plan the inventory restocking schedule for a grocery chain with "
                "{n_stores} stores and {n_suppliers} suppliers. Products have varying "
                "shelf lives and demand patterns. Minimise waste and stockouts.",
                {"n_stores": [5, 10, 25], "n_suppliers": [3, 6, 10]},
            ),
            (
                "Organise the loading sequence for a cargo ship carrying {n_containers} "
                "containers to {n_ports} ports. Containers must be accessible in "
                "port-visit order. Consider weight distribution and hazardous materials.",
                {"n_containers": [50, 100, 200], "n_ports": [3, 5, 7]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "logistics", complexity,
        )

    def generate_project_planning_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate project management and task decomposition prompts."""
        templates = [
            (
                "Create a project plan for developing a {app_type} application with "
                "a team of {team_size} people over {duration} months. Include milestones, "
                "task assignments, risk mitigation, and quality checkpoints.",
                {"app_type": ["mobile", "web", "desktop", "IoT"],
                 "team_size": [4, 8, 15], "duration": [3, 6, 12]},
            ),
            (
                "Plan the construction of a {building_type} in {location}. The project "
                "has a ${budget}M budget and must be completed in {months} months. "
                "Include procurement, permits, construction phases, and inspections.",
                {"building_type": ["community centre", "school", "clinic"],
                 "location": ["suburban area", "urban centre", "rural community"],
                 "budget": [2, 5, 10], "months": [12, 18, 24]},
            ),
            (
                "Develop an implementation plan for adopting {methodology} across an "
                "organisation of {org_size} employees. Include training, pilot programs, "
                "tool selection, change management, and success metrics.",
                {"methodology": ["agile", "DevOps", "lean manufacturing", "OKRs"],
                 "org_size": [50, 200, 1000]},
            ),
            (
                "Plan the merger integration of two {industry} companies with a combined "
                "workforce of {employees} employees. Address organisational structure, "
                "system consolidation, culture integration, and customer transition.",
                {"industry": ["technology", "healthcare", "financial services"],
                 "employees": [500, 2000, 5000]},
            ),
            (
                "Create a product roadmap for a {product_type} over the next {quarters} "
                "quarters. Balance new features, technical debt, performance improvements, "
                "and competitive responses. Include dependency mapping and resource needs.",
                {"product_type": ["CRM platform", "analytics dashboard", "collaboration tool"],
                 "quarters": [4, 6, 8]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "project_management", complexity,
        )

    def generate_cooking_recipe_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate cooking and recipe planning prompts with ingredient management."""
        templates = [
            (
                "Plan the preparation of a {cuisine} feast for {guests} guests. "
                "Create a menu with {courses} courses, a detailed shopping list, "
                "preparation timeline, and day-of cooking schedule. Account for "
                "one guest with {restriction} dietary restriction.",
                {"cuisine": ["Italian", "Japanese", "Mexican", "Indian", "French"],
                 "guests": [6, 10, 20], "courses": [3, 4, 5],
                 "restriction": ["gluten-free", "vegetarian", "dairy-free", "kosher"]},
            ),
            (
                "Design a {days}-day meal plan for a family of {family_size} on a "
                "${budget} weekly grocery budget. Include breakfast, lunch, dinner, "
                "and snacks. Maximise ingredient reuse across meals to minimise waste.",
                {"days": [5, 7], "family_size": [3, 4, 5],
                 "budget": [100, 150, 200]},
            ),
            (
                "Plan the baking of {n_items} different items for a charity bake "
                "sale using a single home oven. Items include {item_types}. "
                "Schedule oven usage, prep work, cooling, and packaging.",
                {"n_items": [6, 8, 12],
                 "item_types": [
                     "cookies, brownies, and bread",
                     "cakes, pies, and muffins",
                     "scones, tarts, and cupcakes",
                 ]},
            ),
            (
                "Create a recipe and execution plan for a {dish} that requires "
                "{technique} technique. Include ingredient sourcing, equipment "
                "needed, step-by-step instructions with timing, and plating.",
                {"dish": ["soufflé", "croissants", "ramen from scratch",
                          "beef Wellington", "macarons"],
                 "technique": ["lamination", "fermentation", "tempering",
                               "emulsification", "caramelisation"]},
            ),
            (
                "Plan a cooking class for {students} students covering {topic}. "
                "Include mise en place organisation, demo and practice schedule, "
                "safety briefing, equipment allocation, and clean-up protocol.",
                {"students": [8, 12, 20],
                 "topic": ["knife skills and stocks", "pastry fundamentals",
                           "Asian stir-fry techniques", "bread making basics"]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "cooking", complexity,
        )

    def generate_travel_planning_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate travel itinerary creation prompts."""
        templates = [
            (
                "Plan a {duration}-day trip to {destination} for {travelers} "
                "travellers with a ${budget} budget. Include flights, accommodation, "
                "daily itinerary with activities, restaurant recommendations, and "
                "transportation between sites.",
                {"duration": [5, 7, 14], "destination": [
                    "Japan", "Iceland", "Peru", "Morocco", "New Zealand",
                 ], "travelers": [1, 2, 4], "budget": [2000, 4000, 8000]},
            ),
            (
                "Design a {theme}-themed road trip covering {distance} miles over "
                "{days} days. Include route planning, overnight stops, fuel "
                "calculations, attraction schedules, and emergency contingencies.",
                {"theme": ["national parks", "historic sites", "culinary", "coastal"],
                 "distance": [500, 1000, 2000], "days": [3, 5, 10]},
            ),
            (
                "Plan a destination wedding in {location} for {guests} guests. "
                "Include travel logistics for guests, venue selection, "
                "accommodation blocks, pre-wedding events, and local activity "
                "recommendations for the wedding weekend.",
                {"location": ["Tuscany", "Bali", "Cancún", "Scottish Highlands"],
                 "guests": [30, 50, 100]},
            ),
            (
                "Organise a {duration}-day school field trip for {students} "
                "students to {destination}. Include educational objectives, "
                "transportation, supervision ratios, meal planning, safety "
                "protocols, and permission slip details.",
                {"duration": [1, 3, 5],
                 "students": [20, 30, 50],
                 "destination": ["Washington D.C.", "a marine biology centre",
                                 "a technology museum", "a national forest"]},
            ),
            (
                "Plan a {type} retreat for {attendees} people at a {setting} "
                "location for {days} days. Balance structured activities with "
                "free time. Include logistics, meals, facilitation schedule, "
                "and outcomes framework.",
                {"type": ["corporate team-building", "wellness", "creative writing",
                          "leadership development"],
                 "attendees": [10, 20, 40], "setting": ["mountain", "beach", "rural"],
                 "days": [2, 3, 5]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "travel", complexity,
        )

    def generate_problem_solving_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate multi-step reasoning and problem decomposition prompts."""
        templates = [
            (
                "A {org_type} is experiencing a {problem}. Develop a structured "
                "problem-solving plan using root cause analysis, stakeholder "
                "engagement, solution evaluation, implementation, and monitoring.",
                {"org_type": ["hospital", "university", "city council",
                              "non-profit organisation"],
                 "problem": [
                     "30% increase in staff turnover",
                     "declining customer satisfaction scores",
                     "budget shortfall of 15%",
                     "cybersecurity breach",
                 ]},
            ),
            (
                "Design a plan to reduce {metric} by {target}% in a {context} "
                "over {timeframe}. Include data collection, analysis, intervention "
                "design, pilot testing, full-scale implementation, and evaluation.",
                {"metric": ["energy consumption", "waste generation",
                            "patient wait times", "defect rate"],
                 "target": [20, 30, 50],
                 "context": ["manufacturing plant", "hospital", "school district"],
                 "timeframe": ["6 months", "1 year", "2 years"]},
            ),
            (
                "A team of {team_size} researchers must {objective} within "
                "{deadline}. Resources include {resources}. Develop a plan "
                "addressing task decomposition, parallel workstreams, "
                "integration points, risk management, and quality assurance.",
                {"team_size": [3, 6, 12],
                 "objective": [
                     "develop a prototype AI diagnostic tool",
                     "complete a clinical trial analysis",
                     "publish a comprehensive literature review",
                 ],
                 "deadline": ["3 months", "6 months", "1 year"],
                 "resources": [
                     "limited compute and a shared lab",
                     "a $100K grant and 2 interns",
                     "access to a large dataset and cloud computing",
                 ]},
            ),
            (
                "Develop a disaster recovery plan for a {disaster} affecting a "
                "{entity}. Cover immediate response, short-term recovery, "
                "long-term rebuilding, communication strategy, and lessons learned.",
                {"disaster": ["flooding", "data centre fire", "earthquake",
                              "pandemic outbreak"],
                 "entity": ["small coastal town", "regional hospital",
                            "multinational corporation", "university campus"]},
            ),
            (
                "Solve the following resource allocation puzzle: {puzzle}. "
                "Show your reasoning step by step, identify constraints, "
                "explore possible solutions, and select the optimal one.",
                {"puzzle": [
                    "Distribute 100 units of supply across 5 centres with "
                    "varying demand (10, 15, 25, 20, 30) using 3 transport "
                    "routes with different costs and capacities",
                    "Assign 8 volunteers to 4 projects requiring 2-3 people "
                    "each, given skill requirements and volunteer preferences",
                    "Schedule 6 tasks on 2 machines to minimise total "
                    "completion time, given task durations and dependencies",
                ]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "problem_solving", complexity,
        )

    def generate_strategy_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate game theory, decision making, and strategic planning prompts."""
        templates = [
            (
                "Develop a {horizon} strategic plan for a {company_type} facing "
                "{challenge}. Include SWOT analysis, strategic options, evaluation "
                "criteria, implementation roadmap, and key performance indicators.",
                {"horizon": ["3-year", "5-year", "10-year"],
                 "company_type": [
                     "mid-size SaaS company", "regional bank",
                     "family-owned restaurant chain",
                 ],
                 "challenge": [
                     "digital disruption", "market saturation",
                     "new regulatory requirements",
                 ]},
            ),
            (
                "Analyse the strategic options for a {player} in a {scenario}. "
                "Use game-theoretic thinking to identify dominant strategies, "
                "Nash equilibria, and potential cooperative solutions. Recommend "
                "a course of action with justification.",
                {"player": ["small retailer", "new market entrant", "incumbent firm"],
                 "scenario": [
                     "price war with a larger competitor",
                     "patent dispute with multiple parties",
                     "standards-setting committee negotiation",
                 ]},
            ),
            (
                "Design a decision framework for {decision}. Include criteria "
                "identification, option generation, risk assessment, sensitivity "
                "analysis, and a recommendation with contingency plans.",
                {"decision": [
                    "choosing between building vs buying a key technology component",
                    "deciding whether to expand internationally or deepen domestic market",
                    "selecting between three acquisition targets",
                    "determining the optimal pricing strategy for a new product",
                ]},
            ),
            (
                "Create a stakeholder management strategy for {project}. Identify "
                "key stakeholders, assess their interests and influence, develop "
                "engagement tactics, anticipate objections, and plan for coalition building.",
                {"project": [
                    "building a new data centre in a residential area",
                    "implementing a company-wide restructuring",
                    "launching a controversial public health campaign",
                ]},
            ),
            (
                "Develop a {type} strategy for a {context}. Consider multiple "
                "time horizons, competitive dynamics, resource constraints, "
                "and uncertainty. Present at least 3 distinct strategic options "
                "with trade-off analysis.",
                {"type": ["growth", "turnaround", "innovation", "sustainability"],
                 "context": [
                     "declining newspaper publisher",
                     "start-up competing with tech giants",
                     "municipal government facing climate change",
                 ]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "strategy", complexity,
        )

    def generate_constraint_satisfaction_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate constraint satisfaction planning prompts."""
        templates = [
            (
                "Schedule {n_events} events in {n_venues} venues over {n_days} days. "
                "Each event has specific duration, equipment needs, and audience "
                "capacity requirements. Some events conflict and cannot be simultaneous. "
                "Find a feasible schedule.",
                {"n_events": [10, 15, 25], "n_venues": [3, 5, 8],
                 "n_days": [3, 5, 7]},
            ),
            (
                "Assign {n_students} students to {n_projects} group projects with "
                "{group_size} students each. Each student has ranked preferences, "
                "skill requirements per project must be met, and no group should "
                "have more than {max_same} students from the same background.",
                {"n_students": [20, 30, 50], "n_projects": [5, 8, 10],
                 "group_size": [3, 4, 5], "max_same": [2, 3]},
            ),
            (
                "Design a {diet_type} diet plan for {days} days satisfying "
                "nutritional constraints: minimum {min_cal} and maximum {max_cal} "
                "calories, at least {protein}g protein, under {fat}g fat, and "
                "at least {fibre}g fibre per day. Use foods from a limited "
                "ingredient list.",
                {"diet_type": ["vegetarian", "Mediterranean", "low-carb"],
                 "days": [5, 7], "min_cal": [1500, 1800], "max_cal": [2000, 2500],
                 "protein": [50, 70], "fat": [60, 80], "fibre": [25, 30]},
            ),
            (
                "Organise a {sport} tournament bracket for {n_teams} teams across "
                "{n_fields} fields over {n_days} days. Each team plays at most "
                "{max_games} games per day with at least {rest} hours between games. "
                "Higher-seeded teams cannot meet before round {min_round}.",
                {"sport": ["soccer", "basketball", "volleyball"],
                 "n_teams": [8, 16, 32], "n_fields": [2, 4, 6],
                 "n_days": [2, 3, 5], "max_games": [2, 3], "rest": [2, 3],
                 "min_round": [2, 3]},
            ),
            (
                "Layout {n_offices} offices, {n_meeting} meeting rooms, and "
                "{n_common} common areas on a {area} sq ft floor plan. "
                "Constraints include noise separation between departments, "
                "natural light access, ADA accessibility, and fire code compliance.",
                {"n_offices": [20, 40, 60], "n_meeting": [3, 5, 8],
                 "n_common": [2, 3, 5], "area": [5000, 10000, 20000]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "constraint_satisfaction", complexity,
        )

    def generate_causal_reasoning_prompts(
        self, count: int = 5, complexity: str = "moderate"
    ) -> List[TaskPrompt]:
        """Generate cause-effect chain analysis prompts."""
        templates = [
            (
                "Trace the causal chain of effects when {event} in a {context}. "
                "Identify direct effects, secondary effects, feedback loops, "
                "and long-term consequences. Distinguish between probable and "
                "speculative causal links.",
                {"event": [
                    "a major employer closes its factory",
                    "a new highway bypass is built",
                    "free public Wi-Fi is deployed city-wide",
                    "a universal basic income program is introduced",
                 ],
                 "context": [
                     "small rural town", "densely populated urban neighbourhood",
                     "mid-size suburban city", "economically depressed region",
                 ]},
            ),
            (
                "Analyse why {outcome} occurred by constructing a causal diagram. "
                "Work backwards from the outcome to identify contributing factors, "
                "root causes, and the chain of events. Identify which causes were "
                "necessary and which were sufficient.",
                {"outcome": [
                    "a successful product launch exceeded sales targets by 200%",
                    "a bridge collapsed during routine traffic",
                    "a school achieved the highest test scores in its district",
                    "a start-up failed despite significant funding",
                ]},
            ),
            (
                "Predict the likely causal consequences of {intervention} in "
                "{system}. Map out primary, secondary, and tertiary effects. "
                "Identify potential unintended consequences and tipping points.",
                {"intervention": [
                    "banning single-use plastics",
                    "mandating 4-day work weeks",
                    "introducing congestion pricing",
                    "requiring AI transparency in hiring",
                ],
                 "system": [
                     "a national economy", "a metropolitan area",
                     "the technology industry", "the education sector",
                 ]},
            ),
            (
                "Compare two causal explanations for {phenomenon}: (A) {explanation_a} "
                "and (B) {explanation_b}. Evaluate the evidence for each, identify "
                "confounding variables, and determine which explanation is more "
                "strongly supported.",
                {"phenomenon": [
                    "rising healthcare costs",
                    "declining birth rates in developed nations",
                    "increasing social media usage among youth",
                ],
                 "explanation_a": [
                     "administrative overhead", "economic factors",
                     "algorithmic engagement design",
                 ],
                 "explanation_b": [
                     "technological advancement costs", "cultural shifts",
                     "peer social pressure",
                 ]},
            ),
            (
                "Design an experiment or observational study to test the causal "
                "claim that {claim}. Specify the independent and dependent variables, "
                "control conditions, potential confounders, and the causal mechanism "
                "you would expect to observe.",
                {"claim": [
                    "green spaces reduce urban crime rates",
                    "remote work increases productivity",
                    "music education improves mathematical ability",
                    "neighbourhood diversity reduces prejudice",
                ]},
            ),
        ]
        return self._generate_from_templates(
            templates, count, "causal_reasoning", complexity,
        )

    # -----------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------

    def format_planning_prompt(
        self,
        scenario: Dict[str, Any],
        include_constraints: bool = True,
        include_goal: bool = True,
    ) -> str:
        """Format a planning scenario dict into a prompt string."""
        parts: List[str] = []

        title = scenario.get("title", "Planning Task")
        parts.append(f"## {title}\n")
        parts.append(scenario.get("prompt", ""))

        if include_goal and scenario.get("goal"):
            parts.append(f"\n**Goal:** {scenario['goal']}")

        if include_constraints and scenario.get("constraints"):
            parts.append("\n**Constraints:**")
            for i, c in enumerate(scenario["constraints"], 1):
                parts.append(f"  {i}. {c}")

        parts.append(
            "\nProvide a detailed, step-by-step plan. For each step, "
            "describe the action, any prerequisites, expected outcomes, "
            "and estimated time."
        )
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _generate_from_templates(
        self,
        templates: List[Tuple[str, Dict[str, list]]],
        count: int,
        domain: str,
        complexity: str,
    ) -> List[TaskPrompt]:
        """Instantiate prompts by randomly filling template parameters."""
        prompts: List[TaskPrompt] = []
        for _ in range(count):
            tpl, params = templates[self._rng.randint(len(templates))]

            filled_params: Dict[str, Any] = {}
            for key, options in params.items():
                filled_params[key] = options[self._rng.randint(len(options))]

            try:
                text = tpl.format(**filled_params)
            except (KeyError, IndexError):
                text = tpl

            prompt_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            prompt = TaskPrompt(
                prompt_id=prompt_id,
                text=text,
                domain=TaskDomain.REASONING,
                metadata={
                    "planning_domain": domain,
                    "complexity": complexity,
                    "template_params": filled_params,
                },
                max_gen_length=1024,
            )
            prompts.append(prompt)

        return prompts

    def generate_all(self, per_domain: int = 5) -> List[TaskPrompt]:
        """Generate prompts across all planning domains."""
        all_prompts: List[TaskPrompt] = []
        all_prompts.extend(self.generate_logistics_prompts(per_domain))
        all_prompts.extend(self.generate_project_planning_prompts(per_domain))
        all_prompts.extend(self.generate_cooking_recipe_prompts(per_domain))
        all_prompts.extend(self.generate_travel_planning_prompts(per_domain))
        all_prompts.extend(self.generate_problem_solving_prompts(per_domain))
        all_prompts.extend(self.generate_strategy_prompts(per_domain))
        all_prompts.extend(self.generate_constraint_satisfaction_prompts(per_domain))
        all_prompts.extend(self.generate_causal_reasoning_prompts(per_domain))
        return all_prompts

    def get_builtin_prompts(self) -> List[TaskPrompt]:
        """Convert built-in scenario bank into TaskPrompt objects."""
        prompts: List[TaskPrompt] = []
        for scenario in self._scenarios:
            text = self.format_planning_prompt(scenario)
            prompt = TaskPrompt(
                prompt_id=scenario["id"],
                text=text,
                domain=TaskDomain.REASONING,
                metadata={
                    "planning_domain": scenario.get("domain", ""),
                    "complexity": scenario.get("complexity", "moderate"),
                    "title": scenario.get("title", ""),
                },
                max_gen_length=1024,
            )
            prompts.append(prompt)
        return prompts


# ---------------------------------------------------------------------------
# PlanningEvaluator
# ---------------------------------------------------------------------------


class PlanningEvaluator(TaskEvaluator):
    """Evaluator specialised for planning and reasoning tasks.

    Extends the base :class:`TaskEvaluator` with plan-specific metrics.
    """

    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(metrics_config or {})
        self._action_verbs_cache: Dict[str, List[str]] = {}

    # -----------------------------------------------------------------
    # Plan quality metrics
    # -----------------------------------------------------------------

    def evaluate_plan_completeness(
        self, generation: str, prompt: TaskPrompt
    ) -> float:
        """Evaluate whether the plan addresses all goals in the prompt.

        Checks for coverage of key topics, constraints, and objectives
        mentioned in the prompt.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        prompt_lower = prompt.text.lower()
        gen_lower = generation.lower()

        # Extract key nouns / concepts from the prompt (simple heuristic)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "must", "can", "could",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "and", "but", "or", "not", "so", "yet", "if", "when",
            "that", "this", "which", "who", "whom", "what", "how",
            "where", "each", "every", "all", "any", "some", "your",
            "you", "i", "we", "they", "it", "its", "their", "our",
            "my", "he", "she", "him", "her", "me", "us", "them",
            "than", "very", "just", "also", "more", "most", "such",
        }
        prompt_words = set(prompt_lower.split()) - stop_words
        # Keep only words >= 4 chars as likely content words
        key_concepts = {w for w in prompt_words if len(w) >= 4}

        if not key_concepts:
            return 0.5

        covered = sum(1 for concept in key_concepts if concept in gen_lower)
        concept_coverage = covered / len(key_concepts)

        # Check if the plan has identifiable steps
        steps = _extract_numbered_steps(generation)
        has_steps = 1.0 if len(steps) >= 3 else 0.5 * len(steps) / 3.0

        # Check if constraints from metadata are addressed
        plan_constraints = prompt.metadata.get("constraints", [])
        if isinstance(plan_constraints, list) and plan_constraints:
            constraint_words = set()
            for c in plan_constraints:
                constraint_words.update(
                    w for w in c.lower().split() if len(w) >= 4 and w not in stop_words
                )
            if constraint_words:
                c_covered = sum(1 for w in constraint_words if w in gen_lower)
                constraint_score = c_covered / len(constraint_words)
            else:
                constraint_score = 0.5
        else:
            constraint_score = 0.5

        completeness = 0.4 * concept_coverage + 0.3 * has_steps + 0.3 * constraint_score
        return float(np.clip(completeness, 0.0, 1.0))

    def evaluate_step_ordering(self, generation: str) -> float:
        """Evaluate whether steps are in a logical temporal sequence.

        Looks for temporal markers in order, numbered steps, and
        cause-before-effect patterns.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        score_components: List[float] = []

        # Check for numbered steps in order
        numbers = re.findall(r"(?m)^\s*(\d+)[.)]\s+", generation)
        if numbers:
            int_numbers = [int(n) for n in numbers]
            if len(int_numbers) >= 2:
                # Check if they are monotonically increasing
                monotonic = all(
                    int_numbers[i] <= int_numbers[i + 1]
                    for i in range(len(int_numbers) - 1)
                )
                score_components.append(1.0 if monotonic else 0.5)
            else:
                score_components.append(0.7)
        else:
            score_components.append(0.3)

        # Check for temporal markers
        temporal_markers = extract_temporal_ordering(generation)
        if len(temporal_markers) >= 2:
            # Check relative order of "first" vs "then" vs "finally"
            marker_texts = [m[0] for m in temporal_markers]
            ordering_words = ["first", "firstly", "second", "then", "next",
                              "after", "finally", "lastly"]
            found_order = [w for w in ordering_words if any(w in m for m in marker_texts)]
            if len(found_order) >= 2:
                score_components.append(0.9)
            else:
                score_components.append(0.6)
        elif len(temporal_markers) == 1:
            score_components.append(0.5)
        else:
            score_components.append(0.3)

        # Check for dependency language ("after X, do Y", "once X, then Y")
        dep_patterns = [
            r"(?i)\bafter\b.{5,50}\b(then|next|proceed)\b",
            r"(?i)\bonce\b.{5,50}\b(then|can|should)\b",
            r"(?i)\bbefore\b.{5,50}\bmust\b",
            r"(?i)\brequires?\b.{5,50}\bfirst\b",
        ]
        dep_count = sum(
            1 for p in dep_patterns if re.search(p, generation)
        )
        if dep_count >= 2:
            score_components.append(1.0)
        elif dep_count == 1:
            score_components.append(0.7)
        else:
            score_components.append(0.4)

        return float(np.mean(score_components)) if score_components else 0.5

    def evaluate_feasibility(self, generation: str) -> float:
        """Evaluate whether plan steps are feasible (no impossible actions).

        Heuristically checks for vague, impossible, or contradictory steps.
        Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        gen_lower = generation.lower()
        score = 1.0

        # Penalty for vague hand-waving
        vague_phrases = [
            "somehow", "magically", "instantly", "effortlessly",
            "just figure it out", "it will work itself out",
            "trivially", "obviously just", "simply do everything",
        ]
        for phrase in vague_phrases:
            if phrase in gen_lower:
                score -= 0.1

        # Penalty for contradictions
        contradiction_pairs = [
            ("increase", "decrease"),
            ("expand", "contract"),
            ("add more", "reduce"),
        ]
        steps = _extract_numbered_steps(generation)
        for step in steps:
            step_lower = step.lower()
            for word_a, word_b in contradiction_pairs:
                if word_a in step_lower and word_b in step_lower:
                    score -= 0.05

        # Reward for specificity
        has_numbers = bool(re.search(r"\d+", generation))
        has_time_refs = bool(re.search(
            r"(?i)\b(\d+\s*(hours?|days?|weeks?|months?|minutes?))\b", generation
        ))
        has_quantities = bool(re.search(r"\$[\d,]+|\d+\s*%|\d+\s*units?", generation))

        specificity_bonus = 0.0
        if has_numbers:
            specificity_bonus += 0.05
        if has_time_refs:
            specificity_bonus += 0.05
        if has_quantities:
            specificity_bonus += 0.05
        score += specificity_bonus

        # Check step count is reasonable
        if len(steps) >= 3:
            score += 0.05
        if len(steps) >= 5:
            score += 0.05

        return float(np.clip(score, 0.0, 1.0))

    def evaluate_efficiency(self, generation: str) -> float:
        """Evaluate whether the plan avoids unnecessary or redundant steps.

        Looks for repetition, filler steps, and excessively granular or
        excessively vague steps.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        steps = _extract_numbered_steps(generation)
        if not steps:
            return 0.3

        score = 1.0

        # Check for near-duplicate steps (Jaccard > 0.7)
        step_word_sets = [set(s.lower().split()) for s in steps]
        duplicate_count = 0
        for i, j in combinations(range(len(step_word_sets)), 2):
            si, sj = step_word_sets[i], step_word_sets[j]
            union = si | sj
            if union:
                jaccard = len(si & sj) / len(union)
                if jaccard > 0.7:
                    duplicate_count += 1

        if duplicate_count > 0:
            score -= 0.1 * min(duplicate_count, 5)

        # Check for filler/trivial steps
        filler_patterns = [
            r"^(and then|also|additionally)\s*$",
            r"^(do the thing|proceed|continue)\s*$",
            r"^(etc|and so on|and more)\s*$",
        ]
        filler_count = 0
        for step in steps:
            for pat in filler_patterns:
                if re.match(pat, step.strip(), re.IGNORECASE):
                    filler_count += 1
                    break

        if filler_count > 0:
            score -= 0.05 * filler_count

        # Check step length variance (very uniform = possibly boilerplate)
        if len(steps) >= 3:
            lengths = [len(s.split()) for s in steps]
            mean_len = np.mean(lengths)
            if mean_len > 0:
                cv = np.std(lengths) / mean_len
                # A reasonable CV is 0.3-0.8; too low means boilerplate
                if cv < 0.1:
                    score -= 0.1

        # Reward reasonable step count
        if 3 <= len(steps) <= 20:
            score += 0.05
        elif len(steps) > 30:
            score -= 0.1

        return float(np.clip(score, 0.0, 1.0))

    def evaluate_constraint_satisfaction(
        self, generation: str, prompt: TaskPrompt
    ) -> float:
        """Evaluate whether the plan addresses constraints from the prompt.

        Checks for explicit mentions or semantic coverage of stated constraints.
        Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        # Try to extract constraints from prompt text and metadata
        constraints_text: List[str] = []

        # From metadata
        meta_constraints = prompt.metadata.get("constraints", [])
        if isinstance(meta_constraints, list):
            constraints_text.extend(meta_constraints)

        # From prompt text: look for "Constraints:" section
        constraint_match = re.search(
            r"(?i)\*?\*?constraints?\*?\*?:?\s*\n((?:\s*\d+\..+\n?)+)",
            prompt.text,
        )
        if constraint_match:
            items = re.findall(r"\d+\.\s*(.+)", constraint_match.group(1))
            constraints_text.extend(items)

        if not constraints_text:
            return 0.5  # No explicit constraints found; neutral score

        gen_lower = generation.lower()
        satisfied = 0
        for constraint in constraints_text:
            # Check if key words from the constraint appear in the plan
            c_words = set(constraint.lower().split())
            stop = {"the", "a", "an", "is", "must", "be", "to", "of",
                     "at", "and", "or", "not", "no", "in", "for", "with",
                     "least", "more", "than", "per", "each", "all", "any"}
            key_words = {w for w in c_words if len(w) >= 3 and w not in stop}
            if not key_words:
                satisfied += 1
                continue
            matches = sum(1 for kw in key_words if kw in gen_lower)
            if matches / len(key_words) >= 0.4:
                satisfied += 1

        return satisfied / len(constraints_text)

    def evaluate_causal_coherence(self, generation: str) -> float:
        """Evaluate whether effects logically follow from actions in the plan.

        Checks for causal language, logical connectives, and
        cause-then-effect patterns.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        gen_lower = generation.lower()
        scores: List[float] = []

        # Check for causal connectives
        causal_connectives = [
            "because", "therefore", "consequently", "as a result",
            "this leads to", "which causes", "due to", "so that",
            "in order to", "resulting in", "this enables", "this allows",
            "this ensures", "which means", "hence", "thus",
            "accordingly", "for this reason",
        ]
        connective_count = sum(1 for c in causal_connectives if c in gen_lower)
        if connective_count >= 4:
            scores.append(1.0)
        elif connective_count >= 2:
            scores.append(0.7)
        elif connective_count >= 1:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # Check for if-then patterns
        if_then = len(re.findall(r"(?i)\bif\b.{5,60}\bthen\b", generation))
        if if_then >= 2:
            scores.append(0.9)
        elif if_then >= 1:
            scores.append(0.6)
        else:
            scores.append(0.3)

        # Check for explicit outcome/effect statements
        effect_patterns = [
            r"(?i)this\s+(will|would|should)\s+\w+",
            r"(?i)the\s+result\s+(is|will be|would be)",
            r"(?i)expected\s+outcome",
            r"(?i)this\s+step\s+(ensures?|produces?|creates?|enables?)",
        ]
        effect_count = sum(
            1 for p in effect_patterns if re.search(p, generation)
        )
        if effect_count >= 2:
            scores.append(0.9)
        elif effect_count >= 1:
            scores.append(0.6)
        else:
            scores.append(0.3)

        return float(np.mean(scores)) if scores else 0.5

    def evaluate_detail_level(self, generation: str) -> float:
        """Evaluate whether the plan has an appropriate level of specificity.

        Checks for a balance between high-level overview and concrete detail.
        Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        words = generation.split()
        total_words = len(words)

        steps = _extract_numbered_steps(generation)
        scores: List[float] = []

        # Check word count adequacy
        if total_words >= 200:
            scores.append(1.0)
        elif total_words >= 100:
            scores.append(0.7)
        elif total_words >= 50:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # Check average step detail
        if steps:
            avg_step_words = np.mean([len(s.split()) for s in steps])
            if 15 <= avg_step_words <= 60:
                scores.append(1.0)
            elif 10 <= avg_step_words < 15 or 60 < avg_step_words <= 100:
                scores.append(0.7)
            elif avg_step_words < 10:
                scores.append(0.4)
            else:
                scores.append(0.5)

        # Check for specific details: numbers, names, dates
        detail_indicators = [
            (r"\d+", "numbers"),
            (r"\$[\d,.]+", "monetary"),
            (r"\d+:\d+", "time"),
            (r"(?i)(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "days"),
            (r"(?i)(january|february|march|april|may|june|july|august|september|october|november|december)", "months"),
        ]
        detail_count = 0
        for pattern, _ in detail_indicators:
            if re.search(pattern, generation):
                detail_count += 1
        scores.append(min(1.0, detail_count / 3.0))

        # Check for action verbs (specificity of actions)
        verbs = extract_action_verbs(generation)
        if len(verbs) >= 5:
            scores.append(1.0)
        elif len(verbs) >= 3:
            scores.append(0.7)
        else:
            scores.append(0.4)

        return float(np.mean(scores)) if scores else 0.5

    def evaluate_robustness(self, generation: str) -> float:
        """Evaluate whether the plan handles edge cases and contingencies.

        Looks for contingency planning, risk assessment, and alternative
        approaches.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        gen_lower = generation.lower()
        scores: List[float] = []

        # Check for contingency language
        contingency_phrases = [
            "backup plan", "contingency", "plan b", "alternative",
            "fallback", "in case", "if this fails", "risk",
            "mitigation", "worst case", "best case", "edge case",
            "exception", "unexpected", "unforeseen", "challenge",
        ]
        c_count = sum(1 for p in contingency_phrases if p in gen_lower)
        if c_count >= 3:
            scores.append(1.0)
        elif c_count >= 2:
            scores.append(0.7)
        elif c_count >= 1:
            scores.append(0.5)
        else:
            scores.append(0.1)

        # Check for conditional logic
        conditional_patterns = [
            r"(?i)\bif\b", r"(?i)\bunless\b", r"(?i)\bwhen\b.{5,40}\b(might|could|may)\b",
            r"(?i)\bshould\b.{5,40}\b(fail|not work|break)\b",
        ]
        cond_count = sum(1 for p in conditional_patterns if re.search(p, generation))
        if cond_count >= 2:
            scores.append(0.9)
        elif cond_count >= 1:
            scores.append(0.6)
        else:
            scores.append(0.2)

        # Check for monitoring / verification steps
        monitor_phrases = [
            "monitor", "verify", "check", "validate", "confirm",
            "review", "assess", "evaluate", "measure", "track",
            "test", "audit", "inspect",
        ]
        m_count = sum(1 for p in monitor_phrases if p in gen_lower)
        if m_count >= 3:
            scores.append(1.0)
        elif m_count >= 1:
            scores.append(0.6)
        else:
            scores.append(0.2)

        return float(np.mean(scores)) if scores else 0.3

    def evaluate_creativity(self, generation: str, prompt: TaskPrompt) -> float:
        """Evaluate whether the plan uses novel or creative approaches.

        Checks for unconventional vocabulary, diverse action verbs, and
        non-obvious strategies.  Returns a score in [0, 1].
        """
        if not generation.strip():
            return 0.0

        scores: List[float] = []

        # Lexical diversity (TTR)
        words = generation.lower().split()
        if words:
            ttr = len(set(words)) / len(words)
            scores.append(min(1.0, ttr * 1.5))
        else:
            scores.append(0.0)

        # Diversity of action verbs
        verbs = extract_action_verbs(generation)
        if len(verbs) >= 8:
            scores.append(1.0)
        elif len(verbs) >= 5:
            scores.append(0.7)
        elif len(verbs) >= 3:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # Check for creative/innovative language
        creative_markers = [
            "innovative", "creative", "novel", "unique", "unconventional",
            "outside the box", "reimagine", "transform", "pioneer",
            "breakthrough", "synergy", "leverage", "hybrid approach",
            "cross-functional", "interdisciplinary", "holistic",
        ]
        gen_lower = generation.lower()
        cr_count = sum(1 for m in creative_markers if m in gen_lower)
        scores.append(min(1.0, cr_count / 3.0))

        # Structural creativity: does the plan use non-linear structure?
        structure = detect_plan_structure(generation)
        if structure in (PlanStructureType.PARALLEL, PlanStructureType.HIERARCHICAL,
                         PlanStructureType.MIXED):
            scores.append(0.9)
        elif structure == PlanStructureType.BRANCHING:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Low overlap with prompt (rephrasing rather than copying)
        prompt_words = set(prompt.text.lower().split())
        gen_words = set(words)
        if prompt_words and gen_words:
            overlap = len(prompt_words & gen_words) / len(gen_words)
            originality = 1.0 - overlap
            scores.append(min(1.0, originality * 1.3))

        return float(np.mean(scores)) if scores else 0.5

    # -----------------------------------------------------------------
    # Aggregate scores
    # -----------------------------------------------------------------

    def compute_plan_quality_score(
        self, generation: str, prompt: TaskPrompt
    ) -> Dict[str, float]:
        """Compute all plan quality metrics and an aggregate score.

        Returns a dictionary with individual metric scores and an overall
        quality score.
        """
        scores: Dict[str, float] = {
            "completeness": self.evaluate_plan_completeness(generation, prompt),
            "step_ordering": self.evaluate_step_ordering(generation),
            "feasibility": self.evaluate_feasibility(generation),
            "efficiency": self.evaluate_efficiency(generation),
            "constraint_satisfaction": self.evaluate_constraint_satisfaction(
                generation, prompt
            ),
            "causal_coherence": self.evaluate_causal_coherence(generation),
            "detail_level": self.evaluate_detail_level(generation),
            "robustness": self.evaluate_robustness(generation),
            "creativity": self.evaluate_creativity(generation, prompt),
        }

        # Weighted aggregate
        weights = {
            "completeness": 0.20,
            "step_ordering": 0.12,
            "feasibility": 0.15,
            "efficiency": 0.10,
            "constraint_satisfaction": 0.12,
            "causal_coherence": 0.10,
            "detail_level": 0.08,
            "robustness": 0.08,
            "creativity": 0.05,
        }
        total_weight = sum(weights.values())
        quality = sum(
            scores[k] * weights.get(k, 0.1) for k in scores
        ) / total_weight
        scores["overall_quality"] = float(np.clip(quality, 0.0, 1.0))

        return scores

    def compute_plan_diversity_score(
        self, generations: List[str], prompts: List[TaskPrompt]
    ) -> Dict[str, float]:
        """Compute diversity metrics across a set of plans.

        Returns a dictionary with individual diversity metrics and an
        aggregate diversity score.
        """
        if len(generations) < 2:
            return {"overall_diversity": 0.0}

        div_metrics = PlanDiversityMetrics()
        scores: Dict[str, float] = {
            "strategy_diversity": div_metrics.compute_strategy_diversity(generations),
            "step_sequence_diversity": div_metrics.compute_step_sequence_diversity(
                generations
            ),
            "resource_diversity": div_metrics.compute_resource_utilization_diversity(
                generations
            ),
            "abstraction_diversity": div_metrics.compute_abstraction_level_diversity(
                generations
            ),
            "structure_diversity": div_metrics.compute_plan_structure_diversity(
                generations
            ),
            "solution_coverage": div_metrics.compute_solution_space_coverage(
                generations
            ),
        }

        # Aggregate
        values = [v for v in scores.values() if isinstance(v, float)]
        scores["overall_diversity"] = float(np.mean(values)) if values else 0.0

        return scores


# ---------------------------------------------------------------------------
# PlanDiversityMetrics
# ---------------------------------------------------------------------------


class PlanDiversityMetrics:
    """Compute diversity metrics across a set of generated plans."""

    def compute_strategy_diversity(self, plans: List[str]) -> float:
        """Measure how different the overall strategies are across plans.

        Uses action-verb profile dissimilarity as a proxy for strategic
        approach diversity.  Returns a score in [0, 1].
        """
        if len(plans) < 2:
            return 0.0

        verb_profiles: List[Counter] = []
        for plan in plans:
            verbs = extract_action_verbs(plan)
            verb_profiles.append(Counter(verbs))

        # Pairwise Jaccard distance on verb sets
        distances: List[float] = []
        for i, j in combinations(range(len(verb_profiles)), 2):
            keys_i = set(verb_profiles[i].keys())
            keys_j = set(verb_profiles[j].keys())
            union = keys_i | keys_j
            if not union:
                distances.append(0.0)
                continue
            jaccard_dist = 1.0 - len(keys_i & keys_j) / len(union)
            distances.append(jaccard_dist)

        return float(np.mean(distances)) if distances else 0.0

    def compute_step_sequence_diversity(self, plans: List[str]) -> float:
        """Measure diversity of step orderings across plans.

        Compares the sequences of action verbs between plans using
        edit-distance-based similarity.  Returns a score in [0, 1].
        """
        if len(plans) < 2:
            return 0.0

        verb_sequences: List[List[str]] = []
        for plan in plans:
            steps = _extract_numbered_steps(plan)
            seq: List[str] = []
            for step in steps:
                verbs = extract_action_verbs(step)
                if verbs:
                    seq.append(verbs[0])
                else:
                    words = step.lower().split()
                    if words:
                        seq.append(words[0])
            verb_sequences.append(seq)

        distances: List[float] = []
        for i, j in combinations(range(len(verb_sequences)), 2):
            dist = self._normalised_edit_distance(
                verb_sequences[i], verb_sequences[j]
            )
            distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def compute_resource_utilization_diversity(self, plans: List[str]) -> float:
        """Measure diversity of resources/tools/entities mentioned across plans.

        Extracts noun-like tokens (capitalised words, specific terms) as
        resource proxies and computes set diversity.  Returns [0, 1].
        """
        if len(plans) < 2:
            return 0.0

        resource_sets: List[Set[str]] = []
        resource_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        )
        # Also capture domain-specific resource indicators
        tool_pattern = re.compile(
            r"(?i)\b(tool|software|platform|system|service|equipment|"
            r"material|resource|budget|team|staff|machine|vehicle|"
            r"facility|database|framework|library)\b"
        )

        for plan in plans:
            resources: Set[str] = set()
            for m in resource_pattern.finditer(plan):
                token = m.group(0).lower()
                # Skip very common words
                if len(token) > 2 and token not in {"the", "this", "that", "and"}:
                    resources.add(token)
            for m in tool_pattern.finditer(plan):
                # Get the surrounding context (word before + keyword)
                start = max(0, m.start() - 30)
                context = plan[start:m.end()].strip().lower()
                context_words = context.split()[-3:]
                resources.add(" ".join(context_words))
            resource_sets.append(resources)

        distances: List[float] = []
        for i, j in combinations(range(len(resource_sets)), 2):
            union = resource_sets[i] | resource_sets[j]
            if not union:
                distances.append(0.0)
                continue
            dist = 1.0 - len(resource_sets[i] & resource_sets[j]) / len(union)
            distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def compute_abstraction_level_diversity(self, plans: List[str]) -> float:
        """Measure diversity in abstraction level (high-level vs detailed).

        Uses a composite of step count, average words per step, and
        presence of sub-steps as indicators.  Returns [0, 1].
        """
        if len(plans) < 2:
            return 0.0

        abstraction_scores: List[float] = []
        for plan in plans:
            steps = _extract_numbered_steps(plan)
            n_steps = len(steps) if steps else 1
            avg_words = float(np.mean([len(s.split()) for s in steps])) if steps else 0.0

            # High abstraction = few steps, many words each
            # Low abstraction = many steps, fewer words each
            # Encode as a single number in [0, 1] where 0=high-level, 1=detailed
            step_score = min(1.0, n_steps / 20.0)
            word_score = 1.0 - min(1.0, avg_words / 50.0)

            # Sub-step indicators (a), (b), i., ii., etc.
            sub_step_count = len(re.findall(
                r"(?m)^\s*(?:[a-z]\)|\([a-z]\)|[ivx]+\.)\s+", plan
            ))
            sub_step_bonus = min(0.3, sub_step_count * 0.05)

            level = 0.5 * step_score + 0.3 * word_score + 0.2 * sub_step_bonus
            abstraction_scores.append(level)

        # Diversity = standard deviation normalised to [0, 1]
        scores_arr = np.array(abstraction_scores)
        if len(scores_arr) < 2:
            return 0.0
        std_val = float(np.std(scores_arr))
        # Max theoretical std for values in [0,1] is 0.5
        diversity = min(1.0, std_val / 0.5) if std_val > 0 else 0.0

        # Also add range component
        range_val = float(np.max(scores_arr) - np.min(scores_arr))
        diversity = 0.5 * diversity + 0.5 * range_val

        return float(np.clip(diversity, 0.0, 1.0))

    def compute_plan_structure_diversity(self, plans: List[str]) -> float:
        """Measure diversity of plan topologies (linear, branching, parallel, etc.).

        Returns [0, 1] where 1.0 means every plan has a different structure.
        """
        if len(plans) < 2:
            return 0.0

        structures: List[PlanStructureType] = []
        for plan in plans:
            structures.append(detect_plan_structure(plan))

        # Count distinct structures
        unique_structures = len(set(structures))
        total = len(structures)
        # Max possible distinct = min(total, number_of_structure_types)
        max_distinct = min(total, len(PlanStructureType))

        # Normalise
        if max_distinct <= 1:
            return 0.0
        return (unique_structures - 1) / (max_distinct - 1)

    def compute_solution_space_coverage(self, plans: List[str]) -> float:
        """Estimate how well the set of plans covers the space of possible
        solutions.

        Uses a bag-of-words embedding and measures the volume of the
        convex hull in the projected space (approximated by variance).
        Returns [0, 1].
        """
        if len(plans) < 2:
            return 0.0

        # Build vocabulary and embed each plan
        vocab = _build_vocabulary(plans, max_vocab=2000)
        if not vocab:
            return 0.0

        vectors = np.array([_text_to_bow_vector(p, vocab) for p in plans])

        # Reduce dimensionality via random projection if vocab is large
        n_features = vectors.shape[1]
        if n_features > 50:
            rng = np.random.RandomState(42)
            proj = rng.randn(n_features, 50) / np.sqrt(50)
            vectors = vectors @ proj

        # Compute pairwise cosine distances
        n = len(vectors)
        distances: List[float] = []
        for i, j in combinations(range(n), 2):
            vi, vj = vectors[i], vectors[j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i > 0 and norm_j > 0:
                cos_sim = float(np.dot(vi, vj) / (norm_i * norm_j))
                distances.append(1.0 - cos_sim)
            else:
                distances.append(0.0)

        if not distances:
            return 0.0

        mean_dist = float(np.mean(distances))

        # Also compute variance across dimensions as spread indicator
        variances = np.var(vectors, axis=0)
        total_var = float(np.sum(variances))
        # Normalise: scale by number of dimensions
        norm_var = total_var / vectors.shape[1] if vectors.shape[1] > 0 else 0.0

        coverage = 0.6 * min(1.0, mean_dist * 2.0) + 0.4 * min(1.0, norm_var * 10.0)
        return float(np.clip(coverage, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _normalised_edit_distance(
        seq_a: List[str], seq_b: List[str]
    ) -> float:
        """Compute the normalised Levenshtein distance between two sequences."""
        n, m = len(seq_a), len(seq_b)
        if n == 0 and m == 0:
            return 0.0
        if n == 0 or m == 0:
            return 1.0

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        return dp[n][m] / max(n, m)


# ---------------------------------------------------------------------------
# ReasoningChainAnalyzer
# ---------------------------------------------------------------------------


class ReasoningChainAnalyzer:
    """Analyses reasoning chains extracted from generated plan text."""

    def extract_reasoning_steps(self, text: str) -> List[Dict[str, Any]]:
        """Parse reasoning steps from free text.

        Looks for numbered steps, logical connectives, and premise-conclusion
        patterns.  Returns a list of dicts with keys: ``step``, ``type``,
        ``content``, ``position``.
        """
        if not text.strip():
            return []

        steps: List[Dict[str, Any]] = []

        # Try numbered steps first
        numbered = _extract_numbered_steps(text)
        if numbered:
            for i, step_text in enumerate(numbered):
                pattern = self._classify_step_pattern(step_text)
                steps.append({
                    "step": i + 1,
                    "type": pattern.name,
                    "content": step_text,
                    "position": i / max(1, len(numbered) - 1),
                })
            return steps

        # Fall back to sentence splitting with connective detection
        sentences = re.split(r"(?<=[.!?])\s+", text)
        reasoning_keywords = {
            "because", "therefore", "thus", "hence", "so",
            "since", "given", "assuming", "if", "then",
            "consequently", "this means", "it follows",
            "we can conclude", "this implies", "as a result",
        }
        step_idx = 0
        for i, sentence in enumerate(sentences):
            s_lower = sentence.lower().strip()
            is_reasoning = any(kw in s_lower for kw in reasoning_keywords)
            if is_reasoning or len(s_lower) > 20:
                pattern = self._classify_step_pattern(sentence)
                steps.append({
                    "step": step_idx + 1,
                    "type": pattern.name,
                    "content": sentence.strip(),
                    "position": i / max(1, len(sentences) - 1),
                })
                step_idx += 1

        return steps

    def evaluate_chain_validity(self, text: str) -> float:
        """Evaluate the logical consistency of a reasoning chain.

        Checks for:
        - Premise-conclusion structure
        - No logical contradictions
        - Each step follows from the previous
        - No unsupported leaps

        Returns a score in [0, 1].
        """
        steps = self.extract_reasoning_steps(text)
        if not steps:
            return 0.3

        scores: List[float] = []

        # Check for premise-conclusion structure
        has_premise = any(
            s["type"] in ("DEDUCTIVE", "INDUCTIVE") for s in steps
        )
        has_conclusion = any(
            s["position"] > 0.7 and any(
                kw in s["content"].lower()
                for kw in ["therefore", "conclude", "result", "thus", "hence"]
            )
            for s in steps
        )
        if has_premise and has_conclusion:
            scores.append(1.0)
        elif has_premise or has_conclusion:
            scores.append(0.6)
        else:
            scores.append(0.3)

        # Check for logical flow: look for connectives between consecutive steps
        flow_count = 0
        flow_words = {
            "then", "next", "therefore", "because", "since",
            "consequently", "thus", "so", "hence", "furthermore",
            "moreover", "additionally", "also", "after",
        }
        for step in steps:
            words = set(step["content"].lower().split())
            if words & flow_words:
                flow_count += 1
        if len(steps) > 0:
            scores.append(min(1.0, flow_count / (0.5 * len(steps))))

        # Check for contradictions (simplified: look for negation of earlier claims)
        step_texts = [s["content"].lower() for s in steps]
        contradiction_found = False
        for i in range(len(step_texts)):
            for j in range(i + 1, len(step_texts)):
                # Check if one step negates a claim in another
                words_i = set(step_texts[i].split())
                words_j = set(step_texts[j].split())
                overlap = words_i & words_j
                # Simplified: if high overlap but one has "not" and other doesn't
                if len(overlap) >= 3:
                    has_neg_i = any(
                        w in step_texts[i] for w in ["not", "never", "no", "cannot"]
                    )
                    has_neg_j = any(
                        w in step_texts[j] for w in ["not", "never", "no", "cannot"]
                    )
                    if has_neg_i != has_neg_j and len(overlap) > 4:
                        contradiction_found = True
                        break
            if contradiction_found:
                break
        scores.append(0.2 if contradiction_found else 1.0)

        # Check step count reasonableness
        if 3 <= len(steps) <= 20:
            scores.append(1.0)
        elif len(steps) < 3:
            scores.append(0.5)
        else:
            scores.append(0.7)

        return float(np.mean(scores)) if scores else 0.5

    def compute_reasoning_depth(self, text: str) -> float:
        """Compute the depth of reasoning: how many levels of inference.

        Deeper reasoning chains with nested logical dependencies score higher.
        Returns a score in [0, 1].
        """
        steps = self.extract_reasoning_steps(text)
        if not steps:
            return 0.0

        # Count levels of reasoning by tracking inference indicators
        gen_lower = text.lower()

        # Depth indicators: each adds a level
        depth_markers = [
            "furthermore", "moreover", "building on this",
            "taking this further", "at a deeper level",
            "this in turn", "which then", "leading to",
            "consequently", "as a result of this",
            "the implication is", "this suggests",
            "extending this logic", "by extension",
        ]
        depth_count = sum(1 for m in depth_markers if m in gen_lower)

        # Nested conditionals add depth
        nested_if = len(re.findall(
            r"(?i)\bif\b.{10,100}\bif\b", text
        ))
        depth_count += nested_if

        # Multi-step cause-effect chains
        chain_patterns = re.findall(
            r"(?i)(which\s+(leads?|causes?|results?)\s+in|"
            r"this\s+(leads?|causes?|results?|enables?)\s+)",
            text,
        )
        depth_count += len(chain_patterns)

        # Base depth from step count
        base_depth = min(1.0, len(steps) / 10.0)

        # Depth bonus from markers
        marker_depth = min(1.0, depth_count / 5.0)

        return float(np.clip(0.5 * base_depth + 0.5 * marker_depth, 0.0, 1.0))

    def detect_reasoning_patterns(
        self, text: str
    ) -> Dict[str, float]:
        """Detect the prevalence of different reasoning patterns in the text.

        Returns a dictionary mapping :class:`ReasoningPattern` names to
        confidence scores in [0, 1].
        """
        gen_lower = text.lower()
        results: Dict[str, float] = {}

        # Deductive: general rules → specific conclusions
        deductive_markers = [
            "therefore", "it follows that", "we can conclude",
            "necessarily", "it must be", "logically",
            "given that", "since", "because",
        ]
        deductive_count = sum(1 for m in deductive_markers if m in gen_lower)
        results["DEDUCTIVE"] = min(1.0, deductive_count / 3.0)

        # Inductive: specific observations → general patterns
        inductive_markers = [
            "pattern", "trend", "typically", "generally",
            "in most cases", "evidence suggests", "data shows",
            "observation", "examples indicate", "tends to",
        ]
        inductive_count = sum(1 for m in inductive_markers if m in gen_lower)
        results["INDUCTIVE"] = min(1.0, inductive_count / 3.0)

        # Abductive: best explanation for observations
        abductive_markers = [
            "best explanation", "most likely", "probably",
            "hypothesis", "suggests that", "plausible",
            "could be because", "one explanation", "might be due to",
        ]
        abductive_count = sum(1 for m in abductive_markers if m in gen_lower)
        results["ABDUCTIVE"] = min(1.0, abductive_count / 3.0)

        # Analogical: reasoning by comparison
        analogical_markers = [
            "similar to", "analogous", "just as", "like",
            "comparable to", "reminiscent of", "parallels",
            "in the same way", "by analogy",
        ]
        analogical_count = sum(1 for m in analogical_markers if m in gen_lower)
        results["ANALOGICAL"] = min(1.0, analogical_count / 3.0)

        # Causal: cause-effect reasoning
        causal_markers = [
            "causes", "leads to", "results in", "because of",
            "due to", "effect of", "consequence", "impact",
            "contributes to", "triggers",
        ]
        causal_count = sum(1 for m in causal_markers if m in gen_lower)
        results["CAUSAL"] = min(1.0, causal_count / 3.0)

        return results

    def compute_reasoning_diversity(self, texts: List[str]) -> float:
        """Compute diversity of reasoning patterns across multiple texts.

        Returns [0, 1] where 1.0 means highly diverse reasoning patterns.
        """
        if len(texts) < 2:
            return 0.0

        all_patterns: List[Dict[str, float]] = []
        for text in texts:
            patterns = self.detect_reasoning_patterns(text)
            all_patterns.append(patterns)

        # Convert to vectors
        keys = sorted(all_patterns[0].keys())
        vectors = np.array([
            [p.get(k, 0.0) for k in keys] for p in all_patterns
        ])

        # Pairwise cosine distances
        distances: List[float] = []
        for i, j in combinations(range(len(vectors)), 2):
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            if norm_i > 0 and norm_j > 0:
                cos_sim = float(np.dot(vectors[i], vectors[j]) / (norm_i * norm_j))
                distances.append(1.0 - cos_sim)
            else:
                distances.append(0.0)

        if not distances:
            return 0.0

        # Also consider pattern type diversity: do different texts emphasise
        # different patterns?
        dominant_patterns: List[str] = []
        for p in all_patterns:
            if p:
                dominant = max(p, key=p.get)  # type: ignore[arg-type]
                dominant_patterns.append(dominant)

        unique_dominants = len(set(dominant_patterns))
        type_diversity = (unique_dominants - 1) / max(1, len(dominant_patterns) - 1)

        mean_dist = float(np.mean(distances))
        return float(np.clip(0.6 * mean_dist + 0.4 * type_diversity, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _classify_step_pattern(step_text: str) -> ReasoningPattern:
        """Classify a single reasoning step by its dominant pattern."""
        s = step_text.lower()

        deductive = sum(1 for kw in [
            "therefore", "thus", "hence", "it follows", "must be",
            "necessarily", "conclude", "logically",
        ] if kw in s)

        inductive = sum(1 for kw in [
            "pattern", "trend", "typically", "generally", "evidence",
            "data", "observation", "examples",
        ] if kw in s)

        abductive = sum(1 for kw in [
            "probably", "likely", "hypothesis", "suggests",
            "plausible", "explanation", "might",
        ] if kw in s)

        analogical = sum(1 for kw in [
            "similar", "analogous", "like", "comparable", "parallel",
        ] if kw in s)

        causal = sum(1 for kw in [
            "causes", "leads to", "results in", "because", "due to",
            "effect", "consequence", "impact",
        ] if kw in s)

        scores = {
            ReasoningPattern.DEDUCTIVE: deductive,
            ReasoningPattern.INDUCTIVE: inductive,
            ReasoningPattern.ABDUCTIVE: abductive,
            ReasoningPattern.ANALOGICAL: analogical,
            ReasoningPattern.CAUSAL: causal,
        }

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return ReasoningPattern.UNKNOWN
        return best


# ---------------------------------------------------------------------------
# PlanningTask  (main GenerationTask subclass)
# ---------------------------------------------------------------------------


@GenerationTask.register("planning")
class PlanningTask(GenerationTask):
    """Planning and multi-step reasoning generation task.

    Generates prompts across logistics, project management, cooking,
    travel, problem solving, strategy, constraint satisfaction, and
    causal reasoning domains.  Evaluates plan completeness, ordering,
    feasibility, efficiency, constraint satisfaction, causal coherence,
    detail, robustness, creativity, and diversity.
    """

    def __init__(self, config: Optional[TaskConfig] = None) -> None:
        if config is None:
            config = self.get_default_config()
        super().__init__(config)
        if isinstance(config, PlanningConfig):
            self._planning_config = config
        else:
            self._planning_config = PlanningConfig(
                name=config.name,
                domain=config.domain,
                num_prompts=config.num_prompts,
                max_length=config.max_length,
                min_length=config.min_length,
                temperature=config.temperature,
                constraints=config.constraints,
                evaluation_metrics=config.evaluation_metrics,
                prompt_template=config.prompt_template,
                seed=config.seed,
            )
        self._prompt_gen = PlanningPromptGenerator(seed=self._planning_config.seed)
        self._plan_evaluator = PlanningEvaluator()
        self._reasoning_analyzer = ReasoningChainAnalyzer()

    @classmethod
    def get_default_config(cls) -> PlanningConfig:
        return PlanningConfig(
            name="planning",
            domain=TaskDomain.REASONING,
            num_prompts=50,
            max_length=1024,
            min_length=50,
            temperature=1.0,
            constraints=[],
            evaluation_metrics=[
                "completeness", "step_ordering", "feasibility",
                "efficiency", "constraint_satisfaction",
                "causal_coherence", "detail_level", "robustness",
                "creativity", "diversity",
            ],
            prompt_template="{text}",
            seed=42,
        )

    # -----------------------------------------------------------------
    # Abstract interface implementation
    # -----------------------------------------------------------------

    def load_prompts(self) -> PromptDataset:
        """Load planning prompts from built-in scenarios and procedural generation."""
        all_prompts: List[TaskPrompt] = []

        # Add built-in scenario prompts
        all_prompts.extend(self._prompt_gen.get_builtin_prompts())

        # Generate additional prompts per domain
        per_domain = max(1, self._planning_config.prompts_per_domain)
        for domain_name in self._planning_config.planning_domains:
            domain_name_lower = domain_name.lower().replace(" ", "_")
            generator_map: Dict[str, Callable] = {
                "logistics": self._prompt_gen.generate_logistics_prompts,
                "project_management": self._prompt_gen.generate_project_planning_prompts,
                "cooking": self._prompt_gen.generate_cooking_recipe_prompts,
                "travel": self._prompt_gen.generate_travel_planning_prompts,
                "problem_solving": self._prompt_gen.generate_problem_solving_prompts,
                "strategy": self._prompt_gen.generate_strategy_prompts,
                "constraint_satisfaction": self._prompt_gen.generate_constraint_satisfaction_prompts,
                "causal_reasoning": self._prompt_gen.generate_causal_reasoning_prompts,
            }
            gen_func = generator_map.get(domain_name_lower)
            if gen_func is not None:
                for complexity in self._planning_config.complexity_levels:
                    all_prompts.extend(gen_func(per_domain, complexity))
            else:
                logger.warning("Unknown planning domain: %s", domain_name)

        # Limit to configured number
        if len(all_prompts) > self._planning_config.num_prompts:
            rng = np.random.RandomState(self._planning_config.seed)
            indices = rng.choice(
                len(all_prompts), size=self._planning_config.num_prompts, replace=False
            )
            all_prompts = [all_prompts[i] for i in indices]

        self._dataset = PromptDataset(
            all_prompts, name="planning-prompts", domain=TaskDomain.REASONING
        )
        return self._dataset

    def format_prompt(self, prompt: TaskPrompt) -> str:
        """Format a planning prompt for model consumption."""
        template = self._planning_config.prompt_template
        formatted = template.replace("{text}", prompt.text)

        # Add system-level planning instructions if the template is default
        if template == "{text}":
            system_prefix = (
                "You are a planning assistant. Provide a detailed, "
                "step-by-step plan. For each step, describe the action, "
                "prerequisites, expected outcomes, and estimated time.\n\n"
            )
            formatted = system_prefix + formatted

        if prompt.context:
            formatted += f"\n\nAdditional context: {prompt.context}"

        return formatted

    def evaluate(
        self, generations: List[str], prompts: List[TaskPrompt]
    ) -> Dict[str, float]:
        """Evaluate a batch of plan generations.

        Computes per-generation quality metrics and set-level diversity.
        """
        if len(generations) != len(prompts):
            raise ValueError(
                f"Length mismatch: {len(generations)} generations vs "
                f"{len(prompts)} prompts"
            )

        all_quality: List[Dict[str, float]] = []
        for gen, prompt in zip(generations, prompts):
            scores = self._plan_evaluator.compute_plan_quality_score(gen, prompt)
            all_quality.append(scores)

        # Aggregate quality scores
        aggregated: Dict[str, float] = {}
        if all_quality:
            metric_names = set()
            for q in all_quality:
                metric_names.update(q.keys())
            for metric in metric_names:
                values = [q[metric] for q in all_quality if metric in q]
                if values:
                    aggregated[f"{metric}_mean"] = float(np.mean(values))
                    aggregated[f"{metric}_std"] = float(np.std(values))

        # Diversity scores
        diversity_scores = self._plan_evaluator.compute_plan_diversity_score(
            generations, prompts
        )
        aggregated.update(diversity_scores)

        # Reasoning chain analysis
        depths: List[float] = []
        validity_scores: List[float] = []
        for gen in generations:
            depths.append(self._reasoning_analyzer.compute_reasoning_depth(gen))
            validity_scores.append(
                self._reasoning_analyzer.evaluate_chain_validity(gen)
            )

        aggregated["reasoning_depth_mean"] = float(np.mean(depths)) if depths else 0.0
        aggregated["chain_validity_mean"] = (
            float(np.mean(validity_scores)) if validity_scores else 0.0
        )
        aggregated["reasoning_diversity"] = (
            self._reasoning_analyzer.compute_reasoning_diversity(generations)
        )

        # Overall combined score
        quality_score = aggregated.get("overall_quality_mean", 0.5)
        diversity_score = aggregated.get("overall_diversity", 0.5)
        w_q = self._planning_config.quality_weight
        w_d = self._planning_config.diversity_weight
        total_w = w_q + w_d
        if total_w > 0:
            aggregated["combined_score"] = (
                w_q * quality_score + w_d * diversity_score
            ) / total_w
        else:
            aggregated["combined_score"] = 0.5

        aggregated["num_generations"] = float(len(generations))
        return aggregated

    def get_constraints(self) -> List[TaskConstraint]:
        """Return planning-specific constraints."""
        constraints: List[TaskConstraint] = []

        # Minimum length constraint
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.LENGTH,
            parameters={
                "min": self._planning_config.min_plan_steps * 10,
                "max": self._planning_config.max_length,
                "unit": "words",
            },
            required=True,
            weight=1.0,
        ))

        # Format constraint: plans should have some structure
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.FORMAT,
            parameters={
                "pattern": r"(?:\d+[.)]\s+|\-\s+|\*\s+)",
            },
            required=False,
            weight=0.5,
        ))

        # Content constraint: plans should have action verbs
        constraints.append(TaskConstraint(
            constraint_type=ConstraintType.CONTENT,
            parameters={
                "min_unique_words": 20,
            },
            required=True,
            weight=0.8,
        ))

        return constraints

    def format_output(self, generation: str, prompt: TaskPrompt) -> Dict[str, Any]:
        """Format a generation with its evaluation for reporting."""
        quality = self._plan_evaluator.compute_plan_quality_score(generation, prompt)
        steps = _extract_numbered_steps(generation)
        structure = detect_plan_structure(generation)

        reasoning_steps = self._reasoning_analyzer.extract_reasoning_steps(generation)
        reasoning_patterns = self._reasoning_analyzer.detect_reasoning_patterns(generation)
        reasoning_depth = self._reasoning_analyzer.compute_reasoning_depth(generation)

        return {
            "prompt_id": prompt.prompt_id,
            "prompt_text": prompt.text[:200] + "..." if len(prompt.text) > 200 else prompt.text,
            "generation": generation,
            "num_steps": len(steps),
            "plan_structure": structure.name,
            "quality_scores": quality,
            "reasoning": {
                "num_reasoning_steps": len(reasoning_steps),
                "patterns": reasoning_patterns,
                "depth": reasoning_depth,
            },
            "action_verbs": extract_action_verbs(generation),
            "temporal_markers": [
                m[0] for m in extract_temporal_ordering(generation)
            ],
            "metadata": prompt.metadata,
        }

    def get_prompts(self) -> List[TaskPrompt]:
        """Return the list of planning prompts (loading if necessary)."""
        dataset = self.get_dataset()
        return list(dataset)

    # -----------------------------------------------------------------
    # Additional task methods
    # -----------------------------------------------------------------

    def get_metric_names(self) -> List[str]:
        """Return all metric names computed by this task."""
        base = super().get_metric_names()
        planning_metrics = [
            "completeness", "step_ordering", "feasibility", "efficiency",
            "constraint_satisfaction", "causal_coherence", "detail_level",
            "robustness", "creativity", "overall_quality",
            "strategy_diversity", "step_sequence_diversity",
            "resource_diversity", "abstraction_diversity",
            "structure_diversity", "solution_coverage",
            "overall_diversity", "reasoning_depth",
            "chain_validity", "reasoning_diversity",
            "combined_score",
        ]
        return list(set(base + planning_metrics))

    def describe(self) -> str:
        """Return a human-readable description of this planning task."""
        domains_str = ", ".join(self._planning_config.planning_domains)
        complexity_str = ", ".join(self._planning_config.complexity_levels)
        return (
            f"Task: {self._planning_config.name}\n"
            f"Domain: {self._planning_config.domain.name}\n"
            f"Description: Planning and multi-step reasoning generation\n"
            f"Planning domains: {domains_str}\n"
            f"Complexity levels: {complexity_str}\n"
            f"Prompts: {self._planning_config.num_prompts}\n"
            f"Plan steps: [{self._planning_config.min_plan_steps}, "
            f"{self._planning_config.max_plan_steps}]\n"
            f"Length: [{self._planning_config.min_length}, "
            f"{self._planning_config.max_length}]\n"
            f"Quality/Diversity weights: {self._planning_config.quality_weight}/"
            f"{self._planning_config.diversity_weight}\n"
            f"Metrics: {', '.join(self._planning_config.evaluation_metrics)}"
        )

    def validate_plan_structure(
        self, plan: PlanStructure
    ) -> Tuple[bool, List[str]]:
        """Validate a structured plan (as opposed to free text).

        Checks step count, dependencies, and constraint coverage.
        """
        errors: List[str] = []

        if plan.num_steps < self._planning_config.min_plan_steps:
            errors.append(
                f"Too few steps: {plan.num_steps} < "
                f"{self._planning_config.min_plan_steps}"
            )
        if plan.num_steps > self._planning_config.max_plan_steps:
            errors.append(
                f"Too many steps: {plan.num_steps} > "
                f"{self._planning_config.max_plan_steps}"
            )

        # Validate dependency graph
        if self._planning_config.require_dependencies:
            is_valid_dag, dag_errors = validate_dependency_graph(plan.steps)
            if not is_valid_dag:
                errors.extend(dag_errors)

        # Check preconditions if required
        if self._planning_config.require_preconditions:
            for step in plan.steps:
                if not step.preconditions and step.order > 0:
                    errors.append(
                        f"Step {step.order} has no preconditions"
                    )

        # Check for empty actions
        for step in plan.steps:
            if not step.action.strip():
                errors.append(f"Step {step.order} has an empty action")

        return len(errors) == 0, errors

    def compare_plans(
        self, plan_a: str, plan_b: str
    ) -> Dict[str, float]:
        """Compare two plan texts and return similarity metrics."""
        overall_sim = compute_plan_similarity(plan_a, plan_b)

        struct_a = detect_plan_structure(plan_a)
        struct_b = detect_plan_structure(plan_b)
        same_structure = 1.0 if struct_a == struct_b else 0.0

        steps_a = _extract_numbered_steps(plan_a)
        steps_b = _extract_numbered_steps(plan_b)
        step_count_sim = 1.0 - abs(len(steps_a) - len(steps_b)) / max(
            len(steps_a), len(steps_b), 1
        )

        verbs_a = set(extract_action_verbs(plan_a))
        verbs_b = set(extract_action_verbs(plan_b))
        verb_union = verbs_a | verbs_b
        verb_overlap = len(verbs_a & verbs_b) / len(verb_union) if verb_union else 1.0

        return {
            "overall_similarity": overall_sim,
            "same_structure": same_structure,
            "structure_a": struct_a.name,
            "structure_b": struct_b.name,
            "step_count_similarity": step_count_sim,
            "verb_overlap": verb_overlap,
            "steps_a": len(steps_a),
            "steps_b": len(steps_b),
        }
