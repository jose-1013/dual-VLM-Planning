# Dual-VLM Planning with Uncertainty and Re-questioning

## Overview

This project investigates task planning using Vision-Language Models (VLMs), with a focus on improving reliability through **uncertainty estimation** and **interactive questioning**.

The ultimate goal is to develop a unified framework that combines:

- Dual-VLM collaborative planning
- Uncertainty-aware reasoning
- Re-questioning mechanism for clarification

At the current stage, the system is divided into two separate experimental setups:

1. Dual-VLM collaborative planning
2. Uncertainty-based interactive planning

---

## Research Objective

The core objective of this project is to answer the following:

> **Can task planning be improved by combining multi-agent collaboration with explicit uncertainty modeling and human interaction?**

To achieve this, the project explores:

- How two VLM agents collaborate to generate plans
- How uncertainty emerges during decision-making
- How asking clarification questions can reduce uncertainty

---

## Current Experimental Setup

The project is currently structured into two independent approaches:

1. Dual-VLM Planning
2. Uncertainty-aware Planning

These are evaluated separately before integrating them into a unified framework.

---

## Method 1: Dual-VLM Collaborative Planning

This approach models planning as a collaboration between two agents.

Each agent:
- Observes a different scene
- Extracts objects and spatial information
- Communicates with the other agent

Through iterative dialogue, the agents:
- Assign roles
- Propose plan updates
- Reach agreement on a final plan

**Key characteristics:**
- Multi-agent reasoning
- Role-based task decomposition
- Interaction between agents
- Agreement-based termination

---

## Method 2: Uncertainty-aware Planning with Re-questioning

This approach focuses on improving decision reliability.

**Process:**
1. Generate multiple candidate decisions
2. Estimate uncertainty using entropy
3. If uncertainty is high:
   - Analyze the cause
   - Generate a clarification question
   - Incorporate user feedback
4. Repeat until confidence is sufficient

**Key characteristics:**
- Self-consistency sampling
- Entropy-based uncertainty estimation
- Interactive questioning
- Incremental belief update

---

## Planned Integration

The final goal is to combine both approaches into a single system:

```
Dual-VLM + Uncertainty + Re-questioning
```

**Expected behavior:**
- Two agents collaborate to generate plans
- Each agent estimates its own uncertainty
- When disagreement or uncertainty is high:
  - Agents ask clarification questions
  - Or request human input
- The system refines the plan iteratively

**This integration aims to achieve:**
- More robust planning
- Reduced hallucination
- Better alignment with user intent

---

## Files

```
.
├── dual_vlm_planning.py       # Implements dual-agent collaborative planning
├── uncertainty_requestion.py  # Implements uncertainty estimation and re-questioning mechanism
├── scene_*.png                # Input scene images
└── README.md
```

---

## Data Source

Some scene images used in this project are obtained from the **AI2-THOR** interactive demo environment provided by the Allen Institute for AI:

> https://ai2thor.allenai.org/demo/

---