# Dual-VLM Planning

## Overview

This project explores collaborative task planning using Vision-Language Models (VLMs), with a focus on **reasoning under uncertainty**. It implements a framework where a model analyzes a scene, generates plans, detects uncertainty, and actively asks clarification questions to improve decision-making.

The core idea is to move beyond one-shot planning and enable **iterative reasoning with uncertainty awareness**.

---

## Key Components

1. **Scene Understanding**
   The model observes an input image and extracts structured information about objects, including their location and state.

2. **Planning**
   Given an instruction and scene description, the model selects a target object and generates a plan to achieve the goal.

3. **Self-Consistency Sampling**
   Multiple plans are sampled to observe variation in selected target objects.

4. **Uncertainty Estimation**
   Uncertainty is computed based on disagreement between sampled targets. Higher diversity in outputs indicates higher uncertainty.

5. **Uncertainty Analysis**
   The model analyzes why uncertainty occurs (e.g., ambiguous objects, spatial confusion, preference issues).

6. **Re-questioning**
   When uncertainty is high, the model generates a clarification question and incorporates user feedback.

7. **Iterative Refinement**
   The process repeats until the model becomes confident or reaches a maximum number of iterations.

---

## Project Structure

```
.
├── dual_vlm_planning.py       # Basic dual-VLM planning pipeline
├── uncertainty_requestion.py  # Uncertainty-aware planning with self-consistency and interactive questioning
├── scene_*.png                # Input scene images
└── README.md
```

---

## How It Works

1. User provides an instruction
2. Model extracts scene information from image
3. Model generates multiple candidate plans
4. Uncertainty is computed based on disagreement
5. If uncertainty is high:
   - Analyze cause
   - Generate a clarification question
   - Update instruction with user input
6. Repeat until confident
7. Output final plan

---

## Example

**Instruction:** How can we prevent potential hazards for babies in this room?

**Process:**
- Detect objects in scene
- Generate multiple plans
- Identify disagreement in target objects
- Ask clarification question
- Refine plan

---

## Key Idea

This project introduces a simple but effective loop:

```
Perception → Planning → Sampling → Uncertainty → Question → Refinement
```

This enables more reliable decision-making compared to single-pass generation.

---

## Data Source

Some scene images used in this project are obtained from the **AI2-THOR** interactive demo environment provided by the Allen Institute for AI:

> https://ai2thor.allenai.org/demo/