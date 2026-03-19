import base64
import json
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

instruction = "How can we prevent potential hazards for babies in this room?"

# =============================
# 유틸
# =============================
def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    text = text.replace("json", "").strip()
    return json.loads(text)

# =============================
# 출력
# =============================
def print_instruction():
    print("\n==============================")
    print("Instruction")
    print("==============================")
    print(instruction)

def print_perception(memory):
    print("\n===== Initial Perception =====")
    for agent in memory:
        print(f"\n[{agent}]")
        for obj in memory[agent]["objects"]:
            print(f"- {obj['name']} ({obj['location']})")

def print_turn(agent, msg):
    print(f"\n[{agent}]")
    print("Role:", msg["role_assignment"].get(agent, ""))

    print("Plan Update:")
    for step in msg["plan_update"]:
        print(" ", step)

def print_goal(goal):
    print("\n===== Final Goal =====")
    print(goal)

def print_final(plan):
    print("\n===== Final Plan (Timeline) =====")
    for step in plan:
        print(step)

# =============================
# 이미지
# =============================
img1 = encode("scene_4_1.png")
img2 = encode("scene_4_2.png")

# =============================
# Perception
# =============================
def perception(agent_name, image):
    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.
Extract all objects with location.

Return JSON:
{{
  "objects": [
    {{"name": "", "location": ""}}
  ]
}}
"""
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            ]
        }]
    )
    return extract_json(res.choices[0].message.content)

# =============================
# Agent Step
# =============================
def agent_step(agent_name, memory, dialogue):

    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.

Your environment:
{memory[agent_name]}

Other agent environment:
{memory}

Dialogue:
{dialogue}

GOAL:
Create a collaborative plan.

CRITICAL RULES:

1. ROLE SPLIT
Each agent must have different responsibilities.

2. GLOBAL TIMELINE
Steps must be ordered (Step1, Step2...)

3. INTERACTION REQUIRED
Agents must:
- pass objects
- request objects
- move between rooms

4. LOCATION REQUIRED
Each step must include object + location

5. REALISTIC FOOD ONLY

6. BALANCED WORKLOAD

7. FINAL PLAN CONDITION
Only generate final_plan when:
- both agents have agreed
- plan is stable

RETURN JSON:

{{
  "role_assignment": {{
    "Agent1": "",
    "Agent2": ""
  }},
  "plan_update": [],
  "agreement": "",
  "final_plan": null or [],
  "final_goal": "" or null
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return extract_json(res.choices[0].message.content)

# =============================
# 실행
# =============================
print_instruction()

memory = {
    "Agent1": perception("Agent1", img1),
    "Agent2": perception("Agent2", img2)
}

print_perception(memory)

dialogue = []
plan1 = None
plan2 = None
final_plan = None
final_goal = None

turn = 1

while True:

    print(f"\n==============================")
    print(f"Turn {turn}")
    print("==============================")

    msg1 = agent_step("Agent1", memory, dialogue)
    print_turn("Agent1", msg1)
    dialogue.append(f"Agent1: {msg1}")

    msg2 = agent_step("Agent2", memory, dialogue)
    print_turn("Agent2", msg2)
    dialogue.append(f"Agent2: {msg2}")

    if msg1["final_plan"]:
        plan1 = msg1["final_plan"]

    if msg2["final_plan"]:
        plan2 = msg2["final_plan"]

    # ✅ 합의 기반 종료
    if plan1 and plan2:
        final_plan = plan1
        final_goal = msg1.get("final_goal", "")
        break

    turn += 1

# =============================
# 출력
# =============================
if final_plan:
    print_goal(final_goal)
    print_final(final_plan)
else:
    print("\nNo final plan generated")