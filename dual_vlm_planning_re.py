import base64
import json
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

instruction = "I want to play with my 5 friends"

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

    print("Agreement:", msg.get("agreement", ""))

def print_final(plan):
    print("\n===== Final Plan (Merged) =====")
    for step in plan:
        print(step)

# =============================
# 이미지
# =============================
img1 = encode("scene_1_1.png")
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
Collaboratively complete the task.

CRITICAL RULES:

1. LOCAL PLAN ONLY
- You MUST generate ONLY your own actions
- Every step must start with {agent_name}

2. NO CONTROL OVER OTHER AGENT
- You MUST NOT generate or modify the other agent's actions

3. SUGGESTION ONLY
- You may suggest improvements for the other agent
- But DO NOT include their actions in your plan

4. INTERACTION
- You can request objects
- You can respond to requests

5. LOCATION REQUIRED
Each step must include object + location

6. AGREEMENT RULE
- If the other agent's plan is acceptable and no changes are needed:
  → agreement = "agree"
- Otherwise:
  → agreement = "disagree"

RETURN JSON:

{{
  "role_assignment": {{
    "Agent1": "",
    "Agent2": ""
  }},
  "plan_update": [
    {{
      "step": "",
      "description": ""
    }}
  ],
  "suggestions_for_other": [
    ""
  ],
  "requests": [
    ""
  ],
  "agreement": ""
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return extract_json(res.choices[0].message.content)

# =============================
# Merge
# =============================
def merge_plans(plan1, plan2):
    return plan1 + plan2

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

plan1 = []
plan2 = []
final_plan = None

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

    # 최신 plan 저장
    if msg1["plan_update"]:
        plan1 = msg1["plan_update"]

    if msg2["plan_update"]:
        plan2 = msg2["plan_update"]

    agree1 = msg1.get("agreement") == "agree"
    agree2 = msg2.get("agreement") == "agree"

    # ✅ 수렴 조건 (핵심)
    if agree1 and agree2:
        final_plan = merge_plans(plan1, plan2)
        break

    turn += 1

# =============================
# 출력
# =============================
if final_plan:
    print_final(final_plan)
else:
    print("\nNo final plan generated")