import base64
import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

instruction = "Two guests are arriving in 15 minutes, so please make sure to get everything ready in time."

# =============================
# 유틸 (안전 JSON)
# =============================
def extract_json(text):
    if not text or text.strip() == "":
        return {}

    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]

    text = text.replace("json", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return {}

    try:
        return json.loads(text[start:end+1])
    except:
        return {}

def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

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
    print("Role:", msg.get("role_assignment", {}).get(agent, ""))
    print("Plan Update:")
    for step in msg.get("plan_update", []):
        print(" ", step)

def print_final(plan):
    print("\n===== Final Plan (Timeline) =====")
    for step in plan:
        print(step)

# =============================
# Perception
# =============================
def perception(agent_name, image):
    prompt = f"""
Instruction:
{instruction}

Extract objects with location.

Return JSON:
{{ "objects": [{{"name": "", "location": ""}}] }}
"""
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            ]
        }],
        temperature=0.3
    )

    return extract_json(res.choices[0].message.content)

# =============================
# Object + Validation
# =============================
def get_available_objects(memory):
    return list(set(
        obj["name"] for agent in memory for obj in memory[agent]["objects"]
    ))

def normalize_step(step):
    if isinstance(step, dict):
        return step
    if isinstance(step, str):
        return {"task": step, "object": ""}
    return None

def validate_plan(plan, available_objects):
    valid = []

    if not isinstance(plan, list):
        return valid

    for step in plan:
        step = normalize_step(step)
        if not step:
            continue

        obj = str(step.get("object", "")).lower()

        # object 없으면 통과
        if obj == "":
            valid.append(step)
            continue

        if any(o.lower() in obj for o in available_objects):
            valid.append(step)

    return valid

# =============================
# Agreement
# =============================
def compute_agreement(plan_a, plan_b):
    if not plan_a or not plan_b:
        return 0.0

    set_a = set([str(p) for p in plan_a])
    set_b = set([str(p) for p in plan_b])

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0

# =============================
# Growth 제한
# =============================
def limit_growth(old_plan, new_plan, max_add=2):
    if len(new_plan) > len(old_plan) + max_add:
        return new_plan[:len(old_plan) + max_add]
    return new_plan

# =============================
# Agent Step
# =============================
def agent_step(agent_name, memory, dialogue, current_plan):

    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.

Current Plan:
{current_plan}

Your environment:
{memory[agent_name]}

Other agent environment:
{memory}

Dialogue:
{dialogue}

Modify the plan. Keep useful steps.

Return JSON ONLY:

{{
  "role_assignment": {{
    "Agent1": "",
    "Agent2": ""
  }},
  "plan_update": []
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    msg = extract_json(res.choices[0].message.content)

    if "plan_update" not in msg:
        msg["plan_update"] = []
    if "role_assignment" not in msg:
        msg["role_assignment"] = {"Agent1": "", "Agent2": ""}

    return msg

# =============================
# 실행
# =============================
print_instruction()

img1 = encode("scene_1_1.png")
img2 = encode("scene_4_2.png")

memory = {
    "Agent1": perception("Agent1", img1),
    "Agent2": perception("Agent2", img2)
}

print_perception(memory)

available_objects = get_available_objects(memory)

dialogue = []
current_plan = []

THRESHOLD = 0.85
MAX_TURN = 6

turn = 1

while turn <= MAX_TURN:

    print(f"\n==============================")
    print(f"Turn {turn}")
    print("==============================")

    prev_plan = current_plan.copy()

    # -------- Agent1 --------
    msg1 = agent_step("Agent1", memory, dialogue, current_plan)
    plan1 = validate_plan(msg1["plan_update"], available_objects)

    print_turn("Agent1", msg1)
    dialogue.append(f"Agent1: {msg1}")

    # -------- Agent2 --------
    msg2 = agent_step("Agent2", memory, dialogue, plan1)
    plan2 = validate_plan(msg2["plan_update"], available_objects)

    plan2 = limit_growth(plan1, plan2)

    print_turn("Agent2", msg2)
    dialogue.append(f"Agent2: {msg2}")

    # -------- agreement 계산 --------
    agreement1 = compute_agreement(prev_plan, plan1)
    agreement2 = compute_agreement(plan1, plan2)

    final_agreement = (agreement1 + agreement2) / 2

    print(f"\nAgent1 Agreement: {agreement1:.2f}")
    print(f"Agent2 Agreement: {agreement2:.2f}")
    print(f"Final Agreement: {final_agreement:.2f}")

    current_plan = plan2 if plan2 else plan1

    if final_agreement > THRESHOLD:
        print(">> Converged")
        break

    turn += 1

# =============================
# 출력
# =============================
if current_plan:
    print_final(current_plan)
else:
    print("\nNo final plan generated")