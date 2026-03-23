import base64
import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

instruction = "I want to watch TV with 5 friends."

# =============================
# 유틸
# =============================
def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_json(text):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
        text = text.replace("json", "").strip()
        return json.loads(text)
    except:
        return {}

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
        for obj in memory[agent].get("objects", []):
            loc = obj.get("location", {})
            print(f"  - {obj['name']}")
            print(f"     absolute: {loc.get('absolute','')}")
            print(f"     relative: {loc.get('relative','')}")
            print(f"     relation: {loc.get('relation','')}")

def print_turn(agent, msg):
    print(f"\n[{agent}]")

    print("Self Plan:")
    for step in msg.get("self_plan", []):
        print(" ", step)

    print("Critique:")
    print(" ", msg.get("critique", ""))

    print("Update:")
    print(" ", msg.get("proposal_update", ""))

    print("Agreement:")
    print(" ", msg.get("agreement", ""))

def print_final(plan):
    print("\n===== Final Plan =====")

    for i, step in enumerate(plan, 1):

        if isinstance(step, dict):
            print(f"\nStep {i}: {step.get('action','')}")
            print(f"  From: {step.get('from','')}")
            print(f"  To: {step.get('to','')}")
            print(f"  Relation: {step.get('relation','')}")
        else:
            print(f"\nStep {i}: {step}")

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

Extract objects with detailed spatial information.

Return JSON:

{{
  "objects": [
    {{
      "name": "",
      "location": {{
        "absolute": "",
        "relative": "",
        "relation": ""
      }}
    }}
  ]
}}
"""

    model = "gpt-4o" if agent_name == "Agent1" else "gpt-4o-mini"

    res = client.chat.completions.create(
        model=model,
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
# Agent Step (핵심)
# =============================
def agent_step(agent_name, memory, dialogue):

    if agent_name == "Agent1":
        model = "gpt-4o"
        temp = 0.6
    else:
        model = "gpt-4o-mini"
        temp = 0.9

    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.

Memory:
{memory}

Dialogue:
{dialogue}

DYNAMIC COLLABORATION RULES:

1. Propose your own plan

2. Critique the other agent:
- You MUST find at least one flaw OR improvement

3. STRICT NEGOTIATION RULES:
- You MUST NOT repeat previous plan
- You MUST change at least one action every turn
- You MUST disagree at least once before agreement

4. If disagreement:
- suggest better alternative

5. Use spatial reasoning:
- from where → to where → relation

6. Finalize only when both agents agree

7. If your plan is identical to previous turn, you must modify it

RETURN JSON:

{{
  "self_plan": [],
  "critique": "",
  "proposal_update": "",
  "agreement": "",
  "final_plan": null or [],
  "final_goal": "" or null
}}

FINAL PLAN FORMAT:

- final_plan must be a list of dictionaries
- each step must follow:

{{
  "action": "",
  "from": "",
  "to": "",
  "relation": ""
}}

DO NOT return strings in final_plan.

Only generate final_plan when:
- both agents explicitly agree
- no conflicts remain
"""

    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
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
final_plan = None

turn = 1
MAX_TURN = 10  # 무한루프 방지

while turn <= MAX_TURN:

    print(f"\n==============================")
    print(f"Turn {turn}")
    print("==============================")

    msg1 = agent_step("Agent1", memory, dialogue)
    print_turn("Agent1", msg1)
    dialogue.append(f"A1: {msg1}")

    msg2 = agent_step("Agent2", memory, dialogue)
    print_turn("Agent2", msg2)
    dialogue.append(f"A2: {msg2}")

    if msg1.get("final_plan"):
        final_plan = msg1["final_plan"]
        break

    if msg2.get("final_plan"):
        final_plan = msg2["final_plan"]
        break

    turn += 1

# =============================
# 출력
# =============================
if final_plan:
    print_final(final_plan)
else:
    print("\nNo final plan (max turn reached)")