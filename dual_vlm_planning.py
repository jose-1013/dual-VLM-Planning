import base64
import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

instruction = "create a space where I can rest comfortably."

# =============================
# JSON 파싱
# =============================
def extract_json(text):
    if not text:
        return {}

    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    text = text.replace("json", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return {}

    try:
        return json.loads(text[start:end+1])
    except:
        return {}

# =============================
# 이미지 encode
# =============================
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
# perception
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
# 초기 plan 생성
# =============================
def initial_plan(agent_name, memory):

    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.

Your environment:
{memory[agent_name]}

Create an initial plan ONLY based on your observation.

Return JSON:
{{
  "plan": [
    {{"action": "", "target": "", "agent": "{agent_name}"}}
  ]
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    msg = extract_json(res.choices[0].message.content)
    return msg.get("plan", [])

# =============================
# plan validation
# =============================
def normalize_step(step):
    if isinstance(step, dict):
        return step
    return None

def validate_plan(plan):
    valid = []
    if not isinstance(plan, list):
        return valid

    for step in plan:
        step = normalize_step(step)
        if step:
            valid.append(step)

    return valid

# =============================
# 🔥 STRICT AGREEMENT
# =============================
def canonical_step(step):
    return (
        step.get("action", "").lower().strip(),
        step.get("target", "").lower().strip(),
        step.get("agent", "").lower().strip()
    )

def compute_agreement(a, b):
    if not a or not b:
        return 0.0

    a_steps = [canonical_step(x) for x in a]
    b_steps = [canonical_step(x) for x in b]

    # set 기반
    set_match = len(set(a_steps) & set(b_steps))
    set_score = set_match / max(len(a_steps), len(b_steps))

    # 순서 기반
    order_match = 0
    for i in range(min(len(a_steps), len(b_steps))):
        if a_steps[i] == b_steps[i]:
            order_match += 1

    order_score = order_match / max(len(a_steps), len(b_steps))

    # 🔥 hybrid (엄격)
    return 0.7 * order_score + 0.3 * set_score

# =============================
# agent step (critic)
# =============================
def agent_step(agent_name, memory, dialogue, current_plan):

    other = "Agent1" if agent_name == "Agent2" else "Agent2"

    prompt = f"""
Instruction:
{instruction}

You are {agent_name}.

Current Plan:
{current_plan}

Your environment:
{memory[agent_name]}

Other agent environment:
{memory[other]}

Dialogue:
{dialogue}

GOAL:
Improve the plan collaboratively.

RULES:
- Only modify if necessary
- Add at most ONE new step
- If plan is sufficient → return unchanged

Return JSON:

{{
  "role_assignment": {{
    "Agent1": "",
    "Agent2": ""
  }},
  "plan_update": [
    {{"action": "", "target": "", "agent": ""}}
  ]
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

img1 = encode("scene_5.png")
img2 = encode("scene_4_2.png")

memory = {
    "Agent1": perception("Agent1", img1),
    "Agent2": perception("Agent2", img2)
}

print_perception(memory)

# 초기 plan
plan1 = initial_plan("Agent1", memory)
plan2 = initial_plan("Agent2", memory)

current_plan = plan1 + plan2
dialogue = []

THRESHOLD = 0.9
MAX_TURN = 5
turn = 1

while turn <= MAX_TURN:

    print(f"\n==============================")
    print(f"Turn {turn}")
    print("==============================")

    prev_plan = current_plan.copy()

    # Agent1
    msg1 = agent_step("Agent1", memory, dialogue, current_plan)
    plan1 = validate_plan(msg1["plan_update"])

    print_turn("Agent1", msg1)
    dialogue.append(f"Agent1: {msg1}")

    # Agent2
    msg2 = agent_step("Agent2", memory, dialogue, plan1)
    plan2 = validate_plan(msg2["plan_update"])

    print_turn("Agent2", msg2)
    dialogue.append(f"Agent2: {msg2}")

    agreement1 = compute_agreement(prev_plan, plan1)
    agreement2 = compute_agreement(plan1, plan2)

    final_agreement = (agreement1 + agreement2) / 2

    print(f"\nAgent1 Agreement: {agreement1:.2f}")
    print(f"Agent2 Agreement: {agreement2:.2f}")
    print(f"Final Agreement: {final_agreement:.2f}")

    current_plan = plan2 if plan2 else plan1

    # 🔥 agreement만으로 종료
    if final_agreement > THRESHOLD:
        print(">> Converged")
        break

    turn += 1

# 결과
if current_plan:
    print_final(current_plan)
else:
    print("\nNo final plan generated")