import base64
import json
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================
# 입력
# =============================
instruction = input("Instruction: ")
image_path = "scene_4_2.png"

# =============================
# 유틸
# =============================
def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_json(text):
    import re
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("JSON parsing failed")

# =============================
# Scene 이해
# =============================
def perception(image):

    prompt = f"""
Instruction:
{instruction}

Describe the scene in structured JSON.

Return:
{{
  "objects": [
    {{"name": "", "location": "", "state": ""}}
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
# Planning (한 번 실행)
# =============================
def generate_plan(scene, query):

    prompt = f"""
Instruction:
{query}

Scene:
{scene}

Select the best action plan.

Return JSON:
{{
  "plan": [],
  "target_object": "",
  "reason": ""
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    return extract_json(res.choices[0].message.content)

# =============================
# Self-consistency
# =============================
def sample_plans(scene, query, n=5):

    outputs = []

    for _ in range(n):
        try:
            plan = generate_plan(scene, query)
            outputs.append(json.dumps(plan))
        except:
            continue

    return outputs

# =============================
# Uncertainty 계산
# =============================
def compute_uncertainty(samples):

    if len(samples) <= 1:
        return 1.0

    unique = len(set(samples))
    total = len(samples)

    return unique / total  # 다양성

# =============================
# Uncertainty 원인 분석
# =============================
def analyze_uncertainty(scene, query, samples):

    prompt = f"""
Instruction:
{query}

Scene:
{scene}

Generated plans:
{samples}

Why is the model uncertain?

Return JSON:
{{
  "type": "object / preference / spatial / planning",
  "reason": ""
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return extract_json(res.choices[0].message.content)

# =============================
# 질문 생성 (핵심)
# =============================
def generate_question(scene, query, reason):

    prompt = f"""
Instruction:
{query}

Scene:
{scene}

Uncertainty:
{reason}

Ask ONE question to reduce uncertainty.

Rules:
- specific
- mention objects
- not vague
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# =============================
# MAIN LOOP
# =============================
def run():

    image = encode(image_path)
    scene = perception(image)

    print("\n[Scene]")
    print(scene)

    query = instruction

    MAX_TURN = 3
    threshold = 0.4

    for t in range(MAX_TURN):

        print(f"\n--- Iteration {t+1} ---")

        samples = sample_plans(scene, query)
        U = compute_uncertainty(samples)

        print(f"[Uncertainty]: {U:.3f}")

        # 확신 높음 → 바로 실행
        if U < threshold:
            print("\n[CONFIDENT → PLAN]")
            final = generate_plan(scene, query)
            return final

        # 불확실 → 질문
        print("\n[UNCERTAIN → ASK]")

        reason = analyze_uncertainty(scene, query, samples)
        print("\n[Reason]")
        print(reason)

        question = generate_question(scene, query, reason)
        print("\n[Question]")
        print(question)

        user_answer = input("\nUser: ")

        query = query + " " + user_answer

    # fallback
    print("\n[FALLBACK PLAN]")
    return generate_plan(scene, query)

# =============================
# 실행
# =============================
result = run()

print("\n[Final Result]")
print(result)