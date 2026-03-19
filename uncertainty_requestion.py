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
# Scene 이해 (VLM)
# =============================
def perception(image):

    prompt = """
Describe the scene in structured JSON.

Return:
{
  "objects": [
    {"name": "", "location": "", "state": ""}
  ]
}
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
# 후보 + confidence 생성
# =============================
def generate_candidates(scene, context):

    prompt = f"""
Instruction:
{context["instruction"]}

User preference so far:
{context.get("preference", "")}

Scene:
{scene}

You are making a decision under uncertainty.

Your goal:
- propose possible choices
- estimate confidence
- update your belief over time

Important:
- Confidence should reflect your current belief
- If user preference exists, incorporate it
- Over iterations, your confidence should become sharper if information increases

Return JSON:
{{
  "candidates": [
    {{"object": "", "confidence": 0.0}}
  ]
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return extract_json(res.choices[0].message.content)

# =============================
# Uncertainty (entropy)
# =============================
def compute_uncertainty(candidates):

    probs = np.array([c["confidence"] for c in candidates])
    probs = probs / probs.sum()

    entropy = -np.sum(probs * np.log(probs + 1e-9))

    return entropy

# =============================
# 질문 생성 (self-reflective)
# =============================
def generate_question(scene, context, candidates):

    prompt = f"""
Instruction:
{context["instruction"]}

User preference so far:
{context.get("preference", "")}

Scene:
{scene}

Candidates:
{candidates}

You are trying to reduce uncertainty.

Think:
- Why is the decision still ambiguous?
- What missing information would change your confidence?

Ask ONE question that would most reduce uncertainty.

Do not ask generic questions.
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# =============================
# Plan 생성
# =============================
def generate_plan(scene, context, target):

    prompt = f"""
Instruction:
{context["instruction"]}

User preference:
{context.get("preference", "")}

Scene:
{scene}

Selected object:
{target}

Generate a simple step-by-step plan.
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# =============================
# 출력
# =============================
def print_candidates(candidates):
    print("\n[Candidates]")
    for c in candidates:
        print(f"- {c['object']}: {c['confidence']:.2f}")

# =============================
# MAIN LOOP
# =============================
def run():

    image = encode(image_path)
    scene = perception(image)

    print("\n[Scene]")
    print(scene)

    context = {
        "instruction": instruction,
        "preference": ""
    }

    MAX_TURN = 3
    entropy_threshold = 0.4
    confidence_threshold = 0.8

    for t in range(MAX_TURN):

        print(f"\n--- Iteration {t+1} ---")

        result = generate_candidates(scene, context)
        candidates = result["candidates"]

        print_candidates(candidates)

        U = compute_uncertainty(candidates)
        print(f"\n[Entropy]: {U:.3f}")

        best = max(candidates, key=lambda x: x["confidence"])
        print(f"[Best]: {best['object']} ({best['confidence']:.2f})")

        # 종료 조건 (확신 기반)
        if best["confidence"] > confidence_threshold:
            print("\n[CONFIDENT → PLAN]")
            return generate_plan(scene, context, best["object"])

        # 불확실 → 질문
        print("\n[UNCERTAIN → ASK]")

        question = generate_question(scene, context, candidates)
        print("\n[Question]")
        print(question)

        user_answer = input("\nUser: ")

        # belief 업데이트 (누적)
        context["preference"] += " " + user_answer

    # fallback
    print("\n[FALLBACK PLAN]")
    return generate_plan(scene, context, best["object"])

# =============================
# 실행
# =============================
result = run()

print("\n[Final Result]")
print(result)