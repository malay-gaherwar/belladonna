import os
from openai import OpenAI

c = OpenAI(api_key=os.environ["VIRTUAL_API_KEY"], base_url=os.environ["BASE_URL"])

print("=== models ===")
try:
    for m in c.models.list().data:
        print("  ", m.id)
except Exception as e:
    print("models err:", e)

print("=== chat variants ===")
msgs = [{"role": "user", "content": "Reply with the single word: OK"}]
variants = [
    ("A enable_thinking=False+max_completion_tokens",
     dict(max_completion_tokens=50, extra_body={"chat_template_kwargs": {"enable_thinking": False}})),
    ("B reasoning_effort=low",
     dict(max_completion_tokens=300, extra_body={"reasoning_effort": "low"})),
    ("C plain max_tokens",
     dict(max_tokens=50)),
    ("D max_completion_tokens only",
     dict(max_completion_tokens=300)),
]
for name, kw in variants:
    try:
        r = c.chat.completions.create(model="GPT-OSS-120B", messages=msgs, **kw)
        m = r.choices[0].message
        reasoning = getattr(m, "reasoning_content", None)
        print("[%s] finish=%r content=%r reasoning_len=%s"
              % (name, r.choices[0].finish_reason, m.content,
                 (len(reasoning) if reasoning else reasoning)))
    except Exception as e:
        print("[%s] ERR %s: %s" % (name, type(e).__name__, e))
