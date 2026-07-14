import json, os
from openai import OpenAI
import classify_factoids as cf

c = OpenAI(api_key=os.environ["VIRTUAL_API_KEY"], base_url=os.environ["BASE_URL"])

batch = [json.loads(l) for l in open("sample.jsonl")][:10]
user = cf.build_user_prompt(batch)

trials = [
    ("reasoning_low max=3000", dict(max_completion_tokens=3000, extra_body={"reasoning_effort": "low"})),
    ("reasoning_low max=1500", dict(max_completion_tokens=1500, extra_body={"reasoning_effort": "low"})),
    ("enable_thinking_false max=4000", dict(max_completion_tokens=4000, extra_body={"chat_template_kwargs": {"enable_thinking": False}})),
]
for name, kw in trials:
    try:
        r = c.chat.completions.create(
            model="GPT-OSS-120B",
            messages=[{"role": "system", "content": cf.SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0, **kw)
        m = r.choices[0].message
        rc = getattr(m, "reasoning_content", None)
        parsed = cf.parse_response(m.content or "", len(batch))
        n_ok = sum(1 for p in parsed if p != cf._DEFAULT) if parsed else 0
        print("\n===", name, "===")
        print("finish=%r reasoning_len=%s content_len=%s parsed=%s nondefault=%s"
              % (r.choices[0].finish_reason, (len(rc) if rc else rc),
                 len(m.content or ""), (len(parsed) if parsed else None), n_ok))
        print("content[:400]:", (m.content or "")[:400])
    except Exception as e:
        print("\n===", name, "=== ERR", type(e).__name__, e)
