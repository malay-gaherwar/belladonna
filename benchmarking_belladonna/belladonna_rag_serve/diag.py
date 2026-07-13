import os, time, sys
os.environ.setdefault('BASE_URL', 'http://your-gateway/v1/')
os.environ.setdefault('VIRTUAL_API_KEY', 'YOUR_VIRTUAL_API_KEY')
os.environ.setdefault('BELLADONNA_VECTOR_BACKEND','chroma')

def step(name, fn):
    t=time.time()
    try:
        r=fn(); print(f'[{time.time()-t:5.1f}s] {name}: OK', flush=True); return r
    except Exception as e:
        print(f'[{time.time()-t:5.1f}s] {name}: ERROR {type(e).__name__}: {str(e)[:200]}', flush=True); return None

q='In HER2+ MBC after trastuzumab+taxane, preferred next therapy? A) T-DXd B) Vinorelbine C) Observation'
from llm import route_sources, generate_grounded_answer
from retriever import retrieve
from critic import grade_evidence

routed = step('route_sources', lambda: route_sources(q, model='glm-4.7-flash'))
print('  routed =', routed, flush=True)
ev = step('retrieve', lambda: retrieve(question=q, sources=routed or ['AGO','ESMO'], top_k=15))
print('  evidence count =', len(ev) if ev else 0, flush=True)
if ev:
    v = step('grade_evidence', lambda: grade_evidence(q, ev, model='glm-4.7-flash'))
    ans = step('generate_grounded_answer', lambda: generate_grounded_answer(q, ev, verdict=v, model='glm-4.7-flash'))
    print('  answer tail:', repr((ans or '')[-150:]), flush=True)
