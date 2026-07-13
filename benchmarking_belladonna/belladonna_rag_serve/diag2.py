import os, time
os.environ.setdefault('BASE_URL', 'http://your-gateway/v1/')
os.environ.setdefault('VIRTUAL_API_KEY', 'YOUR_VIRTUAL_API_KEY')
os.environ.setdefault('BELLADONNA_VECTOR_BACKEND','chroma')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_IMPL'] = 'none'
from concurrent.futures import ThreadPoolExecutor
from retriever import retrieve
q='HER2+ MBC after trastuzumab and taxane preferred next therapy'

def work():
    t=time.time()
    ev=retrieve(question=q, sources=['AGO','ESMO'], top_k=15)
    return f'thread retrieve OK in {time.time()-t:.1f}s, {len(ev)} hits'

print('running retrieve INSIDE a worker thread (mimics uvicorn)...', flush=True)
with ThreadPoolExecutor(max_workers=4) as ex:
    fut=ex.submit(work)
    try:
        print(fut.result(timeout=60), flush=True)
    except Exception as e:
        print('THREAD HANG/ERROR:', type(e).__name__, str(e)[:150], flush=True)
