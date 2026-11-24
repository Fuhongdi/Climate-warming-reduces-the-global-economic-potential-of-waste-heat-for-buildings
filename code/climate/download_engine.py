import os
#!/usr/bin/env python3
"""Download the user-approved, immutable ISIMIP daily manifest; resume safely."""
from pathlib import Path
from collections import deque
import concurrent.futures, datetime, fcntl, hashlib, heapq, json, os, re, shutil
import signal, subprocess, sys, threading, time

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
WORK = ROOT / 'work/climate_download_2026-11-12'
MANIFEST = WORK / 'download_manifest.json'
RESERVE = 100 * 1024**3
MAX_WORKERS = 16
STOP = threading.Event()
GUARD = threading.RLock()
ACTIVE = {}
PROCESSES = {}
VERIFIED = {}
FAILED = {}
STARTED = time.time()
INITIAL = 0
ROWS = []
RETRIES = {}
QUEUED = {}
MANIFEST_SHA256 = 'e655ebc4dcba3994f50d7885b8d7907276e5d507622e3c1f0395285704120288'
PROGRESS_DELAY = 5
RETRY_BASE = 30
RETRY_MAX = 900
MAX_CHECKSUM_REDOWNLOADS = 2
TRANSIENT_CURL = {5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92, 95, 96}
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}

class RetryTransfer(Exception):
    def __init__(self, reason, delay):
        super().__init__(reason)
        self.delay = delay

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)
def event(kind, **fields):
    with GUARD:
        with (WORK/'events.jsonl').open('a') as out:
            out.write(json.dumps({'utc':utc(),'event':kind,**fields}, ensure_ascii=False)+'\n')
        print(json.dumps({'utc':utc(),'event':kind,**fields}, ensure_ascii=False), flush=True)
def target(row):
    p = ROOT / row['target']
    if not p.resolve().is_relative_to(ROOT): raise RuntimeError('Target outside project')
    q = p
    while q != ROOT:
        if q.is_symlink(): raise RuntimeError('Symlink in target path: '+str(q))
        q = q.parent
    return p
def partial(p): return p.with_name(p.name + '.part')
def safe_size(p):
    try: return p.stat().st_size
    except FileNotFoundError: return 0
def receipt(row): return WORK/'verified'/(row['id']+'.json')
def save_retries():
    atomic(WORK/'retry_state.json', {'manifest_sha256':MANIFEST_SHA256,'records':RETRIES})
def sha512(path):
    h = hashlib.sha512()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def accepted_receipt(row, p):
    q = receipt(row)
    if not p.is_file() or not q.is_file(): return None
    try:
        v=json.loads(q.read_text()); st=p.stat()
        if (v['expected_sha512']==row['checksum'] and v['bytes']==row['size']
            and st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns']): return v
    except (ValueError,KeyError,OSError): pass
    return None
def snapshot(state='running'):
    with GUARD:
        active=[dict(v) for v in ACTIVE.values()]
        for a in active:
            p=Path(a['partial_path']); a['downloaded_bytes']=safe_size(p)
        downloaded=0
        for row in ROWS:
            p=target(row); q=partial(p)
            downloaded+=min(safe_size(p) or safe_size(q),row['size'])
        elapsed=max(time.time()-STARTED,1)
        rate=max(downloaded-INITIAL,0)/elapsed
        total=sum(r['size'] for r in ROWS)
        x={'utc':utc(),'state':state,'pid':os.getpid(),'started_epoch':STARTED,
           'total_files':len(ROWS),'total_bytes':total,'verified_files':len(VERIFIED),
           'verified_bytes':sum(v['bytes'] for v in VERIFIED.values()),
           'downloaded_bytes_including_partials':downloaded,'failed_files':len(FAILED),
           'queued_files':len(QUEUED),'retrying_files':sum(1 for k in QUEUED if RETRIES.get(k,{}).get('attempts',0)),
           'failures':FAILED,'active':active,'workers':MAX_WORKERS,'transfer_proxy':'http://127.0.0.1:7897',
           'session_average_bytes_per_second':rate,
           'eta_seconds_at_session_average':(total-downloaded)/rate if rate else None,
           'free_disk_bytes':shutil.disk_usage(ROOT).free,'reserved_free_bytes':RESERVE,
           'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
        atomic(WORK/'status.json',x)
        save_retries()
        return x
def stop_handler(sig, frame):
    STOP.set()
    with GUARD:
        for process in list(PROCESSES.values()):
            if process.poll() is None:
                try: process.terminate()
                except ProcessLookupError: pass
    event('stop_requested', signal=sig)
def verify(row, p):
    if p.stat().st_size != row['size']: raise ValueError('Size mismatch')
    if sha512(p) != row['checksum']: raise ValueError('SHA512 mismatch')
    return {'publisher_sha512_verified':True,'netcdf_header_checked':False,
            'note':'NetCDF coordinate/value checks belong to later data preparation; this download verifies the exact publisher file.'}

def retry_transfer(row, reason, progress):
    with GUARD:
        stat=RETRIES[row['id']]
        stat['no_progress']=0 if progress else stat.get('no_progress',0)+1
        delay=PROGRESS_DELAY if progress else min(RETRY_MAX, RETRY_BASE * 2**min(stat['no_progress']-1,10))
        stat.update(last_reason=reason,next_retry_epoch=time.time()+delay)
    raise RetryTransfer(reason,delay)

def run_curl(row, q):
    # One transfer per worker visit; the scheduler owns retries and fairness.
    # --disable prevents ~/.curlrc from weakening TLS or changing the proxy.
    cmd=['curl','--disable','--noproxy','','--proxy','http://127.0.0.1:7897','--proto','=https','--proto-redir','=https',
         '--location','--fail','--silent','--show-error','--connect-timeout','30',
         '--speed-limit','1024','--speed-time','180','--retry','0',
         '--continue-at','-','--output',str(q),'--write-out','%{http_code}',row['file_url']]
    with (WORK/'transfer_logs'/(row['id']+'.log')).open('a') as log:
        with GUARD:
            if STOP.is_set(): return None,None
            process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log)
            PROCESSES[row['id']]=process
        try:
            output,_=process.communicate()
            raw=output.decode('ascii',errors='replace').strip()
            http=int(raw) if raw.isdigit() else None
            return process.returncode,http
        finally:
            with GUARD: PROCESSES.pop(row['id'],None)

def one(row):
    if STOP.is_set(): return
    try:
        p=target(row); p.parent.mkdir(parents=True,exist_ok=True); q=partial(p)
        old=accepted_receipt(row,p)
        if old:
            with GUARD:
                VERIFIED[row['id']]=old; FAILED.pop(row['id'],None); RETRIES.pop(row['id'],None)
            return
        with GUARD:
            stat=RETRIES.setdefault(row['id'],{})
            stat['attempts']=stat.get('attempts',0)+1
            ACTIVE[row['id']]={'id':row['id'],'name':row['name'],'group':row['group'],
                'expected_bytes':row['size'],'partial_path':str(q),'phase':'preparing',
                'attempt':stat['attempts'],'started_utc':utc()}
        if q.is_symlink(): raise RuntimeError('Symlink partial file; refusing to write')
        if p.exists():
            header=verify(row,p)
        else:
            if q.exists() and q.stat().st_size>row['size']:
                raise RuntimeError('Existing partial larger than expected; retained for review')
            before=safe_size(q); need=row['size']-before
            if shutil.disk_usage(ROOT).free < RESERVE+need:
                raise RuntimeError('Insufficient free disk; partial retained')
            rc=http=None
            if before<row['size']:
                with GUARD: ACTIVE[row['id']]['phase']='downloading'
                rc,http=run_curl(row,q)
                if STOP.is_set(): return
                after=safe_size(q)
                if after>row['size']: raise RuntimeError('Partial larger than manifest size; retained for review')
                # Even an interrupted response can have delivered the complete file.
                if after<row['size']:
                    reason='curl_exit=%s http_status=%s progress_bytes=%s'%(rc,http,after-before)
                    if rc in TRANSIENT_CURL or (rc==22 and http in TRANSIENT_HTTP) or rc==0:
                        retry_transfer(row,reason,after>before)
                    raise RuntimeError('Non-retryable transfer error; '+reason+'; partial retained')
            with GUARD: ACTIVE[row['id']]['phase']='verifying'
            try:
                header=verify(row,q)
            except ValueError as e:
                with GUARD:
                    stat=RETRIES[row['id']]
                    stat['verification_failures']=stat.get('verification_failures',0)+1
                    failures=stat['verification_failures']
                if failures>MAX_CHECKSUM_REDOWNLOADS:
                    raise RuntimeError('Repeated '+str(e)+'; partial retained for review')
                # Preserve corrupt bytes for review; never delete or overwrite final files.
                bad=q.with_name(q.name+'.bad-sha512.'+str(time.time_ns()))
                os.replace(q,bad)
                event('verification_quarantined',id=row['id'],failures=failures,retained_path=str(bad))
                retry_transfer(row,str(e)+'; corrupt partial quarantined',False)
            os.replace(q,p)
        st=p.stat()
        rec={'verified_utc':utc(),'id':row['id'],'path':str(p),'bytes':st.st_size,
             'mtime_ns':st.st_mtime_ns,'expected_sha512':row['checksum'],'version':row['version'],
             'source_url':row['file_url'],'header':header}
        atomic(receipt(row),rec)
        with GUARD: VERIFIED[row['id']]=rec; FAILED.pop(row['id'],None); RETRIES.pop(row['id'],None)
        event('verified',id=row['id'],name=row['name'],bytes=row['size'])
    except RetryTransfer as e:
        event('transfer_requeued',id=row['id'],name=row['name'],delay_seconds=e.delay,reason=str(e))
        return e.delay
    except Exception as e:
        with GUARD:
            FAILED[row['id']]={'name':row['name'],'error':str(e),'utc':utc()}
            RETRIES.setdefault(row['id'],{})['terminal_error']=FAILED[row['id']]
            save_retries()
        event('file_failed',id=row['id'],name=row['name'],reason=str(e))
    finally:
        with GUARD: ACTIVE.pop(row['id'],None); PROCESSES.pop(row['id'],None)

def run_queue(rows):
    ready=[]; pending=deque(); sequence=0; futures={}
    for row in rows:
        old=accepted_receipt(row,target(row))
        if old:
            VERIFIED[row['id']]=old; RETRIES.pop(row['id'],None)
            continue
        state=RETRIES.get(row['id'],{})
        if state.get('terminal_error'):
            FAILED[row['id']]=state['terminal_error']; continue
        due=state.get('next_retry_epoch',0)
        if state.get('attempts',0):
            heapq.heappush(ready,(due,sequence,row)); sequence+=1
        else:
            pending.append(row)
        QUEUED[row['id']]=due
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while (ready or pending or futures) and not (STOP.is_set() and not futures):
            while len(futures)<MAX_WORKERS and not STOP.is_set():
                # Eligible retries take the next slot; hundreds of untouched files
                # must not postpone a partly downloaded file's retry for hours.
                if ready and ready[0][0]<=time.time():
                    _,_,row=heapq.heappop(ready)
                elif pending:
                    row=pending.popleft()
                else:
                    break
                QUEUED.pop(row['id'],None)
                futures[pool.submit(one,row)]=row
            snapshot('stopping' if STOP.is_set() else 'running')
            timeout=10
            if ready and len(futures)<MAX_WORKERS and not STOP.is_set():
                timeout=max(0.01,min(timeout,ready[0][0]-time.time()))
            if not futures:
                if STOP.is_set(): break
                STOP.wait(timeout); continue
            done,_=concurrent.futures.wait(futures,timeout=timeout,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                row=futures.pop(future); delay=future.result()
                if delay is not None and not STOP.is_set():
                    due=time.time()+delay
                    with GUARD: RETRIES[row['id']]['next_retry_epoch']=due
                    heapq.heappush(ready,(due,sequence,row)); sequence+=1
                    QUEUED[row['id']]=due

def main():
    global ROWS,INITIAL
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'download.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print('An existing downloader holds the lock'); return 0
    atomic(WORK/'pid.json',{'pid':os.getpid(),'started_utc':utc()})
    for n in ['verified','transfer_logs']: (WORK/n).mkdir(exist_ok=True)
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()==MANIFEST_SHA256
    d=json.loads(MANIFEST.read_text()); ROWS=d['files']
    assert len(ROWS)==460 and sum(r['size'] for r in ROWS)==1098122464742
    assert len({r['id'] for r in ROWS})==460
    saved=WORK/'retry_state.json'
    if saved.exists():
        prior=json.loads(saved.read_text())
        assert prior['manifest_sha256']==MANIFEST_SHA256
        RETRIES.update(prior['records'])
    for row in ROWS:
        assert re.fullmatch(r'[A-Za-z0-9_-]+',row['id']) and re.fullmatch(r'[0-9a-f]{128}',row['checksum'])
        assert row['file_url'].startswith('https://') and row['size']>0
        p=target(row); q=partial(p)
        INITIAL+=min(p.stat().st_size,row['size']) if p.exists() else min(q.stat().st_size,row['size']) if q.exists() else 0
    remaining=sum(r['size'] for r in ROWS)-INITIAL
    if shutil.disk_usage(ROOT).free<remaining+RESERVE:
        event('insufficient_space_at_start',remaining=remaining);snapshot('blocked_disk_space');return 2
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    event('started',files=len(ROWS),bytes=sum(r['size'] for r in ROWS),workers=MAX_WORKERS)
    run_queue(ROWS)
    state='stopped' if STOP.is_set() else 'completed' if len(VERIFIED)==len(ROWS) else 'needs_attention'
    snapshot(state);event(state,verified_files=len(VERIFIED),failed_files=len(FAILED))
    return 0 if state=='completed' else 1
if __name__=='__main__': sys.exit(main())
