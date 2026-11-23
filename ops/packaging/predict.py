"""Linux CUDA candidate. Uses mounted BTC SDK, never a baked TRAIN SDK."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import platform
import site
import sys
import time

CODE=Path(__file__).resolve().parent
REPO=next((p for p in [CODE,*CODE.parents] if (p/'ops/src').is_dir()),CODE)
sys.path[:0]=[str(REPO/'ops/src'),str(REPO/'llm/baseline_50_2')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
from lifemate.runtime.configuration import strict_json
from lifemate.runtime.container_io import split_root,input_rows,timed_submission,check_timed_output
from lifemate.runtime.smoke_inputs import digest,manifest


class Predictor:
    def __init__(self,data_root='/data',assets='/code/private_assets',runtime='/code/linux_runtime',result='/result',split=None,run_label='predict'):
        from baseline50.asr import MediumASR
        from baseline50.backend import LocalServer
        from lifemate.adapters.baseline50 import Baseline50Policy,Baseline50System
        from lifemate.runtime.contracts import RuntimeConfig
        started=time.perf_counter()
        if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
            raise ValueError('Candidate requires Linux AMD64/CUDA')
        self.root,self.split,self.directory=split_root(data_root,split)
        self.result=Path(result);self.result.mkdir(parents=True,exist_ok=True)
        self.stack=ExitStack()
        assets=Path(assets);runtime=Path(runtime);spec=manifest()
        for relative,info in spec['files'].items():
            if relative.startswith('models/') or relative in ('private_data/split.json','private_data/retrieval_index.json'):
                p=assets/relative
                if p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
                    raise ValueError('Model/private assets differ from frozen release')
        receipt=strict_json((runtime/'runtime_manifest.json').read_text())
        if (receipt['release']!='b11800' or receipt['platform']!='linux-amd64-cuda12.8'
                or receipt['server']!='llama-server' or 'llama-server' not in receipt['files']
                or receipt['archive_sha256']!='e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7'):
            raise ValueError('Wrong candidate runtime variant')
        for relative,info in receipt['files'].items():
            p=(runtime/relative).resolve()
            if not p.is_relative_to(runtime.resolve()) or p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:
                raise ValueError('Linux runtime failed checksum')
        executable=runtime/receipt['server']
        with executable.open('rb') as stream:header=stream.read(4)
        if header!=b'\x7fELF':raise ValueError('Expected Linux ELF server')
        if any(n=='zaic_tools' or n.startswith('zaic_tools.') for n in sys.modules):
            raise ValueError('Fresh process required for mounted SDK')
        sys.path.insert(0,str(self.directory));sdk=importlib.import_module('zaic_tools')
        if not Path(sdk.__file__).resolve().is_relative_to((self.directory/'zaic_tools').resolve()):
            raise ValueError('SDK must come from mounted split')
        catalog=strict_json((self.directory/(self.split+'_tools_catalog.json')).read_text())
        tools={t['name']:t for app in catalog['apps'] for t in app['tools']}
        policy=Baseline50Policy(assets/'private_data')
        policy.identity['assets_verified_by_bridge']=True
        # The CUDA12.8 server needs CUDA12 libraries supplied by torch's pinned
        # NVIDIA wheels, while the container base follows BTC CUDA13.2 guidance.
        library_dirs=[str(p) for base in site.getsitepackages() for p in Path(base).glob('nvidia/*/lib')]
        os.environ['LD_LIBRARY_PATH']=':'.join([str(runtime.resolve()),*library_dirs,os.environ.get('LD_LIBRARY_PATH','')])
        asr=MediumASR(assets/'models/asr',self.root,self.split,spec['asr_vocab_sha256'])
        try:
            backend=self.stack.enter_context(LocalServer(assets/spec['qwen_file'],executable,self.result/(run_label+'_server')))
            self.system=Baseline50System(catalog=tools,policy=policy,asr=asr,transport=backend,
                environment_factory=lambda sample:sdk.ToolEnv(self.split,sample.sample_id,root=self.root),
                runtime=RuntimeConfig(),provenance={'mode':'linux_cuda_candidate','live_model':True,
                  'runtime_parity_verified':False,'sdk_source':'mounted_BTC_split','official_score_claimed':False})
        except BaseException:
            self.stack.close();raise
        print(json.dumps({'model_load_ms':int((time.perf_counter()-started)*1000),'runtime_parity_verified':False}),flush=True)

    def infer(self,prepared,started=None):
        from lifemate.runtime.contracts import Sample
        started=time.perf_counter() if started is None else started
        original=self.system.runtime
        remaining=original.sample_deadline_seconds-(time.perf_counter()-started)
        self.system.runtime=replace(original,sample_deadline_seconds=max(.001,remaining))
        try:result=self.system.run(Sample(prepared['id'],{k:v for k,v in prepared.items() if k!='id'}))
        finally:self.system.runtime=original
        ms=int((time.perf_counter()-started)*1000)
        if ms>30000:
            result.diagnostics.update(status='timeout',failure=True)
        print(json.dumps({'id':prepared['id'],'predict_ms':ms,'status':result.diagnostics['status']}),flush=True)
        return timed_submission(result.submission,ms),result.diagnostics

    def close(self):self.stack.close()


def write_predictions(predictor,filename='submission.jsonl'):
    output=predictor.result/filename
    if output.exists():raise FileExistsError('Preserve existing submission; no automatic rerun')
    ids=[]
    with output.open('x',encoding='utf-8') as stream, (predictor.result/(filename+'.diagnostics.jsonl')).open('x',encoding='utf-8') as log:
        iterator=iter(input_rows(predictor.directory,predictor.split))
        while True:
            started=time.perf_counter()
            try:prepared=next(iterator)
            except StopIteration:break
            record,diagnostics=predictor.infer(prepared,started=started)
            ids.append(prepared['id'])
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            log.write(json.dumps(diagnostics,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
    return check_timed_output(output,ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/data');p.add_argument('--output-dir',default='/result')
    p.add_argument('--assets',default='/code/private_assets');p.add_argument('--runtime',default='/code/linux_runtime')
    p.add_argument('--split');p.add_argument('--check-input',action='store_true')
    args=p.parse_args()
    if args.check_input:
        _,split,directory=split_root(args.data_root,args.split)
        rows=list(input_rows(directory,split))
        print(json.dumps({'input_valid':True,'split':split,'samples':len(rows),'model_inference_run':False}))
        return
    if Path(args.output_dir,'submission.jsonl').exists():raise FileExistsError('Existing output')
    predictor=Predictor(args.data_root,args.assets,args.runtime,args.output_dir,args.split)
    try:print(json.dumps(write_predictions(predictor)))
    finally:predictor.close()


if __name__=='__main__':main()
