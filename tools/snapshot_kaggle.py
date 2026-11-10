"""Read-only, append-only capture; pin a version when the listing supplies it."""
import os
from pathlib import Path
import argparse,datetime,hashlib,json,re

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('KAGGLE_CONFIG_DIR',str(ROOT/'.local/kaggle'))
def scrub(s):
    return re.sub(r'KGAT_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+','[REDACTED]',s)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',action='append',help='owner/slug; defaults to all catalog entries')
    parser.add_argument('--download-outputs',action='store_true',help='Also fetch original output files, requiring CDN network access')
    args=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api=KaggleApi();api.authenticate()
    refs=args.ref or [x['ref'] for x in json.loads((ROOT/'catalog.json').read_text())]
    metadata={x.ref:x for x in api.kernels_list(mine=True,page_size=100,sort_by='dateRun')}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest=ROOT/'snapshots'/stamp;dest.mkdir(parents=True,exist_ok=False)
    records=[]
    for ref in refs:
        item=metadata.get(ref)
        if item is None:raise ValueError('Ref is outside the latest authenticated 100-notebook listing: '+ref)
        version=item.current_version_number;pin=ref+'/'+str(version) if version else ref
        folder=dest/ref.split('/')[-1];folder.mkdir()
        record={'ref':ref,'version':version,'version_pinned':bool(version),'source_endpoint':pin,'captured_utc':stamp,'errors':{}}
        try:
            api.kernels_pull(pin,str(folder),metadata=True,quiet=True)
            for p in folder.iterdir():
                if p.suffix in ['.py','.ipynb','.json']:
                    raw=p.read_bytes();record.setdefault('download_sha256',{})[p.name]=hashlib.sha256(raw).hexdigest()
                    text=raw.decode();safe=scrub(text)
                    if safe!=text:p.write_text(safe);record['credentials_redacted']=True
        except Exception as e:record['errors']['source']=type(e).__name__
        try:record['status']=api.kernels_status(pin).to_dict()
        except Exception as e:record['errors']['status']={'type':type(e).__name__,'http':getattr(getattr(e,'response',None),'status_code',None)}
        names=[];token=None
        try:
            while True:
                req=ApiListKernelSessionOutputRequest();req.user_name=ref.split('/')[0];req.kernel_slug=ref.split('/')[1];req.page_size=100
                if version:req.version_label=api._kernel_version_label(str(version))
                if token:req.page_token=token
                with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(req)
                names.extend(f.file_name for f in response.files or [])
                if response.log:
                    try:log=''.join(x.get('data','') for x in json.loads(response.log))
                    except ValueError:log=response.log
                    (folder/'run-readable.log').write_text(scrub(log))
                token=response.next_page_token
                if not token:break
            record['output_inventory']=names
        except Exception as e:record['errors']['inventory']={'type':type(e).__name__,'http':getattr(getattr(e,'response',None),'status_code',None)}
        if args.download_outputs:
            try:api.kernels_output(pin,str(folder/'original_outputs'),quiet=True,page_size=100)
            except Exception as e:record['errors']['output_download']=type(e).__name__
        record['files']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
        (folder/'capture.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        records.append(record)
        print(ref,'version',version,'capture errors:',record['errors'],flush=True)
    (dest/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('Snapshot:',dest)
if __name__=='__main__':main()
