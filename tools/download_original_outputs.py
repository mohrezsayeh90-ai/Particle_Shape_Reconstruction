"""Archive all API-listed outputs, preserving original paths and recording gaps.

Read-only on Kaggle. Credentials stay in environment bindings. Signed download
URLs are used only in memory and are never included in reports or errors.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('KAGGLE_CONFIG_DIR',str(ROOT/'.local/kaggle'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only',action='store_true')
    args=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    api=KaggleApi();api.authenticate()
    base=ROOT/'kaggle_outputs';base.mkdir(exist_ok=True)
    summary=[];blocked_hosts=set()
    for item in json.loads((ROOT/'catalog.json').read_text()):
        ref=item['ref'];owner,slug=ref.split('/');folder=base/slug;folder.mkdir(exist_ok=True)
        manifest_path=folder/'download_manifest.json'
        previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        known={x['name']:x for x in previous.get('files',[])}
        record={'ref':ref,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[]}
        try:
            token=None;files=[]
            while True:
                req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=100
                if token:req.page_token=token
                with api.build_kaggle_client() as client:r=client.kernels.kernels_api_client.list_kernel_session_output(req)
                files.extend(r.files or [])
                if r.log:
                    try:log=''.join(x.get('data','') for x in json.loads(r.log))
                    except ValueError:log=r.log
                    log=re.sub(r'KGAT_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+','[REDACTED]',log)
                    (folder/'session.log').write_text(log)
                token=r.next_page_token
                if not token:break
        except Exception as e:
            record['listing_error']={'type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}
            files=[]
        for item in files:
            name=item.file_name;host=urlsplit(item.url).hostname
            path=(folder/'files'/name).resolve()
            if not path.is_relative_to((folder/'files').resolve()):raise ValueError('Unsafe output filename')
            entry={'name':name,'download_host':host,'state':'not_downloaded'}
            prior=known.get(name,{})
            if path.is_file() and prior.get('sha256')==sha(path):
                entry.update(state='downloaded',sha256=prior['sha256'],size_bytes=path.stat().st_size)
            elif path.exists():
                entry.update(state='failed',reason='existing_file_not_verified_refused_to_overwrite')
            elif args.inventory_only:entry['reason']='inventory_only'
            elif host in blocked_hosts:entry.update(state='blocked',reason='proxy_denied_host')
            else:
                temp=None
                try:
                    # No Authorization header is sent to the file/CDN host.
                    with requests.get(item.url,stream=True,timeout=(20,120)) as response:
                        response.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True)
                        temp=path.with_name(path.name+'.part');count=0
                        with temp.open('wb') as f:
                            for data in response.iter_content(1024*1024):f.write(data);count+=len(data)
                        expected=response.headers.get('Content-Length')
                        if expected and not response.headers.get('Content-Encoding') and count!=int(expected):raise ValueError('Incomplete file transfer')
                        os.replace(temp,path);entry.update(state='downloaded',sha256=sha(path),size_bytes=count)
                except requests.exceptions.ProxyError:
                    blocked_hosts.add(host);entry.update(state='blocked',reason='proxy_connection_denied')
                except Exception as e:
                    entry.update(state='failed',reason=type(e).__name__)
                finally:
                    if temp is not None and temp.exists():temp.unlink()
            record['files'].append(entry)
        record['listed_files']=len(files)
        record['downloaded_files']=sum(x['state']=='downloaded' for x in record['files'])
        record['complete']=bool(files) and record['downloaded_files']==len(files)
        manifest_path.write_text(json.dumps(record,ensure_ascii=False,indent=2));summary.append(record)
        print(ref,':',record['downloaded_files'],'/',len(files),'files; listing error:',record.get('listing_error'),flush=True)
    (base/'INDEX.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Blocked file hosts:',sorted(blocked_hosts))
    return 0 if args.inventory_only or all(r['complete'] for r in summary) else 3
if __name__=='__main__':raise SystemExit(main())
