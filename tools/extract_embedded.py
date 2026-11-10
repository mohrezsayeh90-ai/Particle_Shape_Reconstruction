"""Export packaged source, tensor checkpoints, cached results and notebook figures.

Only parses literals and container formats. Never executes notebook code or
unpickles checkpoints. Existing files must match; conflicting files are refused.
"""
from pathlib import Path
import ast,base64,hashlib,io,json,re,stat,zipfile,zlib
ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing artifact differs: '+str(path))
    else:path.write_bytes(data)

def main():
    records=[]
    for folder in sorted((ROOT/'evidence').iterdir()):
        notebooks=list(folder.glob('*.ipynb'))
        if not notebooks:continue
        nb=json.loads(notebooks[0].read_text());dest=ROOT/'assets'/folder.name
        seen=set();artifacts=[];archives=0;jsons=0
        for ci,cell in enumerate(nb.get('cells',[])):
            if cell.get('cell_type')=='code':
                try:tree=ast.parse(''.join(cell.get('source',[])))
                except SyntaxError:tree=None
                if tree:
                    labels={}
                    for node in ast.walk(tree):
                        if isinstance(node,ast.Assign):
                            label='_'.join(getattr(t,'id','') for t in node.targets).strip('_')
                            for v in ast.walk(node.value):
                                if isinstance(v,ast.Constant) and isinstance(v.value,str):labels[id(v)]=label
                    for node in ast.walk(tree):
                        if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
                        s=node.value
                        if len(s)<1000:continue
                        label=re.sub(r'[^a-zA-Z0-9_-]+','_',labels.get(id(node),'literal')).strip('_').lower() or 'literal'
                        if s.startswith('UEsDB'):
                            raw=base64.b64decode(s,validate=True);sha=hashlib.sha256(raw).hexdigest()
                            if sha in seen:continue
                            seen.add(sha);archives+=1
                            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                                names=archive.namelist()
                                is_torch=any(n.endswith('/data.pkl') for n in names)
                                if is_torch:
                                    path=dest/'checkpoints'/f'{label}_{archives:02d}.pt';write(path,raw)
                                    artifacts.append({'kind':'tensor_checkpoint','path':str(path.relative_to(ROOT)),'sha256':sha})
                                else:
                                    base=dest/'packages'/f'package_{archives:02d}'
                                    if sum(i.file_size for i in archive.infolist())>300_000_000:raise ValueError('Unexpectedly large embedded archive')
                                    for entry in archive.infolist():
                                        if entry.is_dir():continue
                                        path=(base/entry.filename).resolve()
                                        if not path.is_relative_to(base.resolve()) or '.git' in Path(entry.filename).parts:raise ValueError('Unsafe embedded path')
                                        if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Embedded symlink refused')
                                        write(path,archive.read(entry))
                                    artifacts.append({'kind':'source_and_weights_package','path':str(base.relative_to(ROOT)),'archive_sha256':sha,'members':names})
                        elif s.startswith('eN'):
                            try:raw=zlib.decompress(base64.b64decode(s,validate=True));obj=json.loads(raw)
                            except (ValueError,zlib.error):continue
                            sha=hashlib.sha256(raw).hexdigest()
                            if sha in seen:continue
                            seen.add(sha);jsons+=1
                            path=dest/'cached_results'/f'{label}_{jsons:02d}.json';write(path,json.dumps(obj,ensure_ascii=False,indent=1).encode())
                            artifacts.append({'kind':'cached_results','path':str(path.relative_to(ROOT)),'decoded_sha256':sha})
            for oi,out in enumerate(cell.get('outputs',[])):
                data=out.get('data',{})
                for mime,ext in [('image/png','png'),('image/jpeg','jpg'),('image/gif','gif'),('application/pdf','pdf')]:
                    if mime in data:
                        value=data[mime];value=''.join(value) if isinstance(value,list) else value
                        path=dest/'notebook_figures'/f'cell_{ci:03d}_output_{oi:02d}.{ext}'
                        write(path,base64.b64decode(value));artifacts.append({'kind':'embedded_figure','path':str(path.relative_to(ROOT))})
                if 'text/html' in data:
                    value=data['text/html'];value=''.join(value) if isinstance(value,list) else value
                    path=dest/'notebook_figures'/f'cell_{ci:03d}_output_{oi:02d}.html'
                    write(path,value.encode());artifacts.append({'kind':'embedded_html','path':str(path.relative_to(ROOT))})
                    for mi,match in enumerate(re.finditer(r'data:(image/(?:gif|png|jpeg)|video/mp4);base64,([A-Za-z0-9+/=\s]+)',value)):
                        ext={'image/gif':'gif','image/png':'png','image/jpeg':'jpg','video/mp4':'mp4'}[match[1]]
                        media=dest/'notebook_figures'/f'cell_{ci:03d}_output_{oi:02d}_media_{mi:02d}.{ext}'
                        write(media,base64.b64decode(match[2]));artifacts.append({'kind':'embedded_animation','path':str(media.relative_to(ROOT))})
        record={'notebook':folder.name,'source':str(notebooks[0].relative_to(ROOT)),'artifacts':artifacts}
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'extraction.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        records.append(record);print(folder.name,':',len(artifacts),'embedded artifacts exported')
    (ROOT/'assets/INDEX.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
