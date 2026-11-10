"""Explicit save-and-run submission. Credentials come only from Kaggle config/env.

Optional dependency: pip install kaggle six
KAGGLE_API_TOKEN must be configured outside this repository.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_notebook import build

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kernel',default='mohrezsayeh/master10-magnetic-function-deeponet')
    p.add_argument('--kernel-id',type=int,default=143255479)
    p.add_argument('--folder',default='work/operator_submission')
    p.add_argument('--pretrained',action='store_true')
    p.add_argument('--cpu',action='store_true',help='Revalidate bundled weights on CPU; requires --pretrained')
    p.add_argument('--steps',type=int,default=24000)
    a=p.parse_args()
    if a.cpu and not a.pretrained:p.error('--cpu requires --pretrained; original training requires a GPU')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    notebook=build(Path(a.folder)/'magnetic_operator.ipynb',a.pretrained,a.steps)
    api=KaggleApi();api.authenticate()
    req=ApiSaveKernelRequest();req.id=a.kernel_id;req.slug=a.kernel
    req.new_title='Master10 Magnetic Function DeepONet'
    req.text=notebook.read_text();req.language='python';req.kernel_type='notebook'
    req.is_private=True;req.enable_gpu=not a.cpu;req.enable_tpu=False;req.enable_internet=False
    if not a.cpu:req.machine_shape='NvidiaTeslaT4'
    req.session_timeout_seconds=3600
    # Some API/SDK combinations accept a save while leaving the execution type
    # unspecified. Explicitly request a full run and verify its actual status.
    req.kernel_execution_type=KernelExecutionType.SAVE_AND_RUN_ALL
    with api.build_kaggle_client() as client:
        result=client.kernels.kernels_api_client.save_kernel(req)
    if result.error:raise RuntimeError(result.error)
    (Path(a.folder)/'submission.json').write_text(str(result))
    print(result)

if __name__=='__main__':main()
