"""Read-only MedMod/TRCG cohort audit: AP, temporal ICU validity, most-recent AP.

Uses existing paired_{train,val,test}.csv and local MIMIC-CXR metadata.
Writes only a log report; does NOT change the cohort or any checkpoint.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'logs';LOGS.mkdir(exist_ok=True,parents=True)
parser=argparse.ArgumentParser()
parser.add_argument('--paired-root',default=r'C:\MedModWork\paired')
parser.add_argument('--cxr-root',default=r'S:\MedModData\mimic-cxr-jpg\2.1.0')
args=parser.parse_args()
PAIRED=Path(args.paired_root)
CXR=Path(args.cxr_root)
metadata_path=CXR/'mimic-cxr-2.0.0-metadata.csv'
if not metadata_path.exists():
    metadata_path=CXR/'mimic-cxr-2.0.0-metadata.csv.gz'
if not metadata_path.exists():
    raise FileNotFoundError('Cannot find CXR metadata CSV(.gz) at '+str(CXR))

parts=[]
for split in ['train','val','test']:
    p=PAIRED/f'paired_{split}.csv'
    if not p.exists():raise FileNotFoundError(p)
    sub=pd.read_csv(p)
    sub['split_name']=split
    parts.append(sub)
paired=pd.concat(parts,ignore_index=True)
required={'subject_id','stay_id','dicom_id','intime','outtime','StudyDateTime','ViewPosition'}
if not required.issubset(paired.columns):
    raise KeyError(f'Missing paired columns: {sorted(required-set(paired.columns))}')
paired['intime']=pd.to_datetime(paired['intime'],errors='coerce')
paired['outtime']=pd.to_datetime(paired['outtime'],errors='coerce')
paired['StudyDateTime']=pd.to_datetime(paired['StudyDateTime'],errors='coerce')
paired['ViewPosition']=paired['ViewPosition'].astype('string').str.upper().str.strip()
paired['subject_id']=pd.to_numeric(paired['subject_id'],errors='coerce')
paired['stay_id']=pd.to_numeric(paired['stay_id'],errors='coerce')
report=[]
def line(*xs):
    s=' '.join(map(str,xs));report.append(s);print(s,flush=True)
line('COHORT AP/LATEST-IMAGE AUDIT')
line('Total selected:',len(paired))
line('Split sizes:',paired.groupby('split_name').size().to_dict())
line('Selected ViewPosition:',paired['ViewPosition'].value_counts(dropna=False).to_dict())
line('Distinct subjects:',paired['subject_id'].nunique(),'stays:',paired['stay_id'].nunique())
line('Subjects across splits:',int((paired.groupby('subject_id')['split_name'].nunique()>1).sum()))
line('Duplicate selected stay_id:',int(paired['stay_id'].duplicated().sum()))
end=paired[['outtime','intime']].copy()
last_allowed=pd.concat([paired['outtime'],paired['intime']+pd.Timedelta(hours=48)],axis=1).min(axis=1)
invalid_time=((paired['StudyDateTime'] < paired['intime']) |
              (paired['StudyDateTime'] > last_allowed) |
              paired['StudyDateTime'].isna() |
              paired['intime'].isna() | paired['outtime'].isna())
line('Outside ICU/48h/missing times:',int(invalid_time.sum()))
line('Selected non-AP images:',int((paired['ViewPosition']!='AP').fillna(True).sum()))

m=pd.read_csv(metadata_path,usecols=['subject_id','dicom_id','StudyDate','StudyTime','ViewPosition'])
m['subject_id']=pd.to_numeric(m['subject_id'],errors='coerce')
m['ViewPosition']=m['ViewPosition'].astype('string').str.upper().str.strip()
m=m[m['ViewPosition']=='AP'].copy()
# Metadata uses YYYYMMDD and HHMMSS; seconds are sufficient for selection.
date=m['StudyDate'].astype('Int64').astype('string').str.zfill(8)
time=m['StudyTime'].astype('string').str.split('.',n=1).str[0].str.zfill(6)
m['cxr_dt']=pd.to_datetime(date+time,format='%Y%m%d%H%M%S',errors='coerce')
line('AP metadata records:',len(m),'invalid metadata timestamps:',int(m['cxr_dt'].isna().sum()))
# Only scan images belonging to subjects in this study.
m=m[m.subject_id.isin(paired.subject_id)]
joined=paired[['subject_id','stay_id','intime','outtime','dicom_id','StudyDateTime']].merge(
    m[['subject_id','dicom_id','cxr_dt']],on='subject_id',how='left',suffixes=('_chosen','_candidate'))
cutoff=pd.concat([joined['outtime'],joined['intime']+pd.Timedelta(hours=48)],axis=1).min(axis=1)
valid=joined[(joined['cxr_dt']>=joined['intime']) & (joined['cxr_dt']<=cutoff)]
latest=valid.groupby('stay_id')['cxr_dt'].max()
selected=paired.set_index('stay_id')
last=selected.index.to_series().map(latest)
missing=last.isna()
# Ignore subsecond differences; metadata time usually has second precision.
not_latest=(~missing) & ((selected['StudyDateTime']-last).abs()>pd.Timedelta(seconds=1))
chosen_found = valid.loc[valid['dicom_id_chosen']==valid['dicom_id_candidate'],'stay_id'].unique()
missing_selected = ~selected.index.isin(chosen_found)
line('Selected DICOM absent from eligible AP metadata:',int(missing_selected.sum()))
line('Stays with NO candidate AP CXR in ICU and <=48h:',int(missing.sum()))
line('Selected image NOT most recent eligible AP:',int(not_latest.sum()))
if not_latest.any():
    line('Example mismatched stay IDs:',str(selected.index[not_latest][:10].tolist()))
path=LOGS/'stage7_cxr_cohort_audit.txt'
path.write_text('\n'.join(report)+'\n',encoding='utf-8')
line('Report:',path)
errors=[int(invalid_time.sum()),int((paired['ViewPosition']!='AP').fillna(True).sum()),
        int(paired['stay_id'].duplicated().sum()),
        int((paired.groupby('subject_id')['split_name'].nunique()>1).sum()),
        int(missing.sum()),int(not_latest.sum()),int(missing_selected.sum())]
if any(errors):
    print('AUDIT NEEDS REVIEW. Do not modify existing models; inspect report.',flush=True)
    sys.exit(2)
print('PASS: selected image AP/ICU/48h and latest eligible AP (per metadata).',flush=True)
