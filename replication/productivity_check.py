from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import linalg, stats

root = Path(__file__).resolve().parents[1]
out = root / 'outputs'
out.mkdir(exist_ok=True)
d = pd.read_csv(root / 'data/derived/analysis_sample.csv')

def design(df, grader=False, interaction=False):
    cols = {'Intercept': np.ones(len(df)), 'treatment': df.treatment.to_numpy(float)}
    if interaction:
        cols['prior_use'] = df.prior_use.to_numpy(float)
        cols['treatment:prior_use'] = (df.treatment * df.prior_use).to_numpy(float)
    base = pd.DataFrame(cols)
    cats = pd.DataFrame({'occ_order': df.occupation.astype(str) + ':' + df.a_first.astype(str), 'incentive': df.incentive_arm.astype(str)})
    if grader:
        cats['grader'] = df.grader_id.astype(str)
    dummy = pd.get_dummies(cats, drop_first=True, dtype=float).reset_index(drop=True)
    x = pd.concat([base, dummy], axis=1)
    _, rr, piv = linalg.qr(x.to_numpy(), mode='economic', pivoting=True)
    rank = np.linalg.matrix_rank(rr)
    keep = sorted(piv[:rank])
    return x.iloc[:,keep]

def fit(df, y, grader=False, interaction=False):
    df = df.dropna(subset=[y, 'treatment'] + (['prior_use'] if interaction else [])).copy()
    xx = design(df, grader, interaction)
    x = xx.to_numpy(float)
    yy = df[y].to_numpy(float)
    b = np.linalg.lstsq(x, yy, rcond=None)[0]
    u = yy-x@b
    bread = np.linalg.inv(x.T@x)
    n,k = x.shape
    if grader:
        z = pd.DataFrame(x*u[:,None]);z['cluster']=df.prolific_pid.to_numpy()
        z=z.groupby('cluster').sum().to_numpy()
        g=len(z)
        v=g/(g-1)*(n-1)/(n-k)*bread@(z.T@z)@bread
        dof=g-1
    else:
        z=x*u[:,None]
        v=n/(n-k)*bread@(z.T@z)@bread
        dof=n-k
    rows=[]
    for i,name in enumerate(xx.columns):
        se=np.sqrt(max(v[i,i],0));t=b[i]/se if se>0 else np.nan;crit=stats.t.ppf(.975,dof)
        rows.append({'term':name,'estimate':b[i],'se':se,'lo':b[i]-crit*se,'hi':b[i]+crit*se,'p':2*stats.t.sf(abs(t),dof)})
    res=pd.DataFrame(rows)
    return {'N':n,'K':k,'df':dof,'clusters':int(df.prolific_pid.nunique()),'terms':res.astype(object).where(pd.notnull(res),None).to_dict('records')},res

results={}
results['time'],r=fit(d,'change');r.to_csv(out/'time_verified.csv',index=False)
results['time']['baseline_sd']=float(d.time1.std())
results['time']['standardized']=float(r.loc[r.term=='treatment','estimate'].iloc[0]/d.time1.std())
results['extension'],r=fit(d,'change',interaction=True);r.to_csv(out/'extension_verified.csv',index=False)
results['time_means']=d.groupby('treatment')[['time1','time2']].mean().to_dict()

x=pd.read_csv(root/'data/raw/fullsurvey.csv',encoding='latin1',low_memory=False)
x=x[(x.finished==1)&(x.consent!=2)&(x.consent_exact!=2)].copy()
x['date']=pd.to_datetime(x.startdate.str[:10])
taskdates=x.groupby('prolific_pid').date.min()
g=pd.read_csv(root/'data/raw/grading_long_complete.csv')
g=g[(g.grader_id!='043a718774c572bd8a25adbeb1bfcd5c0256ae11cecf9f9c3f925d0e52beaf89')&(g.retry!=1)].copy()
g['date']=pd.to_datetime(g.StartDate.str[:10])
g['taskdate']=g.grader_id.map(taskdates)
g=g[~(g.date>g.taskdate)].dropna(subset=['overall'])
h=g[~g.prolific_pid.str.lower().eq('chatgpt')]
a=h[['prolific_pid','essaynum','grader_id','overall']].merge(h[['prolific_pid','essaynum','grader_id','overall']],on=['prolific_pid','essaynum'],suffixes=('_self','_other'))
a=a[a.grader_id_self!=a.grader_id_other]
bad=[];slopes=[]
for gid,z in a.groupby('grader_id_self'):
    slope=np.linalg.lstsq(np.column_stack([np.ones(len(z)),z.overall_other]),z.overall_self,rcond=None)[0][1]
    slopes.append([gid,slope])
    if slope<.1: bad.append(gid)
pd.DataFrame(slopes,columns=['grader_id','slope']).to_csv(out/'grader_screening.csv',index=False)
print('Grader screen',len(slopes),len(bad))
w=pd.read_csv(root/'data/raw/grading_wide_complete.csv')
w=w[(w.grader_id!='043a718774c572bd8a25adbeb1bfcd5c0256ae11cecf9f9c3f925d0e52beaf89')&(w.retry_a!=1)&(w.retry_b!=1)]

w=w.drop(columns=['occupation'],errors='ignore').merge(d,on='prolific_pid',how='inner',validate='many_to_one')
w['overall1']=np.where(w.a_first==1,w.overall_a,w.overall_b)
w['overall2']=np.where(w.a_first==1,w.overall_b,w.overall_a)
w['quality_change']=w.overall2-w.overall1
for label,q in [('quality',w[~w.grader_id.isin(bad)].copy()),('quality_all_graders',w.copy())]:
    results[label],r=fit(q,'quality_change',grader=True)
    r.to_csv(out/(label+'_verified.csv'),index=False)
    results[label]['baseline_sd']=float(q.overall1.std())
    results[label]['standardized']=float(r.loc[r.term=='treatment','estimate'].iloc[0]/q.overall1.std())
    q[['prolific_pid','grader_id','treatment','occupation','a_first','incentive_arm','overall1','overall2','quality_change']].to_csv(out/(label+'_sample.csv'),index=False)
results['grader_screen']={'graders':len(slopes),'excluded':len(bad)}
(out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
summary=[]
for model in ['time','quality','quality_all_graders','extension']:
    result=results[model]
    for row in result['terms']:
        if row['term'] in ['treatment','treatment:prior_use']:
            entry={'model':model,'N':result['N'],'clusters':result['clusters'],**row}
            if model!='extension':
                entry['baseline_sd']=result['baseline_sd']
                entry['standardized']=row['estimate']/result['baseline_sd']
            summary.append(entry)
pd.DataFrame(summary).to_csv(out/'productivity_summary.csv',index=False)
for name in ['time','quality','quality_all_graders']:
    print(name, [r for r in results[name]['terms'] if r['term']=='treatment'])
