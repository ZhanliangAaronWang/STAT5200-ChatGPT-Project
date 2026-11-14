from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
d=pd.read_csv(ROOT/'data/derived/analysis_sample.csv')
q=pd.read_csv(ROOT/'outputs/quality_sample.csv')
qa=pd.read_csv(ROOT/'outputs/quality_all_graders_sample.csv')

def model(frame,y,controls=True,grader=False,ancova=False,interaction=False):
    need=[y,'treatment']+(['time1'] if ancova else [])+(['prior_use'] if interaction else [])
    frame=frame.dropna(subset=need).reset_index(drop=True)
    cols={'Intercept':np.ones(len(frame)),'Treatment':frame.treatment.to_numpy(float)}
    if ancova: cols['Baseline time']=frame.time1.to_numpy(float)
    if interaction:
        cols['Previous use']=frame.prior_use.to_numpy(float)
        cols['Treatment x previous use']=(frame.treatment*frame.prior_use).to_numpy(float)
    x=pd.DataFrame(cols)
    cats={}
    if controls:
        cats['Occupation x order']=frame.occupation.astype(str)+':'+frame.a_first.astype(str)
        cats['Incentive']=frame.incentive_arm.astype(str)
    if grader: cats['Evaluator']=frame.grader_id.astype(str)
    if cats: x=pd.concat([x,pd.get_dummies(pd.DataFrame(cats),drop_first=True,dtype=float)],axis=1)
    _,rr,piv=linalg.qr(x.to_numpy(),mode='economic',pivoting=True)
    x=x.iloc[:,sorted(piv[:np.linalg.matrix_rank(rr)])]
    names=list(x.columns);X=x.to_numpy();Y=frame[y].to_numpy()
    b=np.linalg.lstsq(X,Y,rcond=None)[0];u=Y-X@b
    bread=np.linalg.inv(X.T@X);n,k=X.shape;g=frame.prolific_pid.nunique()
    scores=X*u[:,None]
    if grader:
        z=pd.DataFrame(scores);z['id']=frame.prolific_pid
        scores=z.groupby('id').sum().to_numpy()
        V=g/(g-1)*(n-1)/(n-k)*bread@(scores.T@scores)@bread;df=g-1
    else:
        V=n/(n-k)*bread@(scores.T@scores)@bread;df=n-k
    def contrast(weights):
        v=np.zeros(k)
        for name,w in weights.items(): v[names.index(name)]=w
        est=float(v@b);se=float(np.sqrt(v@V@v));tcrit=stats.t.ppf(.975,df)
        return {'estimate':est,'se':se,'lo':est-tcrit*se,'hi':est+tcrit*se,'p':float(2*stats.t.sf(abs(est/se),df))}
    out={'N':n,'participants':int(g),'df':int(df),'effect':contrast({'Treatment':1})}
    if interaction:
        out['previous_use']=contrast({'Previous use':1})
        out['interaction']=contrast({'Treatment x previous use':1})
        out['previous_users_effect']=contrast({'Treatment':1,'Treatment x previous use':1})
        out['counts']=frame.groupby(['treatment','prior_use']).size().to_dict()
        out['counts']={f'{int(a)}:{int(b)}':int(n) for (a,b),n in out['counts'].items()}
    return out

res={
 'time_unadjusted':model(d,'change',controls=False),
 'time_main':model(d,'change'),
 'time_ancova':model(d.dropna(subset=['change']),'time2',ancova=True),
 'quality_main':model(q,'quality_change',grader=True),
 'quality_all':model(qa,'quality_change',grader=True),
 'extension':model(d,'change',interaction=True),
 'extension_complete_sample':model(d.dropna(subset=['prior_use']),'change')
}
for name,sd in [('time_main',d.time1.std()),('quality_main',q.overall1.std()),('quality_all',qa.overall1.std())]:
    res[name]['baseline_sd']=float(sd)
    res[name]['standardized']={k:v/float(sd) for k,v in res[name]['effect'].items() if k!='p'}
dc=d.dropna(subset=['change'])
qc=q.dropna(subset=['quality_change'])
res['time_means']=dc.groupby('treatment')[['time1','time2','change']].agg(['mean','std','count']).to_dict()
res['time_means']={f'{a}_{b}':{str(int(k)):float(v) for k,v in x.items()} for (a,b),x in res['time_means'].items()}
res['quality_means']=qc.groupby('treatment')[['overall1','overall2','quality_change']].mean().to_dict()
res['quality_means']={a:{str(int(k)):float(v) for k,v in x.items()} for a,x in res['quality_means'].items()}
base_grade=q.groupby(['prolific_pid','treatment']).overall1.mean().reset_index()
rows=[]
for tr in [0,1]:
    z=d[d.treatment==tr];zg=base_grade[base_grade.treatment==tr]
    rows.append({'treatment':tr,'assigned':len(z),'complete_time':int(z.change.notna().sum()),'quality_participants':int(qc[qc.treatment==tr].prolific_pid.nunique()),'quality_rows':int((qc.treatment==tr).sum()),'baseline_time':float(z.time1.mean()),'baseline_grade':float(zg.overall1.mean()),'prior_use_observed':int(z.prior_use.notna().sum()),'prior_use_rate':float(z.prior_use.mean()),'linear':int((z.incentive_arm=='linear').sum()),'convex':int((z.incentive_arm=='convex').sum())})
res['descriptive']=rows
res['missingness']=d.assign(missing_time=d.change.isna(),missing_prior=d.prior_use.isna()).groupby('treatment')[['missing_time','missing_prior']].sum().astype(int).to_dict()
res['quality_pair_counts']=qc.groupby('treatment').size().to_dict()
res['ecdf']={}
for name,frame,var,limits in [('time',dc,'time2',(0,90)),('quality',qc,'overall2',(1,7))]:
    grid=np.unique(np.r_[limits,frame[var].to_numpy()])
    res['ecdf'][name]={'x':np.repeat(grid,2).tolist()}
    for tr in [0,1]:
        values=frame.loc[frame.treatment==tr,var].to_numpy()
        res['ecdf'][name][str(tr)]=[p for v in grid for p in [float(np.mean(values<v-1e-8)),float(np.mean(values<=v+1e-8))]]
(OUT/'presentation_checks.json').write_text(json.dumps(res,indent=2))
flat=[]
for name,v in res.items():
    if isinstance(v,dict) and 'effect' in v:
        for term in ['effect','interaction','previous_users_effect']:
            if term in v: flat.append({'model':name,'term':term,'N':v['N'],'participants':v['participants'],**v[term]})
pd.DataFrame(flat).to_csv(OUT/'additional_results.csv',index=False)
print(json.dumps(res,indent=2))
