import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    class _SeabornFallback:
        @staticmethod
        def set_theme(*args, **kwargs):
            plt.style.use('default')

        @staticmethod
        def barplot(x=None, y=None, data=None, ax=None, color=None, palette=None, **kwargs):
            ax = ax or plt.gca()
            if data is not None:
                if x is not None:
                    labels = data[x].astype(str).unique()
                    values = data.groupby(x)[y].mean() if y in data.columns else data[x].value_counts(sort=False).reindex(labels)
                    ax.bar(labels, values, color=color)
                    return ax
            ax.bar(range(len(x)), y, color=color)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([str(v) for v in x])
            return ax

        @staticmethod
        def histplot(values, ax=None, bins=20, color=None, **kwargs):
            ax = ax or plt.gca()
            ax.hist(values.dropna(), bins=bins, color=color, edgecolor='white')
            return ax

        @staticmethod
        def boxplot(data=None, x=None, y=None, hue=None, ax=None, palette=None, legend=False, **kwargs):
            ax = ax or plt.gca()
            groups = data[x].astype(str).unique()
            if hue is None:
                values = [data.loc[data[x].astype(str)==g, y].dropna().tolist() for g in groups]
                ax.boxplot(values, labels=groups)
            else:
                hue_groups = data[hue].astype(str).unique()
                positions, boxes, labels = [], [], []
                width = 0.8 / len(hue_groups)
                for gi, group in enumerate(groups):
                    for hi, hue_group in enumerate(hue_groups):
                        subset = data[(data[x].astype(str)==group) & (data[hue].astype(str)==hue_group)]
                        positions.append(gi + 1 + (hi-(len(hue_groups)-1)/2)*width)
                        boxes.append(subset[y].dropna().tolist())
                        if hi == 0:
                            labels.append(group)
                ax.boxplot(boxes, positions=positions, widths=width)
                ax.set_xticks(range(1, len(groups)+1), labels)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            return ax

        @staticmethod
        def scatterplot(data=None, x=None, y=None, hue=None, palette=None, ax=None, alpha=None, s=None, **kwargs):
            ax = ax or plt.gca()
            if hue is None:
                ax.scatter(data[x], data[y], s=s, alpha=alpha)
            else:
                for group in data[hue].astype(str).unique():
                    subset = data[data[hue].astype(str)==group]
                    ax.scatter(subset[x], subset[y], s=s, alpha=alpha, label=str(group))
                ax.legend(title=hue)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            return ax

        @staticmethod
        def heatmap(data, ax=None, cmap=None, center=None, vmin=None, vmax=None,
                    cbar_kws=None, xticklabels=None, yticklabels=None, annot=False,
                    fmt=None, cbar=True, **kwargs):
            ax = ax or plt.gca()
            image = ax.imshow(data.values if hasattr(data, 'values') else data,
                              cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            if cbar:
                ax.figure.colorbar(image, ax=ax)
            if annot:
                values = data.values if hasattr(data, 'values') else data
                for row in range(values.shape[0]):
                    for column in range(values.shape[1]):
                        ax.text(column, row, format(values[row, column], fmt or 'g'),
                                ha='center', va='center')
            if hasattr(data, 'index'):
                ax.set_xticks(range(len(data.columns)), xticklabels or data.columns)
                ax.set_yticks(range(len(data.index)), yticklabels or data.index)
            return ax

    sns = _SeabornFallback()

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, precision_recall_curve, average_precision_score)
from pathlib import Path
RS=42; NAVY='#14283C'; ORG='#C0362C'
sns.set_theme(style='whitegrid', font_scale=0.95)
PAL={'No':'#4C78A8','Yes':ORG}
d=pd.read_csv('data/employee_attrition_dataset_10000.csv')
d['y']=(d.Attrition=='Yes').astype(int)
R={}
R['n']=len(d); R['rate']=d.y.mean(); R['left']=int(d.y.sum())
num=[c for c in d.select_dtypes('number').columns if c not in ('Employee_ID','y')]
cat=['Gender','Marital_Status','Department','Job_Role','Overtime']
def save(n):
    Path('fig').mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'fig/{n}.png',dpi=160)
    plt.close()

# ---------- DESCRIPTIVE ----------
R['desc']=d[num].describe().T[['mean','std','min','50%','max']].round(2).reset_index().values.tolist()
# 1 class balance
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
vc=d.Attrition.value_counts(); ax[0].pie(vc,labels=vc.index,autopct='%1.1f%%',colors=[PAL['No'],ORG],startangle=90)
ax[0].set_title('Attrition split')
a=d.groupby('Department').y.mean().sort_values()*100
ax[1].barh(a.index,a.values,color=NAVY); ax[1].axvline(d.y.mean()*100,color=ORG,ls='--'); ax[1].set_xlabel('Attrition rate (%)'); ax[1].set_title('Attrition rate by department')
save('01_balance')
# 2 categorical rates
fig,ax=plt.subplots(2,3,figsize=(12,6.5)); ax=ax.ravel()
for i,c in enumerate(cat):
    r=d.groupby(c).y.mean()*100
    sns.barplot(x=r.index,y=r.values,ax=ax[i],color=NAVY); ax[i].axhline(d.y.mean()*100,color=ORG,ls='--')
    ax[i].set_title(f'Attrition % by {c}'); ax[i].set_ylabel('%'); ax[i].set_xlabel(''); ax[i].set_ylim(0,28)
ax[5].axis('off'); ax[5].text(0,0.5,'Dashed line = overall\nattrition rate (20.0%)',fontsize=11)
save('02_categorical')
chi=[]
for c in cat:
    ct=pd.crosstab(d[c],d.y); s,p,_,_=stats.chi2_contingency(ct); chi.append([c,round(s,2),round(p,3)])
R['chi']=chi
# 3 histograms
fig,ax=plt.subplots(3,4,figsize=(14,8)); ax=ax.ravel()
for i,c in enumerate(['Age','Monthly_Income','Hourly_Rate','Years_at_Company','Years_in_Current_Role','Years_Since_Last_Promotion','Training_Hours_Last_Year','Average_Hours_Worked_Per_Week','Absenteeism','Distance_From_Home','Project_Count','Job_Satisfaction']):
    sns.histplot(d[c],ax=ax[i],bins=20,color=NAVY); ax[i].set_title(c.replace('_',' '),fontsize=10); ax[i].set_xlabel('')
save('03_hist')
# 4 boxplots
bp=['Age','Monthly_Income','Years_at_Company','Years_Since_Last_Promotion','Average_Hours_Worked_Per_Week','Absenteeism','Distance_From_Home','Training_Hours_Last_Year']
fig,ax=plt.subplots(2,4,figsize=(14,7)); ax=ax.ravel()
for i,c in enumerate(bp):
    sns.boxplot(data=d,x='Attrition',y=c,hue='Attrition',palette=PAL,ax=ax[i],legend=False); ax[i].set_title(c.replace('_',' '),fontsize=10); ax[i].set_xlabel('')
save('04_boxplots')
# 5 box by dept income and job role
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
sns.boxplot(data=d,x='Department',y='Monthly_Income',hue='Attrition',palette=PAL,ax=ax[0]); ax[0].set_title('Monthly income by department & attrition')
sns.boxplot(data=d,x='Job_Role',y='Monthly_Income',hue='Attrition',palette=PAL,ax=ax[1]); ax[1].set_title('Monthly income by job role & attrition')
save('05_box_income')
# 6 scatter plots
s=d.sample(2500,random_state=RS)
fig,ax=plt.subplots(2,2,figsize=(12,8)); ax=ax.ravel()
for a_,(x,y) in zip(ax,[('Age','Monthly_Income'),('Years_at_Company','Years_Since_Last_Promotion'),('Average_Hours_Worked_Per_Week','Job_Satisfaction'),('Distance_From_Home','Absenteeism')]):
    sns.scatterplot(data=s,x=x,y=y,hue='Attrition',palette=PAL,alpha=0.45,s=18,ax=a_); a_.set_title(f'{x.replace("_"," ")} vs {y.replace("_"," ")}',fontsize=10)
save('06_scatter')
# 7 correlation heatmap
fig,ax=plt.subplots(figsize=(11,9))
cm=d[num+['y']].corr(); sns.heatmap(cm,cmap='RdBu_r',center=0,vmin=-0.2,vmax=0.2,ax=ax,cbar_kws={'label':'correlation (scale ±0.2)'},xticklabels=[c[:18] for c in cm.columns],yticklabels=[c[:18] for c in cm.columns])
ax.set_title('Correlation matrix (colour scale clipped at ±0.2)')
save('07_heatmap')
# 8 correlation with target + tests
rows=[]
for c in num:
    r,p=stats.pointbiserialr(d.y,d[c]); mw=stats.mannwhitneyu(d.loc[d.y==1,c],d.loc[d.y==0,c]).pvalue
    rows.append([c,round(r,3),round(p,3),round(mw,3)])
R['tests']=rows
rr=pd.Series({r[0]:r[1] for r in rows}).sort_values()
fig,ax=plt.subplots(figsize=(8,6)); ax.barh(rr.index,rr.values,color=[ORG if v>0 else NAVY for v in rr.values]); ax.axvline(0,color='k',lw=.8)
ax.set_title('Correlation of each numeric feature with attrition'); ax.set_xlabel('point-biserial r'); ax.set_xlim(-0.1,0.1)
save('08_target_corr')
R['maxabs_r']=float(rr.abs().max()); R['sig_t']=int(sum(1 for r in rows if r[2]<0.05)); R['sig_c']=int(sum(1 for r in chi if r[2]<0.05))

# ---------- PREDICTIVE ----------
d['Income_per_Level']=d.Monthly_Income/d.Job_Level
d['Promo_Gap_Ratio']=d.Years_Since_Last_Promotion/(d.Years_in_Current_Role+1)
d['Overwork_Index']=d.Average_Hours_Worked_Per_Week*d.Project_Count
d['Satisfaction_Mean']=d[['Job_Satisfaction','Work_Life_Balance','Work_Environment_Satisfaction','Relationship_with_Manager']].mean(axis=1)
feats_num=num+['Income_per_Level','Promo_Gap_Ratio','Overwork_Index','Satisfaction_Mean']
X=d[feats_num+cat]; y=d.y
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RS)
R['ntr']=len(Xtr); R['nte']=len(Xte); R['tr_rate']=float(ytr.mean()); R['te_rate']=float(yte.mean())
pre_lin=ColumnTransformer([('n',StandardScaler(),feats_num),('c',OneHotEncoder(drop='first'),cat)])
pre_tree=ColumnTransformer([('n','passthrough',feats_num),('c',OneHotEncoder(drop='first'),cat)])
spw=(ytr==0).sum()/(ytr==1).sum()
cv=StratifiedKFold(5,shuffle=True,random_state=RS)
specs={
 'Logistic Regression':(Pipeline([('p',pre_lin),('m',LogisticRegression(max_iter=2000,class_weight='balanced'))]),{'m__C':[0.01,0.1,1,10]}),
 'Decision Tree':(Pipeline([('p',pre_tree),('m',DecisionTreeClassifier(class_weight='balanced',random_state=RS))]),{'m__max_depth':[3,5,8],'m__min_samples_leaf':[20,50,100]}),
 'Random Forest':(Pipeline([('p',pre_tree),('m',RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=RS,n_jobs=-1))]),{'m__max_depth':[4,8,None],'m__min_samples_leaf':[5,20,50]}),
 'XGBoost':(Pipeline([('p',pre_tree),('m',XGBClassifier(n_estimators=200,scale_pos_weight=spw,eval_metric='logloss',random_state=RS,n_jobs=4,verbosity=0))]),{'m__max_depth':[2,3,5],'m__learning_rate':[0.03,0.1],'m__subsample':[0.8,1.0]}),
}
fit={}; res=[]; cvres=[]
for n,(pipe,grid) in specs.items():
    gs=GridSearchCV(pipe,grid,scoring='roc_auc',cv=cv,n_jobs=1).fit(Xtr,ytr); fit[n]=gs
    best=gs.best_estimator_
    cvs=cross_val_score(best,Xtr,ytr,cv=cv,scoring='roc_auc')
    pr=best.predict_proba(Xte)[:,1]; yp=(pr>=0.5).astype(int)
    res.append(dict(Model=n,Accuracy=accuracy_score(yte,yp),Precision=precision_score(yte,yp,zero_division=0),Recall=recall_score(yte,yp),F1=f1_score(yte,yp),ROC_AUC=roc_auc_score(yte,pr),PR_AUC=average_precision_score(yte,pr),CV_AUC=cvs.mean(),CV_SD=cvs.std(),params=str(gs.best_params_)))
    print(n,gs.best_params_,round(res[-1]['ROC_AUC'],3),flush=True)
# baseline
yp0=np.zeros(len(yte)); res.append(dict(Model='Baseline (predict "Stay")',Accuracy=accuracy_score(yte,yp0),Precision=0,Recall=0,F1=0,ROC_AUC=0.5,PR_AUC=float(yte.mean()),CV_AUC=0.5,CV_SD=0,params='-'))
M=pd.DataFrame(res)
M.to_csv('outputs/model_comparison.csv',index=False)
R['models']=M.drop(columns='params').round(4).values.tolist(); R['params']={r['Model']:r['params'] for r in res}
mm=M[M.Model.isin(specs)].reset_index(drop=True)
bestname=mm.loc[mm.ROC_AUC.idxmax(),'Model']; R['best']=bestname
# bootstrap CI for AUC best on test
best=fit[bestname].best_estimator_; pb=best.predict_proba(Xte)[:,1]
rng=np.random.RandomState(RS); bs=[]
yt=yte.values
for _ in range(1000):
    i=rng.randint(0,len(yt),len(yt)); bs.append(roc_auc_score(yt[i],pb[i]))
R['auc_ci']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
# permutation test of AUC for each model: label-shuffle p-value
R['perm_p']={}
for n in specs:
    p_=fit[n].best_estimator_.predict_proba(Xte)[:,1]; a0=roc_auc_score(yte,p_)
    ps=[roc_auc_score(rng.permutation(yt),p_) for _ in range(1000)]
    R['perm_p'][n]=float((np.sum(np.array(ps)>=a0)+1)/1001)

# charts
fig,ax=plt.subplots(figsize=(7,5.5))
cols=['#4C78A8','#F58518','#54A24B',ORG]
for (n,_),c in zip(specs.items(),cols):
    pr=fit[n].best_estimator_.predict_proba(Xte)[:,1]; f,t,_=roc_curve(yte,pr); ax.plot(f,t,color=c,label=f'{n} (AUC {roc_auc_score(yte,pr):.3f})')
ax.plot([0,1],[0,1],'k--',label='Random (AUC 0.500)'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title('ROC curves, hold-out test set'); ax.legend(loc='lower right')
save('09_roc')
fig,ax=plt.subplots(figsize=(7,5.5))
for (n,_),c in zip(specs.items(),cols):
    pr=fit[n].best_estimator_.predict_proba(Xte)[:,1]; p,r,_=precision_recall_curve(yte,pr); ax.plot(r,p,color=c,label=f'{n} (AP {average_precision_score(yte,pr):.3f})')
ax.axhline(yte.mean(),color='k',ls='--',label='Baseline (prevalence 0.200)'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-recall curves, test set'); ax.legend()
save('10_pr')
fig,ax=plt.subplots(figsize=(9,4.6)); m=mm.melt(id_vars='Model',value_vars=['Accuracy','Precision','Recall','F1','ROC_AUC']); sns.barplot(data=m,x='variable',y='value',hue='Model',palette=cols,ax=ax); ax.set_ylim(0,1); ax.set_xlabel(''); ax.set_title('Model comparison on the test set'); ax.legend(fontsize=8,ncol=2)
save('11_compare')
fig,ax=plt.subplots(1,4,figsize=(14,3.4))
for a_,(n,_) in zip(ax,specs.items()):
    pr=fit[n].best_estimator_.predict_proba(Xte)[:,1]; cmx=confusion_matrix(yte,(pr>=0.5).astype(int)); sns.heatmap(cmx,annot=True,fmt='d',cmap='Blues',cbar=False,ax=a_,xticklabels=['Stay','Leave'],yticklabels=['Stay','Leave']); a_.set_title(n,fontsize=10); a_.set_xlabel('Predicted'); a_.set_ylabel('Actual')
save('12_confusion')
# feature importance of best (permutation) on test
from sklearn.inspection import permutation_importance
pi=permutation_importance(best,Xte,yte,scoring='roc_auc',n_repeats=10,random_state=RS,n_jobs=1)
imp=pd.Series(pi.importances_mean,index=X.columns).sort_values().tail(12); sd=pd.Series(pi.importances_std,index=X.columns)[imp.index]
fig,ax=plt.subplots(figsize=(8,5)); ax.barh(imp.index,imp.values,xerr=sd.values,color=NAVY); ax.axvline(0,color='k',lw=.8); ax.set_title(f'Permutation importance ({bestname}, test ΔAUC)'); ax.set_xlabel('Drop in ROC-AUC when shuffled')
save('13_importance')
R['top_imp']=[[k,round(v,4)] for k,v in imp.sort_values(ascending=False).head(5).items()]
# SHAP for XGBoost
try:
    import shap
    xg=fit['XGBoost'].best_estimator_; Xt=xg.named_steps['p'].transform(Xte); fn=xg.named_steps['p'].get_feature_names_out()
    fn=[f.split('__')[1] for f in fn]
    sv=shap.TreeExplainer(xg.named_steps['m']).shap_values(Xt)
    plt.figure(figsize=(8,6)); shap.summary_plot(sv,pd.DataFrame(Xt,columns=fn),show=False,max_display=12); plt.title('SHAP summary (XGBoost, test set)'); save('14_shap')
except ModuleNotFoundError:
    print('SHAP not available, skipping SHAP analysis')
# threshold analysis on best
ths=np.linspace(0.05,0.95,91); pr_=[];rc=[];f1_=[]
for t in ths:
    yp=(pb>=t).astype(int); pr_.append(precision_score(yte,yp,zero_division=0)); rc.append(recall_score(yte,yp)); f1_.append(f1_score(yte,yp))
fig,ax=plt.subplots(figsize=(7,4.2)); ax.plot(ths,pr_,label='Precision'); ax.plot(ths,rc,label='Recall'); ax.plot(ths,f1_,label='F1'); ax.set_xlabel('Decision threshold'); ax.set_title(f'Threshold trade-off ({bestname})'); ax.legend()
save('15_threshold')
# lift / decile
dec=pd.DataFrame({'p':pb,'y':yt}); dec['d']=pd.qcut(dec.p.rank(method='first'),10,labels=False)
lift=dec.groupby('d').y.mean()[::-1].values/yt.mean()
fig,ax=plt.subplots(figsize=(7,4)); ax.bar(range(1,11),lift,color=NAVY); ax.axhline(1,color=ORG,ls='--'); ax.set_xlabel('Risk decile (1 = highest predicted risk)'); ax.set_ylabel('Lift vs average'); ax.set_title('Decile lift chart (test set)')
save('16_lift'); R['lift_top']=float(lift[0])
# LR coefficients table (statsmodels-like via sklearn) top
json.dump(R,open('results.json','w'),default=float,indent=1)
print(M.drop(columns='params').round(3).to_string()); print(R['best'],R['auc_ci'],R['perm_p'],R['top_imp'],R['lift_top'],R['sig_t'],R['sig_c'])
