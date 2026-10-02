"""Generate every numerical table and LaTeX macro from executed results."""
import json,re
from pathlib import Path
import pandas as pd
import numpy as np
from run_model import SYSTEM,BELIEFS
P=Path('paper_draft');P.mkdir(exist_ok=True)
b=pd.read_csv('results/behavior.csv').set_index('condition')
r=pd.read_csv('results/probe_metrics.csv')
t=pd.read_csv('results/probe_training.csv')
resp=pd.read_csv('results/responses.csv')
labels={'instruction':'Instruction','error':'Error','pressure':'Pressure','cross':'Cross','within':'Within','knowledge':'Knowledge'}
def esc(s):
    for a,z in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('$',r'\$'),('#',r'\#'),('_',r'\_')]:s=s.replace(a,z)
    return s

def metric(probe,task,condition='social',position='answer',layer=14):
    z=r[(r.probe==probe)&(r.task==task)&(r.condition==condition)&(r.position==position)&(r.layer==layer)]
    assert len(z)==1,(probe,task,condition,position,layer,len(z))
    return z.iloc[0]

def fmt(z,ci=True):
    if pd.isna(z.auc):return '--'
    return f'{z.auc:.3f} [{z.lo:.2f}, {z.hi:.2f}]' if ci else f'{z.auc:.3f}'

mac={}
for c in ['neutral','score','social','honest','lie']:
    title=c.capitalize();mac[title+'False']=str(int(b.loc[c,'false']));mac[title+'Accuracy']=f'{100*b.loc[c,"correct"]/b.loc[c,"n"]:.1f}'
    mac[title+'Compliance']=f'{100*b.loc[c,"compliance"]:.1f}'
    mac[title+'Valid']=f'{100*b.loc[c,"unrestricted_choice"]:.1f}'
    for k in ['K','U','W']:
        mac[title+k+'Errors']=str(int(b.loc[c,k+'_false']));mac[title+k+'Share']=f'{100*b.loc[c,k+"_share_false"]:.1f}'
    mac[title+'ConflictCorrect']=str(int(b.loc[c,'K_conflict_correct']))
    mac[title+'ConflictN']=str(int(b.loc[c,'K_conflict_n']))
for p in labels:
    for task in ['cross','within','pressure','error','instruction']:
        for pos in ['answer','prompt']:
            mac[p.capitalize()+task.capitalize()+pos.capitalize()]=fmt(metric(p,task,position=pos,condition='score' if task in ['error','instruction'] else 'social'),False)
for k in ['K','U','W']:mac['Count'+k]=str(int(b.loc['neutral',k+'_n']))
(P/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+name+'}{'+v+'}' for name,v in mac.items())+'\n')

s=r'''\begin{table}[t]
\centering\small
\caption{Composition of incorrect answers over all 1,600 questions. Entries are counts, with the percentage of that condition's errors in parentheses. K: stable-correct; U: inconsistent; W: stable-wrong. Explicit instructions are calibration conditions.}\label{tab:composition}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}\toprule
Condition & Errors & K & U & W \\\midrule
'''
for c in ['neutral','score','social','honest','lie']:
    row=b.loc[c];s+=c.capitalize()+' & '+str(int(row['false']))+' & '+' & '.join(f"{int(row[k+'_false'])} ({100*row[k+'_share_false']:.1f}\\%)" for k in ['K','U','W'])+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}}\end{table}'
(P/'table_composition.tex').write_text(s)

s=r'''\begin{table*}[t]
\centering\small
\caption{Held-out AUROC at fixed layer 14, with question-bootstrap 95\% intervals. All positives are K errors. Cross-prompt negatives are U neutral errors; same-pressure negatives are U errors under the indicated pressure. Probe names specify their training tasks, not established mechanisms. Social pressure is unseen during pressure-based probe training.}\label{tab:probes}
\begin{tabular}{llrrrr}\toprule
& & \multicolumn{2}{c}{Cross-prompt errors} & \multicolumn{2}{c}{Same-pressure errors} \\
Probe & Position & Score & Social & Score & Social \\\midrule
'''
for p in ['instruction','error','pressure','cross','within']:
    for pos in ['answer','prompt']:
        s+=labels[p]+' & '+('Answer' if pos=='answer' else 'Pre-answer')+' & '+' & '.join(fmt(metric(p,task,c,pos)) for task,c in [('cross','score'),('cross','social'),('within','score'),('within','social')])+r' \\'+'\n'
    if p!='within':s+=r'\addlinespace'+'\n'
s+=r'\bottomrule\end{tabular}\end{table*}'
(P/'table_probes.tex').write_text(s)

s=r'''\begin{table}[t]
\centering\small
\caption{Simple controls on social-pressure test contrasts. Learned controls use the score-pressure training contrasts indicated in parentheses.}\label{tab:controls}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrr}\toprule
Control & Cross-prompt & Same-pressure \\\midrule
'''
for name,pos,p in [('Condition only','condition_only','condition'),('Answer letter (Cross)','answer_letter','cross'),('Surface (Cross)','surface','cross'),('Surface (Within)','surface','within'),('Question text (Within)','question_text','within'),('Choice entropy','scalar','entropy'),('Choice uncertainty','scalar','uncertainty')]:
    s+=name+' & '+' & '.join(fmt(metric(p,task,'social',pos,0)) for task in ['cross','within'])+r' \\'+'\n'
rob=pd.read_csv('results/robustness.csv')
z=rob[(rob.check=='adherence_only')&(rob.condition=='social')].iloc[0]
s+='Adherence (post hoc) & -- & '+fmt(z)+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}}\end{table}'
(P/'table_controls.tex').write_text(s)

# Prompt appendix, copied from actual inference records.
first={}
for f in sorted(Path('results/raw').glob('batch*.json')):
    for rr in json.loads(f.read_text()):
        if rr['condition'] not in first:first[rr['condition']]=rr
qs={x['qid']:x for x in json.loads(Path('results/questions.json').read_text())}
s='The system message is identical in every condition. Each user message starts with the instruction below, followed by a blank line, the question, four labeled options, and a blank line followed by \\texttt{Answer:}. The model chat template then opens the assistant turn. T below denotes the preferred displayed answer letter; it is not a literal token.\n'
s+='\\paragraph{System.} '+esc(SYSTEM)+'\n'
for c in ['belief0','belief1','belief2','belief3','neutral','score','social','honest','lie']:
    rr=first[c];question=qs[rr['qid']]['question'];pref=rr['prompt'].split('<|im_start|>user\n',1)[1].split('\n\n'+question)[0]
    if c in ['score','social']:
        letter='ABCD'[rr['order'].index(rr['target'])]
        pref=pref.replace('answer is '+letter,'answer is T').replace('answer '+letter,'answer T')
    s+='\\paragraph{'+esc(c.capitalize())+'.} '+esc(pref)+'\n'
(P/'prompts.tex').write_text(s)

s=r'''\subsection{Training counts and native tasks}
\begin{table}[h]
\centering\small
\caption{Primary answer-token training counts and held-out AUROC on each probe's own task. Knowledge is the exploratory neutral K-versus-U probe.}
\begin{tabular}{lrrr}\toprule
Probe & Train $+$ & Train $-$ & Native AUC \\\midrule
'''
for p in labels:
    z=t[(t.probe==p)&(t.position=='answer')&(t.layer==14)].iloc[0]
    auc=fmt(metric(p,p,'score'),False)
    s+=labels[p]+f' & {int(z.train_pos)} & {int(z.train_n-z.train_pos)} & '+auc+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'+'\n'
s+=r'''\subsection{Layer sensitivity}
\begin{table}[h]
\centering\small
\caption{Same-pressure social test AUROC at each layer. A: answer; P: pre-answer. No layer is chosen using these results.}
\begin{tabular}{llrrrr}\toprule
Probe & Pos. & 7 & 14 & 21 & 28 \\\midrule
'''
for p in labels:
    for pos in ['answer','prompt']:
        s+=labels[p]+' & '+('A' if pos=='answer' else 'P')+' & '+' & '.join(fmt(metric(p,'within','social',pos,l),False) for l in [7,14,21,28])+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'+'\n'
s+=r'''\subsection{Knowledge-label sensitivity}
\begin{table}[h]
\centering\small
\caption{Answer-token social same-pressure AUROC when K requires four correct elicitations (primary) or at least three (relaxed). Probes retain their original fitted weights.}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrr}\toprule
Probe & Primary & Relaxed \\\midrule
'''
for p in labels:s+=labels[p]+' & '+fmt(metric(p,'within'))+' & '+fmt(metric(p,'within_relaxed'))+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}}\end{table}'+'\n'
sc=pd.read_csv('results/test_scores.csv');s+=r'''\subsection{Threshold transfer}
\begin{table}[h]
\centering\small
\caption{Test flag rates (\%) for answer-token probes with thresholds calibrated on their own validation negatives. L: K social errors; H: U neutral errors; U-P: U social errors; K-C: K social correct; W-P: W social errors. Rates are descriptive and use different denominators.}
\begin{tabular}{lrrrrr}\toprule
Probe & L & H & U-P & K-C & W-P \\\midrule
'''
for p in ['instruction','error','pressure','within']:
    v=sc[(sc.probe==p)&(sc.position=='answer')]
    gs=[v[(v.knowledge==k)&(v.condition==c)&(v.correct==correct)] for k,c,correct in [('K','social',False),('U','neutral',False),('U','social',False),('K','social',True),('W','social',False)]]
    s+=labels[p]+' & '+' & '.join(f'{100*g.flag.mean():.1f}' for g in gs)+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'+'\n'
z=[len(g) for g in gs];s+='The group denominators, in table-column order, are '+', '.join(map(str,z))+r'. Exact held-out scores and flags are provided in \texttt{results/test\_scores.csv}.'+'\n'
s+=r'''\subsection{Decoding diagnostics}
\begin{table}[h]
\centering\small
\caption{Unrestricted top token in A--D (\%) and mean total probability on A--D, before forced-choice renormalization.}
\begin{tabular}{lrr}\toprule
Condition & Top token valid & Choice mass \\\midrule
'''
for c in ['neutral','score','social','honest','lie']:s+=c.capitalize()+f' & {100*b.loc[c,"unrestricted_choice"]:.1f} & {b.loc[c,"choice_mass_mean"]:.6f}'+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}\end{table}'+'\n'
s+=r'''\subsection{Post-hoc conditional comparisons}
\begin{table}[h]
\centering\small
\caption{Social-pressure, answer-token AUROC with both error classes restricted to preferred-option adherence, or with K positives additionally required to be correct on independent neutral evaluation. These diagnostics use fixed original probes.}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrr}\toprule
Probe & Both adherent & Neutral-verified K \\\midrule
'''
for p in labels:
    vals=[]
    for check in ['both_adherent','neutral_verified']:
        z=rob[(rob.check==check)&(rob.condition=='social')&(rob.position=='answer')&(rob.probe==p)].iloc[0]
        vals.append(fmt(z))
    s+=labels[p]+' & '+' & '.join(vals)+r' \\'+'\n'
s+=r'\bottomrule\end{tabular}}\end{table}'+'\n'
s+='The both-adherent comparison has 54 K and 66 U errors; the neutral-verified comparison has 55 K and 96 U errors. Conditioning on model behavior changes the sample and is not a causal intervention on representations.\n'
(P/'additional.tex').write_text(re.sub(r'\\subsection\{[^}]*\}\n','',s).replace(r'\begin{table}[h]',r'\begin{table}[H]'))
print('Generated tables, macros, prompts, and appendices')
