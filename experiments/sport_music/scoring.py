"""Coherent prefix-trie evaluation for separately supplied relational probes.

Public API:
  tokenization_preflight(tokenizer, probes, max_length=None) -> dict
  evaluate_probes(model, tokenizer, probes, output_path, batch_size=8,
                  max_new_tokens=8, max_length=None, device=None) -> dict
  summarize_rows(rows) -> dict

The output is one exclusive JSON file with raw rows, summaries, timings and
source/probe hashes. No dataset/model is loaded here; in particular this module
never opens a final-test file. Models and explicitly selected probes are supplied
by the caller. Every exact proper answer prefix is evaluated once per probe.
A node's full-vocabulary FP32 log-softmax supplies ALL its outgoing token edges.
Candidate posterior is a separately labelled normalization over provided labels;
it is never substituted for full-model continuation probability or log utility.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math,statistics,time,unicodedata

SCHEMA_VERSION=1
MASS_TOLERANCE=2e-6
PRIMARY_SCORE='mean correct full-string continuation log probability, no EOS, over primary probes'


def canonical(text):
    if not isinstance(text,str): raise TypeError('Expected string')
    return unicodedata.normalize('NFKC',text).casefold().strip()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_probe(probe):
    for k in ('id','prompt','probe_family','prompt_family','correct_answer','alternatives'):
        if k not in probe: raise ValueError('Missing probe field: '+k)
    if not isinstance(probe['id'],str) or not probe['id'] or not isinstance(probe['prompt'],str) or not probe['prompt']:
        raise ValueError('Nonempty string ID and prompt required')
    labels=probe['alternatives']
    if not isinstance(labels,list) or len(labels)<2 or any(not isinstance(a,str) or not canonical(a) or a!=a.strip() for a in labels):
        raise ValueError('At least two nonempty trimmed candidate strings required')
    if len(set(map(canonical,labels)))!=len(labels): raise ValueError('Canonical candidate collision')
    if probe['correct_answer'] is not None and labels.count(probe['correct_answer'])!=1: raise ValueError('Ground truth must exactly match one candidate or be null')
    if not isinstance(probe.get('answer_prefix',' '),str): raise ValueError('answer_prefix must be a string')


def encode_probe(tokenizer,probe,max_length=None):
    validate_probe(probe)
    prompt_ids=list(tokenizer.encode(probe['prompt'],add_special_tokens=False))
    if not prompt_ids: raise ValueError('Empty prompt token sequence')
    candidates=[]
    for answer in probe['alternatives']:
        ids=list(tokenizer.encode(probe['prompt']+probe.get('answer_prefix',' ')+answer,add_special_tokens=False))
        if ids[:len(prompt_ids)]!=prompt_ids: raise ValueError('Tokenizer changes prompt prefix at answer boundary: '+probe['id'])
        target=ids[len(prompt_ids):]
        if not target: raise ValueError('Empty candidate token sequence')
        if max_length is not None and len(ids)>max_length: raise ValueError('Truncation prohibited: '+probe['id'])
        candidates.append({'answer':answer,'token_ids':target,'token_count':len(target)})
    seq=[tuple(c['token_ids']) for c in candidates]
    for i,left in enumerate(seq):
        for j,right in enumerate(seq):
            if i!=j and len(left)<=len(right) and right[:len(left)]==left:
                raise ValueError('Candidate token sequences must be distinct and prefix-free: '+probe['id'])
    starts=[c['token_ids'][0] for c in candidates]
    if probe.get('paper_rank_required',False) and len(set(starts))!=len(starts):
        raise ValueError('Published first-token candidate rank requires distinct starts: '+probe['id'])
    # One node for each unique proper candidate prefix. Empty tuple is root.
    nodes={}
    for candidate in candidates:
        seq=candidate['token_ids']
        for depth,token in enumerate(seq):
            prefix=tuple(seq[:depth]);nodes.setdefault(prefix,set()).add(token)
    return {'id':probe['id'],'prompt_token_ids':prompt_ids,'candidates':candidates,
            'nodes':[{'prefix_token_ids':list(prefix),'outgoing_token_ids':sorted(children)} for prefix,children in sorted(nodes.items(),key=lambda x:(len(x[0]),x[0]))],
            'distinct_first_token_ids':sorted(set(starts)),
            'maximum_sequence_tokens':max(len(prompt_ids)+c['token_count'] for c in candidates),
            'candidate_sequences_prefix_free':True}


def tokenization_preflight(tokenizer,probes,max_length=None):
    if not isinstance(probes,list) or not probes or len({p['id'] for p in probes})!=len(probes):
        raise ValueError('Nonempty list of unique probe IDs required')
    records=[encode_probe(tokenizer,p,max_length) for p in probes]
    return {'status':'PASS_TOKENIZATION_PREFIX_TRIE','probe_count':len(records),
            'candidate_count':sum(len(r['candidates']) for r in records),
            'unique_prefix_nodes':sum(len(r['nodes']) for r in records),
            'maximum_sequence_tokens':max(r['maximum_sequence_tokens'] for r in records),
            'prefix_free':True,'EOS_added_or_scored':False,'records':records}


def sync(device):
    import torch
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def infer_device(model,device=None):
    if device is not None: return device
    try: return next(model.parameters()).device
    except StopIteration:
        try: return next(model.buffers()).device
        except StopIteration: return 'cpu'


def score_trie(model,tokenizer,probes,encoded,device,batch_size=8):
    """Internal read-only model evaluation. Caller supplies eval-mode model."""
    import torch
    if batch_size<1 or tokenizer.pad_token_id is None: raise ValueError('Positive batch size and pad token required')
    flat=[]
    for pi,record in enumerate(encoded):
        for ni,node in enumerate(record['nodes']):
            flat.append({'probe_index':pi,'node_index':ni,'ids':record['prompt_token_ids']+node['prefix_token_ids'],'children':node['outgoing_token_ids']})
    scored=[[None for _ in r['nodes']] for r in encoded];timings=[]
    seen=set()
    for first in range(0,len(flat),batch_size):
        group=flat[first:first+batch_size];width=max(len(n['ids']) for n in group)
        ids=torch.full((len(group),width),tokenizer.pad_token_id,dtype=torch.long,device=device);attention=torch.zeros_like(ids)
        for row,node in enumerate(group):
            size=len(node['ids']);ids[row,:size]=torch.tensor(node['ids'],device=device);attention[row,:size]=1
        sync(device);start=time.perf_counter()
        with torch.inference_mode():
            output=model(input_ids=ids,attention_mask=attention,use_cache=False,return_dict=True)
            if output.logits.shape[:2]!=ids.shape: raise ValueError('Expected one causal logit vector per input position')
            for row,node in enumerate(group):
                key=(node['probe_index'],node['node_index'])
                if key in seen: raise AssertionError('A shared prefix was evaluated twice')
                seen.add(key)
                # Prompt-only last position at root, or exact gold proper prefix.
                logits=output.logits[row,len(node['ids'])-1,:].float()
                logp=logits.log_softmax(-1)
                if not bool(torch.isfinite(logp).all()): raise ValueError('Nonfinite full-vocabulary distribution')
                child=torch.tensor(node['children'],device=device,dtype=torch.long)
                values=logp[child].detach().cpu().tolist()
                if any(v>1e-6 for v in values): raise ValueError('Positive log probability')
                child_mass=math.fsum(math.exp(v) for v in values)
                if child_mass>1+MASS_TOLERANCE: raise ValueError('Outgoing edge mass exceeds1')
                original=encoded[key[0]]['nodes'][key[1]]
                scored[key[0]][key[1]]={'prefix_token_ids':original['prefix_token_ids'],'input_token_count':len(node['ids']),
                    'edges':[{'token_id':t,'logprob':v} for t,v in zip(node['children'],values)],
                    'outgoing_probability_mass':child_mass,'full_vocabulary_logsumexp_after_normalization':float(logp.logsumexp(0)),
                    'full_vocabulary_size':len(logp),'forward_batch_index':first//batch_size}
                del logits,logp
        sync(device);timings.append({'first_prefix_node':first,'prefix_nodes':len(group),'input_tokens':sum(len(n['ids']) for n in group),'padded_tokens':len(group)*width,'seconds':time.perf_counter()-start})
        del output,ids,attention
    if len(seen)!=len(flat) or any(n is None for group in scored for n in group): raise AssertionError('Missing or duplicate trie evaluation')
    rows=[]
    for probe,record,nodes in zip(probes,encoded,scored):
        lookup={tuple(node['prefix_token_ids']):{e['token_id']:e['logprob'] for e in node['edges']} for node in nodes};root=lookup[()]
        unique_starts=len(root)==len(record['candidates'])
        candidates=[]
        for c in record['candidates']:
            lp=[lookup[tuple(c['token_ids'][:j])][token] for j,token in enumerate(c['token_ids'])];total=math.fsum(lp)
            candidates.append({**c,'token_logprobs':lp,'sum_logprob':total,'mean_logprob':total/c['token_count'],'full_model_probability':math.exp(total),'first_token_logprob':lp[0],'EOS_scored':False})
        mass=math.fsum(c['full_model_probability'] for c in candidates)
        if mass>1+MASS_TOLERANCE: raise ValueError('Prefix-free candidate mass exceeds1: '+probe['id'])
        highest=max(c['sum_logprob'] for c in candidates);logz=highest+math.log(math.fsum(math.exp(c['sum_logprob']-highest) for c in candidates))
        for c in candidates:
            c['candidate_posterior']=math.exp(c['sum_logprob']-logz)
            c['rank_min_zero_based']=sum(o['sum_logprob']>c['sum_logprob'] for o in candidates)
            c['rank_max_zero_based']=sum(o['sum_logprob']>=c['sum_logprob'] for o in candidates)-1
            c['first_token_rank_min_zero_based']=sum(v>c['first_token_logprob'] for v in root.values()) if unique_starts else None
            c['first_token_rank_max_zero_based']=sum(v>=c['first_token_logprob'] for v in root.values())-1 if unique_starts else None
        correct=next((c for c in candidates if c['answer']==probe['correct_answer']),None)
        metric={'sum_logprob':correct['sum_logprob'] if correct else None,
            'mean_token_logprob':correct['mean_logprob'] if correct else None,
            'candidate_posterior':correct['candidate_posterior'] if correct else None,
            'rank_min_zero_based':correct['rank_min_zero_based'] if correct else None,
            'rank_max_zero_based':correct['rank_max_zero_based'] if correct else None,
            'margin':correct['sum_logprob']-max(c['sum_logprob'] for c in candidates if c is not correct) if correct else None,
            'first_token_logprob':correct['first_token_logprob'] if correct else None,
            'first_token_rank_min_zero_based':correct['first_token_rank_min_zero_based'] if correct else None,
            'first_token_rank_max_zero_based':correct['first_token_rank_max_zero_based'] if correct else None,
            'candidate_mass':mass}
        metric['candidate_accuracy']=metric['margin']>0 if correct else None
        metric['first_token_accuracy']=correct['first_token_logprob']>max(v for t,v in root.items() if t!=correct['token_ids'][0]) if correct and unique_starts else None
        rows.append({**probe,'prompt_token_ids':record['prompt_token_ids'],'trie_nodes':nodes,'candidates':candidates,'correct':correct,'metrics':metric,
            'root_first_token_distribution':{'candidate_token_ids':sorted(root),'logprobs':[root[t] for t in sorted(root)],'source':'one prompt-only forward and full-vocabulary FP32 log-softmax','unique_start_count':len(root),'candidate_count':len(candidates),
                'rank_status':'APPLICABLE_UNIQUE_CANDIDATE_STARTS' if unique_starts else 'NOT_APPLICABLE_NONUNIQUE_CANDIDATE_STARTS',
                'rank_reason':None if unique_starts else 'Two or more candidate strings share a first token; first-token candidate rank/accuracy is undefined and remains null. Coherent full-string ranking is valid.'},
            'candidate_posterior_definition':'exp(string_logprob) / sum(exp(string_logprob)) over supplied prefix-free labels only; not full-model probability',
            'argmax_label':max(candidates,key=lambda c:c['sum_logprob'])['answer'],
            'argmax_tie_rule':'label order used only for the distribution diagnostic; accuracy requires strictly positive margin'})
    return rows,timings


def score_generation(probe,response,ids,correct_token_ids,eos_token_id,max_new_tokens):
    text=canonical(response);target=canonical(probe['correct_answer']) if probe['correct_answer'] is not None else None
    leading=bool(target is not None and text.startswith(target) and (len(text)==len(target) or not (text[len(target)].isalnum() or text[len(target)]=='_')))
    return {'response':response,'generated_token_ids':ids,'generated_tokens':len(ids),
        'hit_max_new_tokens':bool(len(ids)==max_new_tokens and (not ids or ids[-1]!=eos_token_id)),
        'exact_match':text==target if target is not None else None,
        'leading_answer_match':leading if target is not None else None,
        'leading_answer_token_match':ids[:len(correct_token_ids)]==correct_token_ids if correct_token_ids is not None else None,
        'response_sha256':hashlib.sha256(response.encode()).hexdigest(),
        'exact_rule':'NFKC, casefold, outer-whitespace strip; entire response equals isolated label',
        'leading_rule':'Normalized response begins with complete label and then end or non-word boundary; later prose is not graded'}


def _summary(rows):
    def mean(values):
        v=[x for x in values if x is not None];return statistics.fmean(v) if v else None
    field=lambda k:mean([r['metrics'][k] for r in rows])
    gen=lambda k:mean([r.get('generation',{}).get(k) for r in rows])
    lp=[r['metrics']['sum_logprob'] for r in rows if r['metrics']['sum_logprob'] is not None]
    result={'count':len(rows),'labelled_count':len(lp),'mean_sum_logprob':field('sum_logprob'),'mean_token_logprob':field('mean_token_logprob'),
        'candidate_accuracy':field('candidate_accuracy'),'mean_rank_zero_based':field('rank_min_zero_based'),'mean_rank_max_zero_based':field('rank_max_zero_based'),'mean_margin':field('margin'),
        'mean_candidate_posterior':field('candidate_posterior'),'mean_first_token_logprob':field('first_token_logprob'),'first_token_accuracy':field('first_token_accuracy'),
        'mean_first_token_rank_zero_based':field('first_token_rank_min_zero_based'),'first_token_rank_applicable_count':sum(r['metrics']['first_token_rank_min_zero_based'] is not None for r in rows),'exact_accuracy':gen('exact_match'),'leading_accuracy':gen('leading_answer_match'),'leading_token_accuracy':gen('leading_answer_token_match'),
        'median_sum_logprob':statistics.median(lp) if lp else None,'min_sum_logprob':min(lp) if lp else None,'max_sum_logprob':max(lp) if lp else None,
        'argmax_label_counts':dict(Counter(r['argmax_label'] for r in rows)),'maximum_candidate_mass':max((r['metrics']['candidate_mass'] for r in rows),default=None),
        'greedy_truncation_count':sum(r.get('generation',{}).get('hit_max_new_tokens',False) for r in rows)}
    result.update(sum_logprob=result['mean_sum_logprob'],candidate_top1_accuracy=result['candidate_accuracy'],candidate_mean_rank_zero_based=result['mean_rank_zero_based'],candidate_sum_margin=result['mean_margin'])
    return result


def summarize_rows(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('Nonempty unique scored rows required')
    def groups(field,subset=rows):
        values=sorted({str(r.get(field,'unspecified')) for r in subset})
        return {v:_summary([r for r in subset if str(r.get(field,'unspecified'))==v]) for v in values}
    primary=[r for r in rows if r.get('primary',r['probe_family']=='composed')]
    family=groups('prompt_family',primary)
    robust={}
    for metric in ('mean_sum_logprob','candidate_accuracy','mean_rank_zero_based','mean_margin','exact_accuracy','leading_accuracy'):
        values=[r[metric] for r in family.values() if r[metric] is not None]
        robust[metric]={'mean':statistics.fmean(values),'median':statistics.median(values),'min':min(values),'max':max(values)} if values else None
    return {'primary':_summary(primary),'overall':_summary(rows),'by_probe_family':groups('probe_family'),'by_prompt_family':groups('prompt_family'),'by_class_id':groups('class_id'),
        'primary_by_prompt_family':family,'primary_by_class_id':groups('class_id',primary),'primary_prompt_robustness':robust,
        'by_probe_and_prompt_family':{k:groups('prompt_family',[r for r in rows if r['probe_family']==k]) for k in sorted({r['probe_family'] for r in rows})},
        'by_probe_and_class':{k:groups('class_id',[r for r in rows if r['probe_family']==k]) for k in sorted({r['probe_family'] for r in rows})},
        'primary_score':PRIMARY_SCORE,'rank_ties':'minimum and maximum zero-based ranks retained; primary accuracy requires unique best string score',
        'posterior_note':'Candidate posterior is restricted-label normalization, not full-model probability',
        'dependence_note':'Prompt families repeat the same relations; reported query counts are not independent retraining/task seeds.'}


def evaluate_probes(model,tokenizer,probes,output_path,batch_size=8,max_new_tokens=8,max_length=None,device=None):
    import torch
    if batch_size<1 or max_new_tokens<1 or tokenizer.pad_token_id is None: raise ValueError('Positive batch/generation limits and pad token required')
    path=Path(output_path)
    if path.exists(): raise FileExistsError('Refusing to overwrite evaluation: '+str(path))
    tokenization=tokenization_preflight(tokenizer,probes,max_length)
    device=infer_device(model,device);old_training=model.training;old_padding=tokenizer.padding_side
    old_cache=getattr(model.config,'use_cache',False);old_checkpointing=bool(getattr(model,'is_gradient_checkpointing',False))
    start=time.perf_counter();generation_times=[]
    model.eval();model.config.use_cache=True
    if old_checkpointing: model.gradient_checkpointing_disable()
    try:
        rows,trie_times=score_trie(model,tokenizer,probes,tokenization['records'],device,batch_size)
        tokenizer.padding_side='left'
        for first in range(0,len(probes),batch_size):
            group=probes[first:first+batch_size]
            inputs=tokenizer([p['prompt'] for p in group],return_tensors='pt',padding=True,add_special_tokens=False)
            inputs={k:v.to(device) for k,v in inputs.items()};width=inputs['input_ids'].shape[1]
            if max_length is not None and width>max_length: raise ValueError('Generation prompt truncation prohibited')
            sync(device);began=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            sync(device)
            if not torch.equal(generated[:,:width],inputs['input_ids']): raise ValueError('Generation modified prompt prefix')
            counts=[]
            for offset,ids in enumerate(generated[:,width:].detach().cpu().tolist()):
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids: ids=ids[:ids.index(tokenizer.eos_token_id)+1]
                row=rows[first+offset];response=tokenizer.decode(ids,skip_special_tokens=True);target_ids=row['correct']['token_ids'] if row['correct'] else None
                row['generation']=score_generation(row,response,ids,target_ids,tokenizer.eos_token_id,max_new_tokens);counts.append(len(ids))
            generation_times.append({'first_probe':first,'probes':len(group),'generated_tokens':sum(counts),'seconds':time.perf_counter()-began})
            del generated,inputs
    finally:
        tokenizer.padding_side=old_padding;model.config.use_cache=old_cache
        if old_checkpointing: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train(old_training)
    result={'status':'COMPLETE','schema_version':SCHEMA_VERSION,'rows':rows,'summary':summarize_rows(rows),'tokenization':tokenization,
        'timing':{'total_seconds':time.perf_counter()-start,'trie_batches':trie_times,'generation_batches':generation_times},
        'settings':{'batch_size':batch_size,'max_new_tokens':max_new_tokens,'max_length':max_length,'device':str(device),'generation':'greedy','EOS_added_or_scored_in_likelihood':False,'candidate_mass_tolerance':MASS_TOLERANCE},
        'provenance':{'scoring_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'probes_sha256':canonical_sha(probes),'tokenizer_name_or_path':getattr(tokenizer,'name_or_path',None),'model_name_or_path':getattr(model.config,'_name_or_path',None),'caller_responsibility':'Caller binds exact model/checkpoint, data partition and training configuration; this module loads none.'},
        'interpretation':['Every unique token prefix is evaluated once per probe and every edge from that prefix shares the same FP32 normalized full-vocabulary distribution.','Candidate sequences are prefix-free; their probabilities are disjoint continuation events with mass at most1.','No EOS is scored, so full-string probabilities concern continuation prefixes, not stopping after the answer.','Candidate posterior is a separate restricted normalization and must not replace the full-model log-probability utility.']}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    return result
