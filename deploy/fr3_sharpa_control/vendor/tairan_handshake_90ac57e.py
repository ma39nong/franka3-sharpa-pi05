"""V2 perception handshake. No GT, episode length, model imports or real control."""
from dataclasses import dataclass,asdict
import math,heapq
LABELS=('pickup','harvest','return','adjust');NEXT=dict(zip(LABELS,LABELS[1:]+LABELS[:1]))
DEFAULT=dict(margin=0.,theta_prepare=.65,theta_commit=.8,K=1,min_confirm_span_s=0.,theta_ack=.7,ack_timeout_s=1.5,late_ack_enabled=False,late_ack_timeout_s=5.,legal_edges=True,ack_enabled=True,latency_s=.09,plan_period_s=.5,plan_offset_s=0.,prediction_ttl_s=.75)

def validate(config):
    if set(config)-set(DEFAULT):raise ValueError('Unknown policy option')
    c={**DEFAULT,**config}
    for k,v in c.items():
        if k in ['legal_edges','ack_enabled','late_ack_enabled']:
            if type(v)!=bool:raise ValueError(k+' must be boolean')
        elif type(v) not in [int,float] or not math.isfinite(v):raise ValueError(k+' must be finite numeric')
    if not 0<=c['margin']<=1:raise ValueError('Invalid margin')
    if not 0<=c['theta_prepare']<=c['theta_commit']<=1 or not 0<=c['theta_ack']<=1:raise ValueError('Invalid thresholds')
    if type(c['K'])!=int or c['K'] not in [1,2]:raise ValueError('K must be 1/2')
    if not 1<=c['ack_timeout_s']<=2 or c['min_confirm_span_s']<0 or not 0<=c['latency_s']<.5 or c['plan_period_s']<=0 or not 0<=c['plan_offset_s']<c['plan_period_s'] or c['prediction_ttl_s']<=0:raise ValueError('Invalid clock/span/timeout')
    if not 2<=c['late_ack_timeout_s']<=30 or (c['late_ack_enabled'] and c['late_ack_timeout_s']<=c['ack_timeout_s']):raise ValueError('Late ACK hard deadline must exceed soft timeout (2–30 seconds)')
    return c

@dataclass(frozen=True,slots=True)
class Evidence:
    observed_at:float
    available_at:float
    current:tuple
    future:tuple
    valid:bool=True
    def __post_init__(self):
        if not all(math.isfinite(v) for v in [self.observed_at,self.available_at]) or self.available_at<self.observed_at:raise ValueError('Noncausal time')
        for p in [self.current,self.future]:
            if len(p)!=4 or any(not math.isfinite(v) or v<0 for v in p) or abs(sum(p)-1)>1e-6:raise ValueError('Invalid probabilities')

@dataclass(frozen=True,slots=True)
class PromptSnapshot:
    active_prompt:str
    version:int
    committed_at:float|None

class Handshake:
    def __init__(self,c):
        self.c=validate(c);self.active='pickup';self.version=0;self.state='READY';self.pending=None;self.last=None;self.commit_at=None;self.deadline=None;self.hard_deadline=None;self.late_count=0;self.late_last=None;self.events=[]
    def snapshot(self):return PromptSnapshot(self.active,self.version,self.commit_at)
    def emit(self,kind,t,**kw):
        e=dict(kind=kind,time=round(t,9),active_stage=self.active,prompt_version=self.version,policy_state=self.state,pending=dict(self.pending) if self.pending else None,**kw);
        if self.c['late_ack_enabled']:e.update(ack_deadline=self.deadline,hard_ack_deadline=self.hard_deadline,late_ack_support_count=self.late_count)
        self.events.append(e);return e
    def cancel(self,t,reason):
        if self.pending:self.pending=None;self.state='READY';self.emit('cancel',t,reason=reason)
    def timer(self,t):
        if self.state=='WAIT_CURRENT_ACK' and t>=self.deadline-1e-9:
            if self.c['late_ack_enabled']:
                self.state='WAIT_LATE_ACK';self.late_count=0;self.late_last=None
                self.emit('timeout',t,reason='soft_ACK_timeout_waiting_late_evidence',severity='soft')
            else:
                self.state='RECOVERY_REQUIRED';self.emit('timeout',t,reason='perception_ACK_timeout_not_physical_failure')
        if self.state=='WAIT_LATE_ACK' and t>=self.hard_deadline-1e-9:
            self.state='RECOVERY_REQUIRED';self.late_count=0;self.late_last=None
            self.emit('hard_timeout',t,reason='late_ACK_window_closed_explicit_recovery_required',severity='hard')
        if self.pending and self.last and t-self.last.available_at>self.c['prediction_ttl_s']+1e-9:self.cancel(t,'stale_evidence')
    def recover(self,t,reason):
        if self.state!='RECOVERY_REQUIRED' or not reason:raise ValueError('Explicit external recovery required')
        self.state='READY';self.late_count=0;self.late_last=None;self.emit('external_recovery',t,reason=reason)
    def prediction(self,m):
        if self.last and m.observed_at<=self.last.observed_at:raise ValueError('Duplicate/HELD/reversed anchor')
        t=m.available_at
        if self.last and t-self.last.available_at>self.c['prediction_ttl_s']+1e-9:self.cancel(t,'stale_evidence')
        if self.state=='WAIT_CURRENT_ACK' and t>self.deadline+1e-9:self.timer(self.deadline)
        if self.state=='WAIT_LATE_ACK' and t>self.hard_deadline+1e-9:self.timer(self.hard_deadline)
        self.last=m
        if not m.valid:
            self.late_count=0;self.late_last=None
            if self.state=='WAIT_LATE_ACK':self.emit('late_ack_reset',t,reason='invalid_observation')
            self.cancel(t,'invalid_observation');return
        cur=LABELS[max(range(4),key=lambda i:m.current[i])];fut=LABELS[max(range(4),key=lambda i:m.future[i])]
        if self.state=='WAIT_CURRENT_ACK':
            if t<=self.deadline+1e-9 and m.observed_at>self.commit_at+1e-9 and cur==self.active and m.current[LABELS.index(cur)]>=self.c['theta_ack']:
                self.state='READY';self.emit('ack',t,observed_at=m.observed_at,commit_at=self.commit_at)
            else:self.emit('wait_ack',t,observed_at=m.observed_at);return
        if self.state=='WAIT_LATE_ACK':
            qualifies=(m.observed_at>self.commit_at+1e-9 and cur==self.active and m.current[LABELS.index(self.active)]>=self.c['theta_ack'])
            if qualifies:
                if self.late_last is None or t-self.late_last>self.c['prediction_ttl_s']+1e-9:self.late_count=0
                self.late_count+=1;self.late_last=t
                self.emit('late_ack_support',t,observed_at=m.observed_at,count=self.late_count)
                if self.late_count>=2:
                    self.state='READY';self.emit('late_ack',t,observed_at=m.observed_at,commit_at=self.commit_at,ack_delay_s=t-self.commit_at)
                    self.late_count=0;self.late_last=None
                else:return
            else:
                self.late_count=0;self.late_last=None
                self.emit('late_ack_reset',t,observed_at=m.observed_at,reason='current_not_confidently_active')
                self.timer(t);return
        if self.state=='RECOVERY_REQUIRED':self.emit('blocked',t,reason='explicit_recovery_required');return
        if fut==self.active:self.cancel(t,'candidate_equals_active');return
        if self.c['legal_edges'] and NEXT[self.active]!=fut:self.cancel(t,'candidate_changed');self.emit('illegal_candidate',t,candidate=fut);return
        p=m.future[LABELS.index(fut)]
        if p<self.c['theta_prepare'] or p-m.future[LABELS.index(self.active)]<0:self.cancel(t,'support_withdrawn');return
        if self.pending and self.pending['stage']!=fut:self.cancel(t,'candidate_changed')
        if self.pending is None:
            self.pending=dict(stage=fut,count=0,first_observed_at=m.observed_at,first_available_at=t,first_candidate_at=t,
                first_qualified_at=None,request_at=None,catch_up=fut==cur,source_target_time=m.observed_at+1,relation='anticipatory' if cur==self.active else 'catch_up' if cur==fut else 'disagreement')
            self.state='CANDIDATE_PENDING';self.emit('propose',t)
        q=self.pending;q['count']+=1;q['last_observed_at']=m.observed_at;q['last_confirm_available_at']=t;q['latest_confidence']=p;q['source_target_time']=m.observed_at+1
        qualified=q['count']>=self.c['K'] and m.observed_at-q['first_observed_at']>=self.c['min_confirm_span_s']-1e-9 and p>=self.c['theta_commit'] and p-m.future[LABELS.index(self.active)]>=self.c['margin']
        if qualified:
            if q['first_qualified_at'] is None:q['first_qualified_at']=t
            if q['request_at'] is None:q['request_at']=t;self.state='SWITCH_REQUESTED';self.emit('request',t)
        elif self.state=='SWITCH_REQUESTED':q['request_at']=None;self.state='CANDIDATE_PENDING';self.emit('request_withdrawn',t)
        self.emit('support',t,observed_at=m.observed_at)
    def planning(self,t):
        self.timer(t)
        if self.state!='SWITCH_REQUESTED':return
        q=dict(self.pending);old=self.active;self.active=q['stage'];self.version+=1;self.pending=None;self.commit_at=t
        self.state='WAIT_CURRENT_ACK' if self.c['ack_enabled'] else 'READY';self.deadline=t+self.c['ack_timeout_s']
        self.hard_deadline=t+self.c['late_ack_timeout_s'];self.late_count=0;self.late_last=None
        self.emit('commit',t,from_stage=old,to_stage=self.active,commit_at=t,evidence=q,prompt_snapshot=asdict(self.snapshot()))

def replay(rows,config,duration):
    c=validate(config);p=Handshake(c);queue=[]
    for i,r in enumerate(rows):
        # Construct a strict whitelist; future_valid and GT fields never enter policy.
        heads=r['heads']
        if set(heads)!=set(['current','future']):raise ValueError('Handshake requires actual paired Joint Current+Future')
        m=Evidence(r['observed_at'],round(r['observed_at']+c['latency_s'],9),tuple(heads['current']['probabilities'][k] for k in LABELS),tuple(heads['future']['probabilities'][k] for k in LABELS))
        heapq.heappush(queue,(m.available_at,0,i,'prediction',m));heapq.heappush(queue,(round(m.available_at+c['prediction_ttl_s']+1e-8,9),1,i,'timer',None))
    n=0
    while c['plan_offset_s']+n*c['plan_period_s']<duration:
        t=round(c['plan_offset_s']+n*c['plan_period_s'],9);heapq.heappush(queue,(t,2,n,'planning',None));n+=1
    while queue:
        t,_,i,kind,m=heapq.heappop(queue)
        if t>=duration:continue
        before=p.version
        if kind=='prediction':p.prediction(m)
        elif kind=='planning':p.planning(t)
        else:p.timer(t)
        if p.version!=before and c['ack_enabled']:heapq.heappush(queue,(round(p.deadline,9),1,n+p.version,'timer',None))
        if p.version!=before and c['ack_enabled'] and c['late_ack_enabled']:heapq.heappush(queue,(round(p.hard_deadline,9),1,n+p.version+100000,'timer',None))
    intervals=[];start=0.;stage='pickup';version=0
    for e in p.events:
        if e['kind']=='commit':intervals.append(dict(start_s=start,end_s=e['time'],stage=stage,prompt_version=version));start=e['time'];stage=e['active_stage'];version=e['prompt_version']
    intervals.append(dict(start_s=start,end_s=duration,stage=stage,prompt_version=version))
    pending=[];opened=None
    for e in p.events:
        if e['kind']=='propose':opened=dict(start_s=e['time'],stage=e['pending']['stage'])
        if e['kind'] in ['cancel','commit'] and opened:pending.append(dict(**opened,end_s=e['time']));opened=None
    if opened:pending.append(dict(**opened,end_s=duration))
    summary=dict(commits=p.version,acks=sum(e['kind']=='ack' for e in p.events),timeouts=sum(e['kind']=='timeout' for e in p.events),state=p.state,external_recoveries=0,finish='not_supported')
    if c['late_ack_enabled']:summary.update(late_acks=sum(e['kind']=='late_ack' for e in p.events),hard_timeouts=sum(e['kind']=='hard_timeout' for e in p.events),late_ack_required_support=2)
    return dict(config=c,events=p.events,active_intervals=[v for v in intervals if v['end_s']>v['start_s']],pending_intervals=pending,
        summary=summary,policy_version='current-future-handshake-v2',prediction_hz=2)
