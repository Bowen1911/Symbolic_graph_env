"""Opt-in, read-only post-goal path verification; original search remains strict."""
import re
from . import env

PROTOCOL='search_recheck_v1'


def block(world,plan):
    state=world.initial;lines=['RECHECK_BEGIN STATE '+env.format_vector(state,world.symbols,'sparse')]
    for rid in plan:
        state=env.step(state,world.rule(rid));assert state is not None
        lines.append(f'RECHECK_STEP {rid} STATE '+env.format_vector(state,world.symbols,'sparse'))
    assert all(a>=b for a,b in zip(state,world.goal))
    return '\n'.join(lines+['RECHECK_END'])


def verify(world,text,**kwargs):
    from . import verify as native
    text=native.restore_prefill(text,kwargs.pop('prefill',None))
    lines=text.splitlines();clean=[];active=False;state=None;paths=[];steps=0;path=[];error=None;goal_seen=False;closed=False
    def fail(code,line,detail):
        nonlocal error
        if error is None:error=native.Issue(code,line,detail)
    for number,line in enumerate(lines,1):
        line=line.strip()
        if not line:
            if not active:clean.append(line)
            continue
        if line.startswith('RECHECK'):
            if not goal_seen or closed:fail(env.UNEXPECTED_EVENT,number,'RECHECK 仅允许在 GOAL 后、think 闭合前')
            parts=line.split()
            if parts[:2]==['RECHECK_BEGIN','STATE']:
                if active:fail(env.UNEXPECTED_EVENT,number,'嵌套 RECHECK')
                value,_,issue=env.parse_vector(parts[2:],world.symbols)
                if issue or value!=world.initial:fail(issue or env.ROOT_MISMATCH,number,'核验初态错误')
                active=True;state=world.initial;path=[]
            elif len(parts)>=4 and parts[0]=='RECHECK_STEP' and parts[2]=='STATE':
                if not active:fail(env.UNEXPECTED_EVENT,number,'核验步骤缺少 BEGIN');continue
                rule=world.rule(parts[1]);value,_,issue=env.parse_vector(parts[3:],world.symbols)
                expected=env.step(state,rule) if rule is not None else None
                if rule is None:fail(env.UNKNOWN_RULE,number,parts[1])
                elif expected is None:fail(env.INSUFFICIENT_RESOURCE,number,parts[1])
                elif issue or value!=expected:fail(issue or env.WRONG_STATE,number,'核验状态与规则执行结果不一致')
                if expected is not None:state=expected
                path.append(parts[1]);steps+=1
            elif parts==['RECHECK_END']:
                if not active:fail(env.UNEXPECTED_EVENT,number,'核验缺少 BEGIN')
                elif not all(a>=b for a,b in zip(state,world.goal)):fail(env.BAD_GOAL_CLAIM,number,'核验路径未达到目标')
                else:paths.append(tuple(path))
                active=False
            else:fail(env.PARSE_ERROR,number,'非法 RECHECK 事件')
            continue
        if active:fail(env.UNEXPECTED_EVENT,number,'核验区段内出现其他事件')
        if re.fullmatch(r'GOAL n[0-9]+',line):goal_seen=True
        if '</think>' in line:closed=True
        clean.append(line)
    if active:fail(env.INCOMPLETE_TRACE,len(lines),'核验区段未闭合')
    limit=kwargs.get('max_events')
    if limit is not None and sum(bool(l.strip()) and l.strip() not in ['<think>','</think>'] for l in lines)>limit:
        fail(env.PARSE_ERROR,len(lines),'事件数超过上限，核验事件计入预算')
    result=native.verify_record(world,'\n'.join(clean),**kwargs)
    result.text=text
    result.recheck_metrics={'blocks':len(paths),'steps':steps,'unique_paths':len(set(paths)),
        'redundant_blocks':len(paths)-len(set(paths)),'valid':error is None,'main_search_valid':result.search_valid}
    if error is not None:
        result.search_valid=False;result.first_error=result.first_error or error
        result.hack=result.plan_valid is True;result.hack_kind='invalid_recheck' if result.hack else None
        result.compatible_modes=[]
        if error.code==env.PARSE_ERROR:result.parse_status=native.PARSE_ERROR_STATUS
    return result
