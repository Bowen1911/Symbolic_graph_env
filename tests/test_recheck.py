"""Rechecking is opt-in and every repeated transition remains verified."""
from symplan import env,qa,search,verify
from symplan.recheck import block,PROTOCOL
from tests.fixtures import example_world


def example():
    world=example_world();solution=search.solve(world,'forward-bfs')
    record=qa.render_qa(world,solution,view='search',thinking='on',prompt_style=qa.PROMPT_STRUCTURED)
    return world,record.assistant,block(world,solution.plan)


def test_recheck_isolation_and_counts():
    world,clean,b=example()
    for count in [1,4]:
        text=clean.replace('</think>',(b+'\n')*count+'</think>')
        r=verify.verify_record(world,text,kind='search',protocol=PROTOCOL)
        assert r.passed and r.recheck_metrics['blocks']==count
        assert r.recheck_metrics['redundant_blocks']==count-1
        assert r.to_dict()['verification_policy']==PROTOCOL
        assert not verify.verify_record(world,text,kind='search').passed
    assert verify.verify_record(world,clean,kind='search',protocol=PROTOCOL).passed


def test_recheck_invalid_steps_placement_and_termination():
    world,clean,b=example();text=clean.replace('</think>',b+'\n</think>')
    rid=b.splitlines()[1].split()[1]
    corrupt=[text.replace('RECHECK_STEP '+rid,'RECHECK_STEP r999',1),
        text.replace('RECHECK_END',''),text.replace('RECHECK_END','unrecognized\nRECHECK_END'),
        text.replace('RECHECK_BEGIN STATE','RECHECK_BEGIN STATE ZZZ:1',1),
        clean.replace('<think>','<think>\n'+b),clean+'\n'+b]
    for value in corrupt:
        r=verify.verify_record(world,value,kind='search',protocol=PROTOCOL)
        assert not r.passed and r.first_error is not None


def test_recheck_whitespace_and_event_limit():
    world,clean,b=example();text=clean.replace('</think>',b.replace('\n','\n\n  ')+'\n</think>')
    assert verify.verify_record(world,text,kind='search',protocol=PROTOCOL).passed
    assert not verify.verify_record(world,text,kind='search',protocol=PROTOCOL,max_events=2).passed


def test_tagged_protocol_and_invalid_check(tmp_path,monkeypatch):
    import json
    from symplan import cli
    world,clean,b=example();text=clean.replace('</think>',b+'\n</think>')
    w=tmp_path/'worlds.jsonl';w.write_text(json.dumps(world.to_dict())+'\n')
    source=tmp_path/'predictions.jsonl'
    values=[{'world_id':world.world_id,'record_id':'valid','text':text,'generation_wall_seconds':1.5},
            {'world_id':world.world_id,'record_id':'invalid','text':text.replace('RECHECK_END','broken\nRECHECK_END')}]
    source.write_text(''.join(json.dumps(r)+'\n' for r in values))
    monkeypatch.setattr(cli,'RESULT_ROOT',tmp_path/'result')
    assert cli.main(['verify','--worlds',str(w),'--input',str(source),'--kind','search','--protocol',PROTOCOL,'--tag','probe'])==0
    root=tmp_path/'result/probe';summary=json.loads((root/'summary.json').read_text())
    assert summary['verification_policy']==PROTOCOL and summary['metrics']['passed']['passed']==1
    rows=[json.loads(l) for l in (root/'qa.jsonl').read_text().splitlines()]
    assert rows[0]['generation_wall_seconds']==1.5 and rows[1]['verification']['hack'] is True
