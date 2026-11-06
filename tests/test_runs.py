from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import pytest
from sentinel_bc.config import ROOT, load_config
from sentinel_bc.runs import RunManager, atomic_json, worker
from sentinel_bc.storage import ReviewStore
from sentinel_bc.training import train


def test_benchmark_cannot_be_overwritten():
    for root in [None, ROOT, ROOT/'artifacts', ROOT/'results/full']:
        with pytest.raises(ValueError): train("quick",root)


def test_atomic_json_retries_windows_reader_conflict(tmp_path, monkeypatch):
    import sentinel_bc.runs as module
    replace = module.os.replace
    calls = []
    def transient(source, target):
        calls.append(1)
        if len(calls)<3: raise PermissionError("reader holds file")
        return replace(source,target)
    monkeypatch.setattr(module.os,"replace",transient)
    atomic_json(tmp_path/'state.json',{"complete":True})
    assert json.loads((tmp_path/'state.json').read_text()) == {"complete":True}
    assert len(calls)==3 and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize("cancelled", [True,False])
def test_cancelled_and_failed_workers_never_become_selectable(tmp_path, monkeypatch, cancelled):
    import sentinel_bc.runs as module
    monkeypatch.setattr(module,"TRAINING_LOCK",tmp_path/'worker.lock')
    directory = tmp_path/'runs/run_interrupted'; directory.mkdir(parents=True)
    atomic_json(directory/'run.json',{"run_id":directory.name,"status":"QUEUED","mode":"quick","started_unix":time.time(),"pid":os.getpid()})
    atomic_json(directory/'config.json',load_config()); (directory/'ready').touch()
    if cancelled: (directory/'cancel.request').touch()
    else: (directory/'dataset.csv').write_text('label\nunknown\n')
    worker(directory)
    state = json.loads((directory/'run.json').read_text())
    assert state['status'] == ('CANCELLED' if cancelled else 'FAILED')
    manager = RunManager(tmp_path/'runs',tmp_path/'no-benchmark')
    assert manager.completed() == []
    with pytest.raises(ValueError,match='completed'): manager.activate(directory.name)
    assert manager.active.engine is None


def test_project_lock_prevents_overlapping_registries(tmp_path, monkeypatch):
    import sentinel_bc.runs as module
    lock = tmp_path/'worker.lock'; monkeypatch.setattr(module,'TRAINING_LOCK',lock)
    atomic_json(lock,{'run_id':'run_busy','pid':os.getpid(),'directory':str(tmp_path/'other')})
    manager = RunManager(tmp_path/'runs',tmp_path/'no-benchmark')
    with pytest.raises(ValueError,match='already running'): manager.start('quick',epochs=2)


def test_interrupted_process_is_marked_and_cannot_activate(tmp_path, monkeypatch):
    import sentinel_bc.runs as module
    monkeypatch.setattr(module,'TRAINING_LOCK',tmp_path/'worker.lock')
    directory=tmp_path/'runs/run_dead'; directory.mkdir(parents=True)
    atomic_json(directory/'run.json',{'run_id':'run_dead','mode':'quick','status':'RUNNING','pid':99999999,'started_unix':time.time()})
    manager=RunManager(tmp_path/'runs',tmp_path/'absent')
    assert manager.status('run_dead')['status']=='INTERRUPTED'
    with pytest.raises(ValueError): manager.activate('run_dead')


def test_simultaneous_referrals_create_one_case_and_keep_original(tmp_path):
    store=ReviewStore(tmp_path/'review.sqlite3')
    record={'model_version':'test-version','decision':'HUMAN_REVIEW_REQUIRED','predicted_class':'MALIGNANT','p_malignant':.9,'reasons':['MALIGNANT_PROBABILITY']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        cases=list(pool.map(lambda _:store.refer_once(record,{'mean radius':10.0}),range(8)))
    assert len(set(cases))==1 and len(store.list_cases())==1
    store.review(cases[0],'BENIGN','Teaching override')
    assert store.inspect(cases[0])['original_output']==record
    assert len(store.inspect(cases[0])['audit'])==2


def test_stale_inference_version_is_rejected(tmp_path):
    from sentinel_bc.inference import Engine, ActiveModel
    if not (ROOT/'artifacts/full/manifest.json').exists(): pytest.skip('Local artifact test')
    engine=Engine(); active=ActiveModel()
    active.set('benchmark',engine,{'version':engine.metadata['version']},ROOT/'results/full')
    with pytest.raises(ValueError,match='changed'): active.require('old-version')
    with pytest.raises(ValueError,match='changed'): active.require('')
