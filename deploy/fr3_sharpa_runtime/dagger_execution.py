"""Inference/stage coordinator for the opt-in persistent DAgger SDK owner."""
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from .dagger import fresh_prediction_budget_s
from .dagger_io import AuditWriter, rpc as dagger_rpc
from .motion_worker import MotionWorker
from .rtc_stream import RTCSettings, require_capabilities
from .runtime import SharpaRuntime
from .task_control import TaskControl, TaskLease, TaskServer
from .task_stages import stage
from .stage_handoff import (HandoffObservationExpired, handoff_prediction_budget,
                            handoff_retry_budget)


def execute_dagger(args):
    from . import run
    from .shutdown import protect_disable_confirmation
    required=('SHARPA_RECORD_SOCKET','SHARPA_DAGGER_SOCKET','SHARPA_DAGGER_RUN_ID','SHARPA_DAGGER_DIRECTORY')
    if not all(os.environ.get(key) for key in required) or getattr(args,'sharpa_backend','sdk_direct') != 'sdk_direct':
        raise ValueError('DAgger需要控制台创建独立运行身份、录制器和SDK执行器')
    root = Path(__file__).resolve().parents[2]
    directory = Path(args.execution_trace_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    staged = args.task_mode != 'none'
    task = task_server = lease = worker = inputs = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dagger-prediction')
    future = None
    audit = AuditWriter(directory/'dagger-inference')
    failure = None
    accepted = 0
    def emit(**event):
        audit.put({'monotonic_ns':time.monotonic_ns(), **event})
    try:
        if staged:
            task = TaskControl(os.environ['SHARPA_TASK_RUN_ID'], args.initial_stage,
                               profile=args.task_mode, emit=emit)
            task.dagger_mode = True
            task_server = TaskServer(os.environ['SHARPA_TASK_SOCKET'], task)
            lease = TaskLease(task, lambda: os.kill(os.getpid(), signal.SIGINT))
            lease.start()
        setup = args.ros_overlay/'setup.bash'
        if not setup.is_file():
            raise FileNotFoundError(setup)
        env_prefix = (f'source {shlex.quote(str(setup))}; '
            f'export PYTHONPATH={shlex.quote(str(root))}:{shlex.quote(str(run.GELLO_REPO))}:'
            f'{shlex.quote(str(run.GELLO_REPO/"ros_ws/src/teleop_core"))}:"${{PYTHONPATH:-}}"; ')
        input_log = Path(os.environ.get('SHARPA_MERGED_LOG',str(directory/'dagger-inputs.log'))).open('ab', buffering=0)
        try:
            inputs = subprocess.Popen(['pixi','run','--locked','-e','ros2','bash','-lc', env_prefix+
                f'exec ros2 launch {shlex.quote(str(root/"deploy/fr3_sharpa_runtime/dagger_inputs.launch.py"))}'],
                cwd=run.ROS_PROJECT, stdout=input_log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            input_log.close()
        worker = MotionWorker(['pixi','run','--locked','-e','ros2','bash','-lc',env_prefix+
            'exec python -m deploy.fr3_sharpa_runtime.sdk_direct_motion_ros --enable-hardware'],
            cwd=run.ROS_PROJECT, protocol_fd=True)
        worker.request({'operation':'ready'}, timeout=30)
        run._start_observation_worker(args.ros_overlay)
        if args.initialization == 'home':
            from .home_pose import home_target, load_home
            from .policy_prepare import HOME_TOLERANCE_RAD
            arms, metadata = load_home(args.home_poses, args.home_task, hand_path=args.hand_home)
            feedback = worker.request({'operation':'feedback'},timeout=6)
            report = worker.request(dict(operation='prepare', target=home_target(arms, feedback['state']).tolist(),
                speed_rad_s=args.prepare_speed_rad_s, tolerance_rad=HOME_TOLERANCE_RAD, home_recovery=True),timeout=180)
            emit(event='home_complete', report=report, home=metadata)
        from .startup_gate import wait_for_cameras
        wait_for_cameras()
        runtime = SharpaRuntime(args.policy_url, **{k:True for k in ('train_rtc','paper_rtc') if getattr(args,k,False)})
        metadata = runtime.inspect_metadata(require_rtc=args.broker_mode=='rtc')
        settings = RTCSettings(args.rtc_initial_delay_steps,args.rtc_delay_buffer_size,
                               args.rtc_prefix_attention_schedule,args.rtc_max_guidance_weight)
        delays = settings.history()
        trained = getattr(args,'train_rtc',False)
        if args.broker_mode == 'rtc' and not trained:
            if getattr(args, 'paper_rtc', False):
                from .rtc_stream import require_paper_capabilities
                require_paper_capabilities(metadata)
            else:
                require_capabilities(metadata)
        if trained:
            from .train_rtc_contract import require_train_capabilities
            require_train_capabilities(metadata)
            if getattr(args, 'train_paper_rtc', False):
                from .train_rtc_contract import require_train_paper_capabilities
                require_train_paper_capabilities(metadata)
            if not 1 <= settings.initial_delay_steps <= min(25, metadata['train_rtc']['max_delay']):
                raise ValueError('DAgger Train RTC delay exceeds trained contract')
        emit(event='model_identity', metadata=metadata)
        from deploy.fr3_sharpa_control.recording import write_json
        write_json(Path(os.environ['SHARPA_DAGGER_DIRECTORY'])/'dagger_manifest.json', dict(
            model=metadata, model_version=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest(),
            task_mode=args.task_mode, action_order=['left_arm_7','left_hand_22','right_arm_7','right_hand_22'],
            action_basis='absolute_software_dispatched_target',fps=30,
            policy_control_hz=30,human_control_hz=100,
            gello_config_sha256=hashlib.sha256((run.GELLO_REPO/'config/modes/gello.yaml').read_bytes()).hexdigest()))
        initial_owner = worker.request(dict(operation='dagger_start',broker_mode=args.broker_mode,
            execute_steps=args.execute_steps,speed_scale=args.execute_speed_scale,
            approach_speed_rad_s=args.prepare_speed_rad_s,online_task_handoff=staged),timeout=20)
        if task:
            task.ready()
        else:
            worker.request({'operation':'dagger_request_policy','generation':initial_owner['dagger']['generation']},timeout=2)
        last_generation = None
        requested_task_version = None
        switch_boundary = switch_ticket = None
        def predict(ticket, generation, context, boundary=None):
            prompt = ticket['prompt'] if ticket else args.prompt
            timeout = context['prefix_attention_horizon']/30 if context else 5.
            if boundary is not None:
                timeout = min(timeout, handoff_retry_budget(boundary, time.monotonic()))
            observation = run.live_observation(args.ros_overlay,prompt,timeout_s=timeout)
            budget = (handoff_prediction_budget(boundary, observation['_captured_monotonic_ns']/1e9, time.monotonic())
                      if boundary is not None else timeout if context else fresh_prediction_budget_s(
                          observation['_captured_monotonic_ns'], time.monotonic_ns()))
            if budget <= 0:
                raise ValueError('DAgger observation expired before inference')
            infer = getattr(runtime, 'infer_task_chunk', runtime.infer_chunk)
            result = infer(observation, rtc_context=context, response_timeout=budget)
            return dict(result=result, receipt=runtime.last_receipt if context else None,
                        captured=observation['_captured_monotonic_ns'],ticket=ticket,
                        generation=generation, prompt=prompt,
                        completed_monotonic_ns=time.monotonic_ns(),
                        inference_timing=dict(getattr(runtime, 'last_timing', {})))
        while True:
            if inputs.poll() is not None:
                raise RuntimeError('DAgger input-only launcher exited; see dagger-inputs.log')
            audit.check()
            if future is not None and future.ticket[2] is None:
                # A fresh prediction has a strict age budget. Read authority
                # without queuing a status request behind the 30 Hz dispatch.
                status = {'dagger': dagger_rpc(os.environ['SHARPA_DAGGER_SOCKET'],
                    os.environ['SHARPA_DAGGER_RUN_ID'], renew_contact=False)}
            else:
                status = worker.request({'operation':'dagger_status'},timeout=2)
            authority = status['dagger']
            generation, mode = authority['generation'], authority['mode']
            if task:
                with task.lock:
                    task.data['dagger_suspended'] = mode not in ('POLICY','RESUMING','IDLE') or (authority.get('intervention_id') is not None and mode != 'POLICY')
                    task.dagger_suspended = task.data['dagger_suspended']
                    task.data['inference_status'] = 'dagger_'+mode.lower()
                    if generation != last_generation and mode in ('HOLDING','HOLD','HUMAN_LEFT'):
                        switch_boundary = switch_ticket = None
                        task.data['stage_version'] += 1
                        if task.data['active_stage']:
                            task.data.update(requested_stage=task.data.get('manual_selected_stage') or task.data['active_stage'], pending_request_id=None)
                        if task.data['execution_state'] in ('SWITCH_REQUESTED','SWITCHING'):
                            task.transition('RUNNING','DAgger takeover cancels pending stage change')
                        requested_task_version = task.data['stage_version']
                        task.event('dagger_invalidated',generation=generation)
                    state = task.data['execution_state']
                    if state == 'STOPPING':
                        break
                    if state == 'SWITCH_REQUESTED' and not task.dagger_suspended:
                        task.transition('SWITCHING','DAgger stage boundary; fresh prediction required')
                        state = 'SWITCHING'
                    if state in ('RESUMING','SWITCHING') and not task.dagger_suspended and requested_task_version != task.data['stage_version']:
                        switch_boundary = switch_ticket = None
                        if state == 'SWITCHING' and mode == 'POLICY' and args.broker_mode == 'rtc':
                            response = worker.request(dict(operation='dagger_stage_switch', generation=generation,
                                task_epoch=task.data['stage_version'], request_id=task.data['pending_request_id'],
                                finish_chunk=args.task_mode == 'tomato-in-hand119'), timeout=2)
                            if response.get('rejected'):
                                continue
                            switch_boundary = response.get('boundary')
                            if switch_boundary:
                                delays = settings.history()
                                emit(event='online_switch_boundary', boundary=switch_boundary,
                                     stage_version=task.data['stage_version'])
                        if switch_boundary is None:
                            # First start / unfinished entry / human recovery retain the old admission.
                            response = worker.request({'operation':'dagger_request_policy','generation':generation},timeout=2)
                        if response.get('rejected'):
                            continue
                        requested_task_version = task.data['stage_version']
                        last_generation = generation
                        continue
                    if switch_ticket is not None and status.get('applied_token') == switch_boundary['token']:
                        with task.installation(switch_ticket) as admitted:
                            if admitted:
                                task.activate(switch_ticket, f'dagger-handoff:{generation}:{accepted}')
                                accepted += 1
                                emit(event='online_switch_applied', token=switch_boundary['token'],
                                     stage_version=switch_ticket['stage_version'])
                        switch_boundary = switch_ticket = None
            last_generation = generation
            if future is not None and future.done():
                ticket, prediction_generation, context = future.ticket
                valid = prediction_generation == generation and mode in ('POLICY','RESUMING')
                if task:
                    valid = valid and task.valid(ticket)
                try:
                    result = future.result()
                except (HandoffObservationExpired, TimeoutError) as error:
                    if valid and switch_boundary is not None and switch_ticket is None:
                        handoff_retry_budget(switch_boundary, time.monotonic())
                        emit(event='online_switch_prediction_retry', reason=str(error),
                             token=switch_boundary['token'], deadline=switch_boundary['deadline'])
                        future = None
                        continue
                    if valid:
                        raise
                    result = None
                except BaseException:
                    if valid:
                        raise
                    result = None
                future = None
                if not valid:
                    emit(event='obsolete_prediction_discarded',generation=prediction_generation)
                    continue
                if switch_boundary is not None:
                    try:
                        handoff_prediction_budget(switch_boundary, result['captured']/1e9, time.monotonic())
                    except HandoffObservationExpired as error:
                        emit(event='online_switch_prediction_retry', reason=str(error),
                             token=switch_boundary['token'], deadline=switch_boundary['deadline'])
                        continue
                metadata, actions, _ = result['result']
                install_request_ns = time.monotonic_ns()
                emit(event='prediction_install_requested', generation=generation,
                     captured_monotonic_ns=result['captured'],
                     prediction_completed_monotonic_ns=result['completed_monotonic_ns'],
                     install_request_monotonic_ns=install_request_ns,
                     coordinator_wait_ms=(install_request_ns-result['completed_monotonic_ns'])/1e6,
                     inference_timing=result['inference_timing'])
                request = dict(operation='dagger_install',actions=actions.tolist(),generation=generation,
                    captured_monotonic_ns=result['captured'], receipt=result['receipt'],
                    install_request_monotonic_ns=install_request_ns,
                    prediction_clip_rad=args.prediction_clip_rad, prompt=result['prompt'],
                    stage=ticket['stage_id'] if ticket else None)
                if switch_boundary is not None:
                    request.update(operation='dagger_stage_handoff', ticket={**switch_boundary,
                        'captured_monotonic_s':result['captured']/1e9})
                if task:
                    with task.installation(ticket) as admitted:
                        installed = worker.request(request,timeout=2) if admitted else {'rejected':True}
                        if not installed.get('rejected') and switch_boundary is None:
                            task.activate(ticket, f'dagger:{generation}:{accepted}')
                else:
                    installed = worker.request(request,timeout=2)
                if installed.get('rejected') and switch_boundary is not None:
                    handoff_retry_budget(switch_boundary, time.monotonic())
                    emit(event='online_switch_prediction_retry', reason=installed.get('reason','handover rejected'),
                         token=switch_boundary['token'], deadline=switch_boundary['deadline'])
                if not installed.get('rejected'):
                    if switch_boundary is not None:
                        switch_ticket = ticket
                        emit(event='online_switch_armed', generation=generation, stage=request['stage'],
                             installation=installed, captured_monotonic_ns=result['captured'], actions=actions.tolist())
                        continue
                    accepted += 1
                    delays.append(installed['skipped'])
                    emit(event='prediction_installed',generation=generation,stage=request['stage'],
                         prompt=request['prompt'],receipt=result['receipt'],installation=installed,
                         captured_monotonic_ns=result['captured'],actions=actions.tolist())
                continue
            eligible = mode in ('RESUMING','POLICY')
            if task:
                eligible = eligible and task.data['execution_state'] in ('RESUMING','SWITCHING','RUNNING')
            if eligible and future is None:
                due = mode == 'RESUMING' or (status.get('policy_done',False) if args.broker_mode=='serial'
                    else status['consumed_since_install'] >= args.execute_steps)
                if switch_boundary is not None:
                    handoff = status.get('handoff') or {}
                    due = switch_ticket is None and (
                        'finish_chunk_at' not in switch_boundary or handoff.get('tail_finished') is True)
                if due:
                    if args.num_chunks and accepted >= args.num_chunks and mode=='POLICY':
                        if task and task.data['execution_state']=='RUNNING':
                            task.transition('STOPPING','configured chunk count reached')
                        break
                    context = None
                    if args.broker_mode=='rtc' and mode=='POLICY' and switch_boundary is None:
                        response = worker.request(dict(operation='dagger_snapshot',generation=generation,
                            delay_steps=settings.initial_delay_steps if trained else max(delays),
                            schedule=settings.prefix_attention_schedule,weight=settings.max_guidance_weight,
                            hard_prefix=trained),timeout=2)
                        if response.get('rejected'):
                            continue
                        context = response['context']
                    ticket = task.ticket() if task else None
                    if switch_boundary is not None:
                        handoff_retry_budget(switch_boundary, time.monotonic())
                    future = pool.submit(predict,ticket,generation,context,
                                         dict(switch_boundary) if switch_boundary else None)
                    future.ticket = (ticket,generation,context)
            time.sleep(.01)
        return dict(status='COMPLETE',accepted_chunks=accepted,dagger_directory=str(directory/'dagger'))
    except BaseException as error:
        failure = str(error) or type(error).__name__
        if task:
            task.fault(error)
        raise
    finally:
        with protect_disable_confirmation():
            if lease:
                lease.close()
            try:
                if worker:
                    worker.close()
            finally:
                if inputs and inputs.poll() is None:
                    os.killpg(inputs.pid,signal.SIGINT)
                    try:
                        inputs.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(inputs.pid,signal.SIGTERM)  # Input-only process group.
                pool.shutdown(wait=False,cancel_futures=True)
                if task:
                    if task.data['execution_state'] == 'STOPPING' and failure is None:
                        task.transition('COMPLETED' if task.data.get('finish_requested') else 'STOPPED','DAgger execution ended; hardware session cleanup follows')
                    (directory/'task-final.json').write_text(json.dumps(task.snapshot(),ensure_ascii=False,indent=2))
                if task_server:
                    task_server.close()
                audit.close()
