"""Export automatic frame/interval labels; never turn a rating into a training pass."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deploy.fr3_sharpa_control.recording import write_json
from .dagger import SLICES, GROUPS
from .protocol import FPS


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def horizon_mask(frame_mask, stamps, interventions, index, horizon=50):
    """Sample supervision on the policy clock, including 100 Hz human dispatches."""
    result = np.zeros((horizon,58), dtype=bool)
    if interventions[index] < 0:
        return result
    stamps = np.asarray(stamps, dtype=np.int64)
    interventions = np.asarray(interventions)
    previous = index
    for offset in range(horizon):
        desired = int(stamps[index]) + offset * 1_000_000_000 // FPS
        if desired > int(stamps[-1]) + offset:
            break
        # Older 30 Hz logs truncate each period to integer nanoseconds.
        target = int(np.searchsorted(stamps, desired + offset, side='right')) - 1
        segment = slice(previous, target + 1)
        if (np.any(interventions[segment] != interventions[index])
                or np.any(np.diff(stamps[segment]) > 50_000_000)
                or np.any(np.diff(stamps[segment]) <= 0)
                or not np.all(np.any(frame_mask[segment] & frame_mask[index], axis=1))):
            break
        result[offset] = frame_mask[target] & frame_mask[index]
        previous = target
    return result


def export(directory):
    directory = Path(directory)
    raw = directory/'dagger'
    events = list(rows(raw/'events.jsonl'))
    admission = {}
    for event in events:
        if event.get('event') == 'gateway_status':
            admission[event['sequence']] = set(event['accepted_sides']) if not event['faults'] else set()
    fields = ['frame_index','sequence','source_timestamp_ns','relative_seconds','intervention_id',
              'mode','phase','stage', *GROUPS,'gateway_left_admitted','gateway_right_admitted']
    intervals = {}
    frame_mask, stamps, interventions = [], [], []
    sequences = {}
    origin = None
    temporary = raw/'labels.csv.tmp'
    with temporary.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(rows(raw/'frames.jsonl')):
            if frame['frame_index'] != index or len(frame['action_owner']) != 4:
                raise ValueError('DAgger frame index/ownership discontinuity')
            stamp = frame['source_timestamp_ns']
            if origin is None:
                origin = stamp
            if stamps and stamp <= stamps[-1]:
                raise ValueError('DAgger source clock is not increasing')
            sequences[frame['sequence']] = dict(frame_index=index, source_timestamp_ns=stamp)
            owners = dict(zip(GROUPS,frame['action_owner'],strict=True))
            accepted = admission.get(frame['sequence'],set())
            mask = np.zeros(58,dtype=bool)
            # SDK send success is recorded as a dispatched target. FR3 targets
            # additionally need their matching gateway sequence, not an old ACK.
            for group, part in zip(GROUPS,SLICES,strict=True):
                if owners[group] == 'human' and (group.endswith('hand') or group.split('_')[0] in accepted):
                    mask[part] = True
            frame_mask.append(mask)
            stamps.append(stamp)
            interventions.append(frame['intervention_id'] if frame['intervention_id'] is not None else -1)
            writer.writerow(dict(frame_index=index,sequence=frame['sequence'],source_timestamp_ns=stamp,
                relative_seconds=f'{(stamp-origin)/1e9:.6f}',intervention_id=frame['intervention_id'],
                mode=frame['mode'],phase=frame['phase'],stage=frame['stage'],**owners,
                gateway_left_admitted='left' in accepted,gateway_right_admitted='right' in accepted))
    temporary.replace(raw/'labels.csv')
    for event in events:
        if event.get('event') not in ('human_start','human_end'):
            continue
        identifier = str(event['intervention_id'])
        interval = intervals.setdefault(identifier,dict(intervention_id=event['intervention_id']))
        side = 'start' if event['event']=='human_start' else 'end'
        interval[side] = {**sequences.get(event['sequence'],{}),
                          'monotonic_ns':event['monotonic_ns'], 'reason':event.get('reason'),
                          'incomplete':event.get('incomplete',False)}
    # Store compact per-frame masks. Horizon consumers must call horizon_mask
    # rather than blindly copy the base-frame mask over all future actions.
    with (raw/'supervision.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, frame_mask=np.asarray(frame_mask,dtype=bool).reshape(-1,58),
            source_timestamp_ns=np.asarray(stamps,dtype=np.int64),
            intervention_id=np.asarray(interventions,dtype=np.int64))
    (raw/'supervision.npz.tmp').replace(raw/'supervision.npz')
    write_json(raw/'interventions.json',list(intervals.values()))
    report = dict(schema='sharpa.dagger.labels.v1',status='COMPLETE',frames=len(stamps),
        interventions=len(intervals),raw_log_finalized=any(e.get('event')=='run_end' for e in events),
        action_basis='software_dispatched_target',training_validation='UNVERIFIED',
        supervision='human-only; match gateway sequence for arms; mask horizon boundaries',
        model_manifest=str(directory/'dagger_manifest.json'))
    write_json(raw/'labels_export.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(export(parser.parse_args().directory),ensure_ascii=False))


if __name__ == '__main__':
    main()
