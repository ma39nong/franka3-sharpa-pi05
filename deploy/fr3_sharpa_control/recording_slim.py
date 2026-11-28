"""Lossless-in-meaning pruning: omit diagnostic timings and unchanged calibration repeats."""
import json

DIAGNOSTIC_TOPICS=frozenset({'/recording/dispatch_timing'})


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if hasattr(value,'get_fields_and_field_types'):
        return {k:plain(getattr(value,k)) for k in value.get_fields_and_field_types()}
    if hasattr(value,'__slots__'):
        return {k:plain(getattr(value,k)) for k in value.__slots__}
    if isinstance(value,(str,int,float,bool,bytes)) or value is None:return value
    return [plain(x) for x in value]


def camera_info_signature(message):
    data=plain(message)
    # CameraInfo's capture timestamp repeats the already-recorded image time.
    # Keep frame_id, all calibration fields, ROI/binning and any actual changes.
    data=dict(data);data['header']=dict(data.get('header',{}));data['header'].pop('stamp',None)
    return json.dumps(data,sort_keys=True,separators=(',',':'))


def slim_profile(profile):
    from copy import deepcopy
    profile=deepcopy(profile)
    for device in profile.get('devices',[]):
        device['channels']=[ch for ch in device.get('channels',[])
                            if ch.get('topic') not in DIAGNOSTIC_TOPICS and ch.get('adapter')!='dagger_timing']
        for ch in device['channels']:
            if ch.get('type')=='sensor_msgs/msg/CameraInfo':
                ch['storage_policy']='first_snapshot_and_calibration_changes'
    return profile
