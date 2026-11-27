"""Snapshot existing workstation evidence without starting devices or promoting calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import yaml
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gello',type=Path,default=Path('/home/descfly/ZH/gello_upper_body_teleop'))
    parser.add_argument('--handeye',type=Path,default=Path('/home/descfly/LTT/franka-handeye'))
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    output=repo/'.deployment/mcap-assets';output.mkdir(parents=True,exist_ok=True)
    index=[]
    def copy(source,name):
        source=Path(source)
        if not source.is_file():return None
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        checksum=hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=checksum:
            raise FileExistsError(f'Existing evidence differs; retain old snapshot and choose a new destination: {target}')
        shutil.copyfile(source,target)
        index.append(dict(source=str(source),path=name,sha256=checksum))
        return target
    sources=[('config/modes/gello.yaml','calibration/gello/gello.yaml'),
        ('config/workcell/current.yaml','configs/workcell.yaml'),
        ('data_collection/config/cameras.yaml','configs/cameras-source.yaml'),
        ('data_collection/config/record_gello_sharpa_tactile.yaml','configs/record-tactile-source.yaml'),
        ('config/litchibot/sharpa_schema.json','models/source/sharpa_schema.json'),
        ('config/litchibot/dexterous_hand_cmc_mapping.json','calibration/hands/cmc_mapping.json'),
        ('config/sharpa_runtime.json','configs/sharpa_runtime.json'),
        ('ops/run/run_gello_full_teleop.sh','configs/full_teleop_entry.sh'),
        ('ops/run/start_sharpa_arm_recording_with_tactile.sh','configs/tactile_record_entry.sh')]
    for source,name in sources:copy(args.gello/source,name)
    env={}
    for line in (args.gello/'docker/.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\'')
    manus=Path(env.get('MANUS_CALIBRATION_DIRECTORY',args.gello/'portable_deps/litchi_hardware/calibration/metaglove'))
    for path in sorted(manus.glob('*.mcal')):copy(path,'calibration/manus/'+path.name)
    parameters={}
    base=args.gello/'ros_ws/src/teleop_camera_bringup/config'
    for cam,original,override in [('cam0','camera_primary_params.yaml','camera_cam0_rgb_params.yaml'),('cam1','camera_cam1_global_params.yaml','camera_cam1_mjpg_params.yaml'),('cam2','camera_secondary_params.yaml',None)]:
        copy(base/original,f'configs/cameras/{cam}-source.yaml')
        active=repo/'.deployment'/override if override else base/original
        path=copy(active,f'configs/cameras/{cam}-deployment.yaml')
        if path:parameters[cam]=yaml.safe_load(path.read_text())
    for path in sorted((args.gello/'ros_ws/src/lychee_fr3_description').rglob('*')):
        if path.is_file() and path.suffix in ('.xacro','.yaml','.urdf','.stl','.dae','.xml','.STL'):
            copy(path,'models/source/fr3_description/'+str(path.relative_to(args.gello/'ros_ws/src/lychee_fr3_description')))
    vendor=args.gello/'portable_deps/litchi_hardware/ros2/src/manus_sharpa_teleop'
    for path in sorted((vendor/'sharpa_wave_description').rglob('*')):
        if path.is_file():copy(path,'models/source/sharpa_wave_description/'+str(path.relative_to(vendor/'sharpa_wave_description')))
    for relative in ('manus_sharpa_teleop/manus_sharpa_teleop/retarget_node.py',
                     'manus_sharpa_teleop/manus_sharpa_teleop/v4_retargeter.py',
                     'manus_sharpa_teleop/vendor/retarget_v4/VERSION.json'):
        copy(vendor/relative,'calibration/retarget/'+Path(relative).name)
    # VERSION.json lists these basenames; keep the real dependencies adjacent
    # in the archive, rather than archiving dangling vendor references.
    retarget=vendor/'manus_sharpa_teleop/vendor/retarget_v4'
    version=json.loads((retarget/'VERSION.json').read_text())
    for name in version['files']:
        if Path(name).name!=name or not (retarget/'include'/name).is_file():
            raise ValueError(f'Missing/invalid vendor retarget dependency: {name}')
        copy(retarget/'include'/name,'calibration/retarget/'+name)
    handeye=[]
    for side in ('left','right'):
        candidates=sorted((args.handeye/f'runs/franka_handeye/live/{side}').glob(f'*/calibration_{side}.json'))
        if not candidates:continue
        source=candidates[-1];data=json.loads(source.read_text())
        transform=np.asarray(data['T_base_camera']);inverse=np.asarray(data['T_camera_base'])
        assert np.allclose(transform@inverse,np.eye(4),atol=1e-8)
        assert np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-8)
        copy(source,f'calibration/handeye/{side}.json')
        copy(source.parent/'samples.json',f'calibration/handeye/{side}-samples.json')
        copy(source.parent/'calibration.json',f'calibration/handeye/{side}-original.json')
        handeye.append(dict(side=side,parent_frame=data['parent_frame'],child_frame=data['child_frame'],
            session=data['session']['id'],source_resolution=[data['camera']['width'],data['camera']['height']],
            quality_pass=data['quality_pass'],independent_validation_pass=data['independent_validation_pass'],
            deployment_eligible=data['deployment_eligible'],camera_identity=data['camera_identity'],matrix_inverse_check=True))
    profile=yaml.safe_load((repo/'deploy/fr3_sharpa_control/mcap-profile.example.yaml').read_text())
    profile['session_id']='workstation-prepared-requires-live-confirmation'
    profile['calibration_root']=str(output/'calibration')
    # Do not accept an unexpanded model source tree as a deployment-ready model.
    profile['model_root']=str(output/'models/materialized-required')
    profile['clock_mapping']['sync_state']='unverified'
    identities=yaml.safe_load((args.gello/'data_collection/config/cameras.yaml').read_text())['camera_bringup']
    for ch in profile['devices'][0]['channels']:
        if ch['type']=='foxglove_msgs/msg/CompressedVideo':
            cam=ch['source_topic'].split('/')[1];params=parameters[cam]
            ch.update(width=params['color_width'],height=params['color_height'],fps=params['color_fps'],
                serial_number=identities[cam]['serial_number'],upstream_codec=params['color_format'],time_domain=params['time_domain'])
            ch['color_source']='REQUIRED: RGB transfer/primaries evidence remains unavailable'
    target=repo/'.deployment/mcap-recording.yaml'
    if target.exists():target=repo/'.deployment/mcap-recording.prepared.yaml'
    target.write_text(yaml.safe_dump(profile,allow_unicode=True,sort_keys=False))
    timing=subprocess.check_output(['timedatectl','show','-p','NTP','-p','NTPSynchronized'],text=True).strip()
    inventory=dict(schema_version=1,files=index,camera_parameters=parameters,handeye=handeye,
        host_ntp_observation=timing,observation_note='Host status only; does not prove inter-device synchronization accuracy',
        profile=str(target),ready_for_capture=False,
        outstanding=['Materialize and verify archived FR3+Sharpa description sources and their dependencies','Confirm head camera serial/mount and independently validate handeye',
          'Obtain current CameraInfo for all three cameras; no rescaling of historical 1280x800 intrinsics',
          'Obtain wrist camera extrinsics if required by the archival contract','Verify RGB color interpretation',
          'Measure/record SDK device-to-host timestamp mapping and uncertainty; never mark locked from host NTP alone',
          'Resolve tactile/depth scope differences between the original collector and deployment'])
    (output/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(profile=str(target),inventory=str(output/'inventory.json'),copied_files=len(index),ready_for_capture=False),ensure_ascii=False))


if __name__=='__main__':main()
