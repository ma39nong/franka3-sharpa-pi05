"""Local recording-only static TF augmentation; never publishes to the robot graph."""
from copy import deepcopy
import math
import numpy as np


def transform_matrix(transform):
    t=transform['translation'];q=transform['rotation']
    x,y,z,w=(float(q[k]) for k in ('x','y','z','w'))
    if not np.isfinite([x,y,z,w]).all() or abs(x*x+y*y+z*z+w*w-1)>1e-5:
        raise ValueError('Invalid unit quaternion in static TF')
    result=np.eye(4)
    result[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                   [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                   [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    result[:3,3]=[t[k] for k in ('x','y','z')]
    return result


def matrix_transform(matrix):
    m=np.asarray(matrix,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-8):
        raise ValueError('Invalid homogeneous extrinsic matrix')
    r=m[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1,atol=1e-6):
        raise ValueError('Invalid proper rotation')
    # Symmetric eigenproblem avoids singular trace branches near 180 degrees.
    k=np.array([[r[0,0]-r[1,1]-r[2,2],r[1,0]+r[0,1],r[2,0]+r[0,2],r[2,1]-r[1,2]],
                [r[1,0]+r[0,1],r[1,1]-r[0,0]-r[2,2],r[2,1]+r[1,2],r[0,2]-r[2,0]],
                [r[2,0]+r[0,2],r[2,1]+r[1,2],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
                [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],r.trace()]])/3
    values,vectors=np.linalg.eigh(k);q=vectors[:,values.argmax()]
    if q[3]<0:q=-q
    return dict(translation=dict(zip(('x','y','z'),m[:3,3].tolist())),rotation=dict(zip(('x','y','z','w'),q.tolist())))


def validate_specs(specs):
    seen=set()
    for spec in specs:
        parent,child=spec['parent_frame'],spec['child_frame']
        if not parent or not child or parent==child or child in seen:
            raise ValueError('Invalid/duplicate calibrated static TF frame')
        seen.add(child);matrix_transform(spec['matrix'])


def merge_static(message, specs, stamp_ns):
    """Preserve optical conventions by attaching the camera's existing local root.

    T_flange_root = T_flange_optical @ inverse(T_root_optical).
    Existing camera-internal transforms are retained. Only the root's old
    incoming edge is replaced; an optical frame never acquires two parents.
    """
    validate_specs(specs)
    edges={x['child_frame_id']:deepcopy(x) for x in message['transforms']}
    for spec in specs:
        child=spec['child_frame'];camera=child.split('_',1)[0]+'_'
        root=child;root_to_child=np.eye(4);visited=set()
        while root in edges and edges[root]['header']['frame_id'].startswith(camera):
            if root in visited:raise ValueError('Cycle in camera static TF')
            visited.add(root);edge=edges[root]
            root_to_child=transform_matrix(edge['transform'])@root_to_child
            root=edge['header']['frame_id']
        target=np.asarray(spec['matrix'])@np.linalg.inv(root_to_child)
        edges[root]=dict(header=dict(stamp=dict(sec=stamp_ns//10**9,nanosec=stamp_ns%10**9),frame_id=spec['parent_frame']),
                         child_frame_id=root,transform=matrix_transform(target))
    for child in edges:
        visited=set();node=child
        while node in edges:
            if node in visited:raise ValueError('Calibrated TF would create a cycle')
            visited.add(node);node=edges[node]['header']['frame_id']
    return {'transforms':list(edges.values())}
