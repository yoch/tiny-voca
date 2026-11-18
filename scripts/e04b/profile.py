import collections,json,pathlib,sys
import numpy as np
import onnxruntime as ort
root=pathlib.Path(sys.argv[1])
out=pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
result={}
for name in ('full-dense','full-dft','full-radix2','native/stage'):
    for threads in (1,4):
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=threads
        opts.inter_op_num_threads=1
        opts.enable_profiling=True
        opts.profile_file_prefix=str(out/(name.split('/')[-1]+'-'+str(threads)))
        sess=ort.InferenceSession(str(root/(name+'.onnx')),opts,providers=['CPUExecutionProvider'])
        data=dict(np.load(root/'full-dense.npz'))
        feeds={x.name:data[x.name] for x in sess.get_inputs()}
        for _ in range(8): sess.run(None,feeds)
        events=json.loads(pathlib.Path(sess.end_profiling()).read_text())
        groups=collections.defaultdict(lambda:[0,0,0,0,0])
        offset=sorted(x['ts'] for x in events if x['name']=='model_run')[3]
        for e in events:
            if e['cat']!='Node' or not e['name'].endswith('_kernel_time') or e['ts']<offset:continue
            node=e['name']; a=e['args']
            key=(node,a.get('op_name'))
            g=groups[key];g[0]+=e['dur'];g[1]+=1
            for i,k in enumerate(('activation_size','output_size','parameter_size'),2):g[i]+=int(a.get(k,0))
        rows=[dict(node=n,op=o,ms=v[0]/5000,calls=v[1]/5,input_bytes=v[2]//5,output_bytes=v[3]//5,parameter_bytes=v[4]//5) for (n,o),v in groups.items()]
        result[f'{name}:{threads}']=sorted(rows,key=lambda x:x['ms'],reverse=True)
        print(name,threads,[(x['node'],round(x['ms'],2)) for x in result[f'{name}:{threads}'][:10]],flush=True)
(out/'profile.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
