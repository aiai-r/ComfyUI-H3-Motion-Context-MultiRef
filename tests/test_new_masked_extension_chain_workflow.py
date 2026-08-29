import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / 'example_workflows' / 'NEW - AV Extension.json'


def load():
    return json.loads(WF.read_text(encoding='utf-8'))


def _nodes(data):
    return {n['id']: n for n in data['nodes']}


def _links(data):
    return {l[0]: l for l in data['links']}


def _input(node, name):
    return next(i for i in node.get('inputs', []) if i['name'] == name)


def test_live_extension_structure_is_checkpoint_free_and_ref2va_only():
    data = load(); types = [n['type'] for n in data['nodes']]
    assert types.count('MiniMaxH3ReferenceToVideo') == 7  # starter + 6 extensions
    assert types.count('MiniMaxH3GeneratedAVMaskedContext') == 5
    assert types.count('MiniMaxH3StartMaskedContext') == 1
    assert types.count('MiniMaxH3AVExtensionController') == 1
    assert types.count('MiniMaxH3StreamLiveExtensionAVToVHS') == 1
    assert types.count('MiniMaxH3AssembleLiveExtensionAV') == 0
    assert types.count('VHS_VideoCombine') == 7  # starter + six clip previews only
    assert types.count('MiniMaxH3CustomKeyframes') >= 1
    assert types.count('UNETLoader') == 1
    assert not any('Checkpoint' in t for t in types)
    assert not any('rgthree' in t.lower() for t in types)
    assert 'MiniMaxH3ImageToVideo' not in types
    assert 'MiniMaxH3OptionalReferenceImage' not in types
    loader = next(n for n in data['nodes'] if n['type'] == 'UNETLoader')
    assert 'ref2va' in loader['widgets_values'][0]


def test_controller_defaults_and_backend_links():
    data=load(); nodes=_nodes(data); links=_links(data)
    c=next(n for n in nodes.values() if n['type']=='MiniMaxH3AVExtensionController')
    assert c['widgets_values'] == ['Existing Video', 1, 8, 'All Active']
    dest_types = {nodes[links[lid][3]]['type'] for lid in c['outputs'][0]['links']}
    assert {'MiniMaxH3StartCanvasSelector','MiniMaxH3StartMaskedContext','MiniMaxH3StreamLiveExtensionAVToVHS'} <= dest_types
    for cid in [103,201,301,401,501,601]:
        lid=_input(nodes[cid],'audio_feather_ticks')['link']; l=links[lid]
        assert l[1]==c['id'] and l[2]==2


def test_starter_is_reference_to_video_plus_frame1_keyframe():
    data=load(); nodes=_nodes(data); links=_links(data)
    starter=nodes[973]
    assert starter['type']=='MiniMaxH3ReferenceToVideo'
    kf=nodes[1022]
    assert kf['type']=='MiniMaxH3CustomKeyframes'
    assert kf['widgets_values'][:2] == ['{"count":1,"positions":[1]}','1-based']
    assert links[_input(kf,'conditioning')['link']][1] == starter['id']
    assert links[_input(kf,'latent')['link']][1] == starter['id']
    assert links[_input(kf,'keyframe_image_1')['link']][1] == 970
    assert links[_input(nodes[975],'conditioning')['link']][1] == kf['id']


def test_reference_images_feed_every_reference_to_video_node_directly():
    data=load(); nodes=_nodes(data); links=_links(data)
    ref_loaders=sorted(
        (n for n in nodes.values() if n.get('title','').startswith('REFERENCE IMAGE ')),
        key=lambda n:int(n['title'].split('REFERENCE IMAGE ',1)[1].split(' ',1)[0]),
    )
    assert len(ref_loaders)==9
    assert all(n['type']=='LoadImage' for n in ref_loaders)
    assert all(n.get('mode')==4 for n in ref_loaders[2:])
    assert all(n['widgets_values'][0]=='' for n in ref_loaders[2:])
    for n in nodes.values():
        if n['type']!='MiniMaxH3ReferenceToVideo': continue
        sockets=[_input(n,f'ref_images.ref_image_{i}') for i in range(9)]
        assert all(socket['link'] is not None for socket in sockets)
        assert [links[socket['link']][1] for socket in sockets] == [
            loader['id'] for loader in ref_loaders]


def test_optional_reference_audio_uses_one_reroute_per_slot_and_feeds_all_r2v():
    data=load(); nodes=_nodes(data); links=_links(data)
    loaders=sorted(
        [n for n in nodes.values() if n['type']=='LoadAudio' and n.get('title','').startswith('REFERENCE AUDIO ')],
        key=lambda n:n['title'],
    )
    reroutes=sorted(
        [n for n in nodes.values() if n['type']=='Reroute' and n.get('title','').startswith('REFERENCE AUDIO ')],
        key=lambda n:n['title'],
    )
    assert len(loaders)==2 and len(reroutes)==2
    assert all(n.get('mode')==4 for n in loaders)
    assert all(n['widgets_values'][0]=='' for n in loaders)
    for loader,reroute in zip(loaders,reroutes):
        assert len(loader['outputs'][0]['links'])==1
        lr=links[loader['outputs'][0]['links'][0]]
        assert lr[3]==reroute['id']
        assert len(reroute['outputs'][0]['links'])==7
    for n in nodes.values():
        if n['type']!='MiniMaxH3ReferenceToVideo': continue
        for name in ('ref_audios.ref_audio_0','ref_audios.ref_audio_1'):
            l=links[_input(n,name)['link']]
            assert nodes[l[1]]['type']=='Reroute'


def test_generated_chain_is_direct_live_latent_between_samplers():
    data=load(); nodes=_nodes(data); links=_links(data)
    for ctx,prev in zip([201,301,401,501,601],[124,214,314,414,514]):
        assert links[_input(nodes[ctx],'source_latent')['link']][1] == prev
    final=next(n for n in nodes.values() if n['type']=='MiniMaxH3StreamLiveExtensionAVToVHS')
    for i,sid in enumerate([124,214,314,414,514,614],1):
        assert links[_input(final,f'extension_{i}')['link']][1] == sid
    assert links[_input(final,'starter_latent')['link']][1] == 977


def test_final_output_streams_directly_to_vhs_without_full_image_output():
    data=load(); nodes=_nodes(data); links=_links(data)
    final=next(n for n in nodes.values() if n['type']=='MiniMaxH3StreamLiveExtensionAVToVHS')
    assert final['outputs'][0]['type']=='VHS_FILENAMES'
    assert final['widgets_values_named']['save_output'] is True
    assert final['widgets_values_named']['context_frames'] == 39
    assert final['widgets_values_named']['video_overlap_frames'] == 39
    assert not any(n.get('title')=='Final Output — VHS Video Combine' for n in nodes.values())


def test_controller_managed_groups_have_single_ownership_and_matching_defaults():
    data=load(); owners=defaultdict(list)
    managed=[]
    for g in data.get('groups',[]):
        meta=(g.get('flags') or {}).get('h3_control')
        if not meta: continue
        b=g['bounding']; members=[]
        for n in data['nodes']:
            cx=n['pos'][0]+n['size'][0]/2; cy=n['pos'][1]+n['size'][1]/2
            if b[0] <= cx <= b[0]+b[2] and b[1] <= cy <= b[1]+b[3]:
                members.append(n); owners[n['id']].append(meta)
        managed.append((meta,members))
    assert not {nid:v for nid,v in owners.items() if len(v)>1}
    assert sum(m['role']=='extension' for m,_ in managed)==6
    assert sum(m['role']=='extension_preview' for m,_ in managed)==6
    assert sum(m['role']=='starter_preview' for m,_ in managed)==1
    assert sum(m['role']=='starter_core' for m,_ in managed)==1
    assert sum(m['role']=='i2v_keyframe' for m,_ in managed)==1
    assert sum(m['role']=='source_video' for m,_ in managed)==1
    for meta,members in managed:
        role=meta['role']; idx=meta.get('index')
        enabled = role=='source_video' or (role=='extension' and idx==1) or (role=='extension_preview' and idx==1)
        assert all(n.get('mode',0)==(0 if enabled else 4) for n in members)


def test_workflow_links_are_internally_consistent():
    data=load(); nodes=_nodes(data); links=_links(data)
    for n in nodes.values():
        for slot,inp in enumerate(n.get('inputs',[])):
            lid=inp.get('link')
            if lid is None: continue
            assert lid in links and links[lid][3]==n['id'] and links[lid][4]==slot
        for slot,out in enumerate(n.get('outputs',[])):
            for lid in out.get('links') or []:
                assert lid in links and links[lid][1]==n['id'] and links[lid][2]==slot


def test_controller_frontend_uses_real_bypass_and_metadata_groups():
    src=(ROOT/'js'/'h3_av_extension_controller.js').read_text()
    assert 'MODE_BYPASS = 4' in src
    assert 'group.recomputeInsideNodes' in src
    assert 'group?.flags?.h3_control' in src
    assert 'beforeQueued' in src
    assert 'afterConfigureGraph' in src
    assert 'setInterval' in src
    assert 'nativeWouldSkip' in src
    assert 'this.updateParameters(params, true)' in src
    assert '.updateSource(' not in src
    assert 'app.canvas?.isDragging' in src
    assert 'expected exactly one H3 AV Extension Controller' in src
    assert 'unsupported h3_control schema' in src
    assert 'belongs to multiple H3-controlled groups' in src
