import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / 'example_workflows' / 'NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json'


def load():
    with WF.open(encoding='utf-8') as f:
        return json.load(f)


def test_new_masked_extension_chain_structure():
    data = load()
    nodes = data['nodes']
    types = [n['type'] for n in nodes]
    assert types.count('MiniMaxH3ReferenceToVideo') == 6
    assert types.count('MiniMaxH3StartCheckpointMaskedContext') == 1
    assert types.count('MiniMaxH3GeneratedAVMaskedContext') == 5
    assert types.count('MiniMaxH3CheckpointSavePath') == 7  # starter + 6 extensions
    assert types.count('MiniMaxH3ResumeCheckpointLatent') == 5
    assert types.count('MiniMaxH3OptionalReferenceImage') == 2
    assert types.count('MiniMaxH3OptionalStartFrame') == 1
    assert types.count('MiniMaxH3ExtensionStartMode') == 1
    assert types.count('MiniMaxH3StartCanvasSelector') == 1
    assert types.count('MiniMaxH3ImageToVideo') == 1
    assert types.count('MiniMaxH3CheckpointTrigger') == 1
    assert types.count('MiniMaxH3AssembleStarterOrExtensionCheckpoints') == 1
    assert types.count('UNETLoader') == 2  # REF2VA extensions + FL2VA starter
    assert 'MiniMaxH3MotionContext' not in types
    assert 'ImageBatchExtendWithOverlap' not in types
    assert 'AudioConcat' not in types
    assert types.count('VAEDecode') == 7  # generated starter + 6 extension previews
    assert types.count('VAEDecodeAudio') == 7
    assert types.count('VHS_VideoCombine') == 7


def test_new_masked_extension_chain_seeds_and_controls():
    data = load()
    nodes = data['nodes']
    noises = [n for n in nodes if n['type'] == 'RandomNoise']
    assert len(noises) == 7
    assert all(n['widgets_values'][1] == 'fixed' for n in noises)

    active = next(n for n in nodes if n.get('title') == 'GLOBAL ACTIVE EXTENSION COUNT — MATCH HIGHEST ENABLED (DEFAULT 1)')
    resume = next(n for n in nodes if n.get('title') == 'GLOBAL RESUME FROM EXTENSION — 0 = NORMAL')
    mode = next(n for n in nodes if n['type'] == 'MiniMaxH3ExtensionStartMode')
    assert active['widgets_values'][0] == 1
    assert resume['widgets_values'][0] == 0
    assert mode['widgets_values'][0] in ('start with T2V/I2V', 'Start from existing video')

    final = next(n for n in nodes if n['type'] == 'MiniMaxH3AssembleStarterOrExtensionCheckpoints')
    assert final['widgets_values'][1] == 'h3_extension_checkpoints/starter'
    assert final['widgets_values'][2] == 'h3_extension_checkpoints/clip'
    assert final['widgets_values'][4] == 39
    assert final['widgets_values'][5] == 39


def test_one_global_start_switch_drives_context_canvas_and_assembler():
    data = load()
    nodes = {n['id']: n for n in data['nodes']}
    mode = next(n for n in nodes.values() if n['type'] == 'MiniMaxH3ExtensionStartMode')
    links = {l[0]: l for l in data['links']}
    destinations = {(links[lid][3], nodes[links[lid][3]]['type']) for lid in mode['outputs'][0]['links']}
    assert {t for _, t in destinations} == {
        'MiniMaxH3StartCheckpointMaskedContext',
        'MiniMaxH3StartCanvasSelector',
        'MiniMaxH3AssembleStarterOrExtensionCheckpoints',
    }


def test_generated_starter_is_fl2va_checkpointed_and_optional_i2v():
    data = load()
    nodes = data['nodes']
    starter_model = next(n for n in nodes if n['type'] == 'UNETLoader' and n['widgets_values'][0] == 'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    assert starter_model['widgets_values'][0] == 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'
    starter = next(n for n in nodes if n['type'] == 'MiniMaxH3ImageToVideo')
    first = next(n for n in nodes if n['type'] == 'MiniMaxH3OptionalStartFrame')
    save = next(n for n in nodes if n.get('title') == 'CHECKPOINT — GENERATED STARTER (DISK / RESUME SAFE)')
    assert starter['inputs'][2]['link'] is not None
    assert isinstance(first['widgets_values'][0], bool)  # preserve workflow author's chosen T2V/I2V default
    assert save['widgets_values'] == ['h3_extension_checkpoints/starter', 1]


def test_workflow_uses_default_titles_for_loaders_and_attention_patches():
    data = load()
    nodes = data['nodes']
    loaders = [n for n in nodes if n['type'] == 'UNETLoader']
    attention = [n for n in nodes if n['type'] in ('PathchSageAttentionKJ', 'ModelAttentionBackend')]
    refs = [n for n in nodes if n['type'] == 'MiniMaxH3ReferenceToVideo']
    assert len(loaders) == 2 and all(n.get('title') is None for n in loaders)
    assert len(attention) == 2 and all(n.get('title') is None for n in attention)
    assert sorted(n.get('title') for n in refs) == [
        f'EXTENSION {i} — MiniMax H3 Reference to Video' for i in range(1, 7)
    ]



def test_optional_extension_switch_defaults_to_one_active_group():
    data = load()
    nodes = data['nodes']
    switch = next(n for n in nodes if n['type'] == 'Fast Groups Bypasser (rgthree)' and n.get('properties', {}).get('matchTitle') == 'OPTIONAL EXTENSION')
    assert switch['title'] == 'OPTIONAL EXTENSIONS 2–6 — ENABLE SEQUENTIALLY'

    groups = {g['id']: g for g in data.get('groups', [])}
    assert [groups[i]['title'] for i in range(4, 9)] == [
        f'OPTIONAL EXTENSION — EXTENSION {i - 2}' for i in range(4, 9)
    ]

    # Every generation/checkpoint node for Extensions 2–6 is bypassed by default.
    optional_ids = set()
    for n in nodes:
        title = str(n.get('title') or '')
        if any(title.startswith(prefix) for prefix in (
            'EXTENSION 2', 'EXTENSION 3', 'EXTENSION 4', 'EXTENSION 5', 'EXTENSION 6',
            'CHECKPOINT — Extension 2', 'CHECKPOINT — Extension 3', 'CHECKPOINT — Extension 4',
            'CHECKPOINT — Extension 5', 'CHECKPOINT — Extension 6',
        )):
            optional_ids.add(n['id'])
    # Include untitled sampler helper nodes by group bounding boxes, excluding the global optional-ref area.
    for gid in range(4, 9):
        x, y, w, h = groups[gid]['bounding']
        for n in nodes:
            px, py = n.get('pos', [10**9, 10**9])
            if x <= px <= x + w and y <= py <= y + h:
                optional_ids.add(n['id'])
    optional_ids.discard(switch['id'])
    assert optional_ids
    assert all(next(n for n in nodes if n['id'] == nid).get('mode', 0) == 4 for nid in optional_ids)

    refs = [n for n in nodes if n['type'] == 'MiniMaxH3OptionalReferenceImage']
    assert all(n.get('mode', 0) == 0 for n in refs)

def test_workflow_links_are_internally_consistent():
    data = load()
    nodes = {n['id']: n for n in data['nodes']}
    links = {l[0]: l for l in data['links']}
    for n in nodes.values():
        for slot, inp in enumerate(n.get('inputs', [])):
            lid = inp.get('link')
            if lid is None:
                continue
            assert lid in links
            link = links[lid]
            assert link[3] == n['id'] and link[4] == slot
        for slot, out in enumerate(n.get('outputs', [])):
            for lid in out.get('links') or []:
                assert lid in links
                link = links[lid]
                assert link[1] == n['id'] and link[2] == slot


def test_start_mode_is_one_user_choice_and_controls_source_group():
    data = load()
    nodes = {n['id']: n for n in data['nodes']}
    mode = next(n for n in nodes.values() if n['type'] == 'MiniMaxH3ExtensionStartMode')
    assert mode['widgets_values'][0] in ('start with T2V/I2V', 'Start from existing video')

    # No second source-mode Fast Groups Bypasser remains in the workflow.
    assert not any(
        n['type'] == 'Fast Groups Bypasser (rgthree)'
        and str(n.get('title', '')).startswith('SOURCE VIDEO BRANCH')
        for n in nodes.values()
    )

    source_groups = [g for g in data.get('groups', []) if str(g.get('title', '')).startswith('START SOURCE VIDEO')]
    assert len(source_groups) == 1
    source_group = source_groups[0]

    # VHS loader + crop live inside the source group; no hidden per-node gate metadata remains.
    gx, gy, gw, gh = source_group['bounding']
    for nid in (99, 100):
        n = nodes[nid]
        x, y = n['pos']
        assert gx <= x <= gx + gw and gy <= y <= gy + gh
        assert 'h3_start_branch_gate' not in n.get('properties', {})
        assert 'h3_start_active_mode' not in n.get('properties', {})

    frontend = (ROOT / 'js' / 'h3_extension_start_mode.js').read_text()
    assert 'START_T2V_I2V = "start with T2V/I2V"' in frontend
    assert 'SOURCE_GROUP_PREFIX = "START SOURCE VIDEO"' in frontend
    assert 'MODE_BYPASS = 4' in frontend
    assert 'candidate.mode = desiredMode' in frontend


def test_start_mode_user_labels_map_to_internal_modes():
    source = (ROOT / 'nodes.py').read_text()
    assert 'START_T2V_I2V = "start with T2V/I2V"' in source
    assert 'START_EXISTING_VIDEO = "Start from existing video"' in source
    assert 'return ("generate_starter",)' in source
    assert 'return ("load_video",)' in source


def test_extension_chain_is_disk_backed_between_samplers_and_vhs_previews():
    data = load()
    nodes = {n['id']: n for n in data['nodes']}
    links = {l[0]: l for l in data['links']}
    saves = [n for n in nodes.values() if n['type'] == 'MiniMaxH3CheckpointSavePath']
    assert len(saves) == 7
    assert all(len(n['outputs']) == 1 and n['outputs'][0]['type'] == 'STRING' for n in saves)

    resumes = [n for n in nodes.values() if n['type'] == 'MiniMaxH3ResumeCheckpointLatent']
    assert len(resumes) == 5
    for n in resumes:
        signal = next(i for i in n['inputs'] if i['name'] == 'checkpoint_signal')
        link = links[signal['link']]
        assert nodes[link[1]]['type'] == 'MiniMaxH3CheckpointSavePath'
        assert link[5] == 'STRING'

    start = next(n for n in nodes.values() if n['type'] == 'MiniMaxH3StartCheckpointMaskedContext')
    starter_signal = next(i for i in start['inputs'] if i['name'] == 'starter_checkpoint_signal')
    assert nodes[links[starter_signal['link']][1]]['type'] == 'MiniMaxH3CheckpointSavePath'

    loaders = [n for n in nodes.values() if n['type'] == 'MiniMaxH3CheckpointLoadPath']
    assert len(loaders) == 7
    for n in loaders:
        link = links[n['inputs'][0]['link']]
        assert nodes[link[1]]['type'] == 'MiniMaxH3CheckpointSavePath'
        assert link[5] == 'STRING'
