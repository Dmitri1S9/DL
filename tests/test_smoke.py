"""End-to-end smoke test of the pipeline wiring, using the mock paths only.

Runs prepare -> generate(mock) -> evaluate(no ASR) and checks the artifacts line
up with the contract. Skips automatically if the audio deps (soundfile/librosa)
are not installed, so it never blocks a bare checkout.
"""

import pytest

pytest.importorskip('numpy')
pytest.importorskip('soundfile')
pytest.importorskip('librosa')


def test_mock_pipeline_runs(tmp_path, monkeypatch):
    from core import config
    from data import prepare as prepare_mod
    from evaluation.evaluate import evaluate
    from model.synthesize import generate_for_manifest

    # Redirect all outputs into tmp_path so the test is hermetic.
    manifest = tmp_path / 'test_manifest.jsonl'
    reference_dir = tmp_path / 'reference'
    generated_dir = tmp_path / 'generated'
    monkeypatch.setattr(config, 'TEST_MANIFEST', manifest)
    monkeypatch.setattr(config, 'REFERENCE_DIR', reference_dir)

    # 1) prepare (mock): writes the manifest + silent reference wavs
    prepare_mod.prepare(mock=True)
    assert manifest.exists()
    n_items = len(manifest.read_text().splitlines())
    assert n_items == len(prepare_mod._MOCK_SENTENCES)

    # 2) generate (mock): one wav per manifest item
    generate_for_manifest(manifest, generated_dir, mock=True)
    assert len(list(generated_dir.glob('*.wav'))) == n_items

    # 3) evaluate (no ASR): MCD computed, WER/CER skipped
    result = evaluate(generated_dir, manifest, label='smoke', compute_asr=False)
    assert result.n == n_items
    assert result.wer is None
    # MCD is left unchecked here: the mock references are silent, so MCD is not
    # meaningful (and may be NaN/None). It's exercised on the real path instead.
