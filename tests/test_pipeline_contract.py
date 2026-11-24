"""Exercise actual episode/reward wiring with local fake tools, never a model."""

import builtins
from contextlib import ExitStack
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def no_model_imports(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'peft'}:
        raise AssertionError(f'Offline test attempted model dependency import: {name}')
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__
with patch('builtins.__import__', side_effect=no_model_imports):
    Q = importlib.import_module('ops.grpo_q4')
    # grpo_eval uses a bare sibling import. Bind it to the module under test so
    # global switches and all service tripwires refer to the same module.
    with patch.dict(sys.modules, {'grpo_q4': Q}), patch.object(Q, 'NEED_LP', [False]):
        E = importlib.import_module('ops.grpo_eval')
    LEGACY_EVAL = importlib.import_module('ops.eval_q4')
    with patch.dict(sys.modules, {'eval_q4': LEGACY_EVAL}):
        RAW_PROBE = importlib.import_module('ops.q4_probe_raw')


class FakeTurn(SimpleNamespace):
    def __init__(self, **kwargs):
        defaults = dict(content='', resp_text='', reasoning='', action=None,
                        obs=None, extra=[], truncated=False)
        defaults.update(kwargs)
        super().__init__(**defaults)


class FakeWorkspace:
    def __init__(self, root, runner):
        self.root = Path(root)

    def reset(self):
        # Each test owns a fresh TemporaryDirectory; no removal is necessary.
        self.root.mkdir(parents=True, exist_ok=False)


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = next(s for s in Q.contract_tasks.specs()
                         if s['name'] == 'contract-scoped-section-edit')
        self.schemas = [{'type': 'function', 'function': {
            'name': 'repair_scoped_section',
            'parameters': {'type': 'object', 'properties': {}}}}]
        self.old_prompt = Mock(side_effect=AssertionError('Old AGENT prompt invoked'))
        self.native = SimpleNamespace(
            native_tools=lambda: self.schemas,
            system_prompt=self.old_prompt,
            to_api_messages=lambda system, user, rounds: [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}],
            _BLOCK_RE=re.compile(r'<tool_call>(.*?)</tool_call>', re.S),
            _FN_RE=re.compile(r'<function=([^>]+)>'),
            _PARAM_RE=re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.S),
            _js_parse=json.loads, _js_stringify=json.dumps,
            recover_from_reasoning=lambda *args: None,
            execute=Mock(side_effect=self.repair),
        )
        self.M = dict(
            native=self.native, Turn=FakeTurn,
            Action=lambda name, args: SimpleNamespace(name=name, args=args),
            Task=SimpleNamespace,
            Config=lambda: SimpleNamespace(episode=SimpleNamespace(data_root=self.root)),
            Runner=Mock(return_value=object()), Workspace=FakeWorkspace,
            dialect=SimpleNamespace(prepare=lambda cfg, task, ws: task),
        )

    def repair(self, workspace, name, args, counters):
        self.assertEqual(name, 'repair_scoped_section')
        target = workspace.root / 'docs/agenda.md'
        before = target.read_bytes()
        after = before.replace(b'## Beta\nDraft status: pending\n', b'## Beta\n', 1)
        self.assertNotEqual(before, after)
        target.write_bytes(after)
        return 'The Beta status line was removed; Alpha and the archive remain unchanged.'

    def run_episode(self, final='ANSWER: docs/agenda.md', stop_type='eos'):
        replies = [
            '</think><tool_call><function=repair_scoped_section></function></tool_call>',
            '</think>' + final,
        ]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: replies[ids[0]])
        requests = []

        def render(tok, messages, tools):
            requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
            return [100 + len(requests)]

        completions = [dict(gen_ids=[i], lp_q4=[0.0], tps=0.0,
                            stop_type='eos' if i == 0 else stop_type) for i in range(2)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(Q, 'MAX_ROUNDS', 2))
            stack.enter_context(patch.object(Q, 'PROMPT_PROFILE', 'work-contract-v1'))
            stack.enter_context(patch.object(Q, 'render', side_effect=render))
            complete = stack.enter_context(patch.object(Q, 'complete', side_effect=completions))
            stack.enter_context(patch.dict(Q.os.environ, {'GRPO_RUNNER': 'offline-fake'}))
            stack.enter_context(patch('builtins.__import__', side_effect=no_model_imports))
            stack.enter_context(patch.object(Q.subprocess, 'Popen', side_effect=AssertionError('Process launch')))
            stack.enter_context(patch.object(Q.requests, 'post', side_effect=AssertionError('Network request')))
            result = Q.episode(self.M, tokenizer, Q.contract_tasks.FAMILY, self.spec, 'offline')
        self.assertEqual(complete.call_count, 2)
        self.old_prompt.assert_not_called()
        self.assertEqual(requests[0][1], self.schemas)
        self.assertIn(self.spec['prompt'], requests[0][0][1]['content'])
        self.assertTrue(requests[0][0][0]['content'].startswith('# Work contract v1'))
        return result, requests

    def test_actual_tool_edit_and_outcome_grader_accept_correct_result(self):
        result, _ = self.run_episode()
        self.assertEqual(result['reward'], 1.0)
        self.assertEqual(result['reason'], 'verified-outcome')
        self.assertEqual(result['stop'], 'answer')
        self.assertEqual(result['score_kind'], 'qualified-development')
        self.native.execute.assert_called_once()

    def test_correct_artifact_with_blank_final_is_failure(self):
        result, _ = self.run_episode(final='')
        self.assertEqual(result['reward'], 0.0)
        self.assertEqual(result['reason'], 'no-complete-final-reply')
        self.assertEqual(result['stop'], 'empty')

    def test_correct_artifact_with_truncated_final_is_failure(self):
        result, _ = self.run_episode(stop_type='limit')
        self.assertEqual(result['reward'], 0.0)
        self.assertEqual(result['reason'], 'no-complete-final-reply')
        self.assertEqual(result['stop'], 'truncated')

    def test_toolless_final_request_explains_capability_limit(self):
        _, requests = self.run_episode()
        system = requests[-1][0][0]['content']
        self.assertIsNone(requests[-1][1])
        self.assertIn(Q.FINAL_ROUND_NOTE, system)
        self.assertIn('tools are unavailable', system.lower())
        self.assertIn('Finish with exactly: ANSWER: docs/agenda.md', requests[-1][0][1]['content'])

    def test_logged_hashes_identify_actual_normal_and_final_system_messages(self):
        result, requests = self.run_episode()
        metadata = result['instruction']
        self.assertEqual(metadata['profile'], 'work-contract-v1')
        for key, index in (('system_sha256', 0), ('final_system_sha256', -1)):
            actual_text = requests[index][0][0]['content']
            self.assertEqual(metadata[key], hashlib.sha256(actual_text.encode('utf-8')).hexdigest())
        self.assertNotEqual(metadata['system_sha256'], metadata['final_system_sha256'])
        self.assertEqual(metadata['system_sha256'], Q.work_contract.prompt_metadata(
            tools=self.schemas)['rendered_sha256'])

    def test_legacy_training_refused_before_workspace_creation(self):
        workspace = Mock(side_effect=AssertionError('Legacy workspace created'))
        with patch.object(Q, 'LEGACY_DIAGNOSTIC', True), patch.object(Q, 'NEED_LP', [True]):
            with self.assertRaises(Q.reward_contract.UnqualifiedTask):
                Q.episode({**self.M, 'Workspace': workspace}, None, 'file_work', {}, 'blocked')
        workspace.assert_not_called()
        self.M['Runner'].assert_not_called()


class EntrypointGateTests(unittest.TestCase):
    def test_both_entrypoints_exit_before_models_services_or_artifacts(self):
        for module, args in ((Q, ['--run', 'must-not-exist']),
                             (E, ['--out', 'must-not-exist', '--adapters', 'base=absent.gguf'])):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', [module.__file__, *args]))
                    stack.enter_context(patch.object(module, 'ROOT', root))
                    stack.enter_context(patch.object(E, 'Q', Q))
                    stack.enter_context(patch('builtins.__import__', side_effect=no_model_imports))
                    server = stack.enter_context(patch.object(Q, 'start_server', side_effect=AssertionError('Server started')))
                    gym = stack.enter_context(patch.object(Q, 'gym', side_effect=AssertionError('Gym loaded')))
                    stack.enter_context(patch.object(Q.subprocess, 'Popen', side_effect=AssertionError('Process launched')))
                    stack.enter_context(patch.object(Q.requests, 'post', side_effect=AssertionError('Network request')))
                    with self.assertRaisesRegex(SystemExit, 'Model execution is disabled'):
                        module.main()
                    server.assert_not_called()
                    gym.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def check_legacy_entrypoint_disabled(self, module, argv):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', [module.__file__, *argv]))
                for key, path in (('ROOT', root), ('GYM', root / 'gym'),
                                  ('LOGS', root / 'logs'), ('TABLE', root / 'report.md')):
                    stack.enter_context(patch.object(LEGACY_EVAL, key, path))
                stack.enter_context(patch.object(RAW_PROBE, 'E', LEGACY_EVAL))
                stack.enter_context(patch.object(RAW_PROBE, 'OUT', root / 'probe'))
                stack.enter_context(patch('builtins.__import__', side_effect=no_model_imports))
                server = stack.enter_context(patch.object(LEGACY_EVAL, 'start_server',
                                                          side_effect=AssertionError('Server started')))
                exclusions = stack.enter_context(patch.object(RAW_PROBE, 'holdout_seeds',
                                                              side_effect=AssertionError('Private tasks read')))
                stack.enter_context(patch.object(LEGACY_EVAL.os, 'chdir', side_effect=AssertionError('Gym entered')))
                stack.enter_context(patch.object(LEGACY_EVAL.subprocess, 'Popen', side_effect=AssertionError('Process launched')))
                stack.enter_context(patch.object(LEGACY_EVAL.subprocess, 'run', side_effect=AssertionError('Command executed')))
                stack.enter_context(patch.object(LEGACY_EVAL.requests, 'post', side_effect=AssertionError('Network request')))
                stack.enter_context(patch.object(LEGACY_EVAL.requests, 'get', side_effect=AssertionError('Network request')))
                with self.assertRaisesRegex(SystemExit, 'Model execution is disabled'):
                    module.main()
                server.assert_not_called()
                exclusions.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_eval_q4_without_dry_run_requires_model_opt_in(self):
        self.check_legacy_entrypoint_disabled(LEGACY_EVAL, ['--size', '4b'])

    def test_raw_probe_requires_model_opt_in_before_holdout_or_runtime_access(self):
        self.check_legacy_entrypoint_disabled(RAW_PROBE, ['--arm', 'r3sft'])


class TrainingSignalTests(unittest.TestCase):
    def test_identical_rewards_skip_update_even_for_all_successes(self):
        # Supplying a module with no tensor/CUDA attributes means any attempted
        # numerical work fails. The real torch package is never imported.
        torch_stub = ModuleType('torch')

        def stub_import(name, *args, **kwargs):
            if name == 'torch':
                return torch_stub
            return no_model_imports(name, *args, **kwargs)

        model, optimizer = Mock(), Mock()
        groups = [[{'family': Q.contract_tasks.FAMILY, 'reward': reward,
                    'seqs': [{'prompt_ids': [1], 'gen_ids': [2], 'lp_q4': [-0.5]}]}
                   for _ in range(3)] for reward in (1.0, 0.0)]
        with patch.dict(sys.modules, {'torch': torch_stub}), patch('builtins.__import__', side_effect=stub_import):
            result = Q.train_step(model, optimizer, groups)
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['dead_groups'], 2)
        self.assertEqual(result['trained_seqs'], 0)
        self.assertEqual(result['trained_tokens'], 0)
        self.assertFalse(result['stepped'])
        self.assertEqual(model.mock_calls, [])
        self.assertEqual(optimizer.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
