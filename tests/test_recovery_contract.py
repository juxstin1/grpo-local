"""Recovery-wrapper and episode wiring tests with invented local tool actions.

No model, native tool implementation, network request, or subprocess executes.
The positive repair is computed from source bytes, never from expected output.
"""
from contextlib import ExitStack
import copy
import importlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ops import grpo_q4 as Q

RECOVERY = importlib.import_module('recovery_contract')
R = Q.reward_contract
TARGET = 'config/worker.json'
PROTECTED = 'notes/worker.md'
EDIT = {'path': TARGET, 'edits': [{'oldText': '"retries": 3', 'newText': '"retries": 5'}]}
READ = {'path': TARGET}


def recovery_spec():
    return next(spec for spec in Q.contract_tasks.specs() if spec['name'] == 'contract-edit-conflict-recovery')


def initialize(root, spec):
    root.mkdir(parents=True, exist_ok=True)
    Q.real_tasks.write_files(spec['files'], root)


def final_turn():
    return SimpleNamespace(content='ANSWER: ' + TARGET, resp_text='ANSWER: ' + TARGET,
                           action=None, extra=[], truncated=False)


def native_read(root):
    return '\n'.join((root / TARGET).read_text(encoding='utf-8').splitlines())


def native_edit(root, args=EDIT):
    path = root / args['path']
    data = path.read_bytes()
    for edit in args['edits']:
        old, new = edit['oldText'].encode(), edit['newText'].encode()
        if data.count(old) != 1:
            return 'Error: oldText must match exactly once.'
        data = data.replace(old, new, 1)
    path.write_bytes(data)
    return 'Applied 1 edit(s) to config/worker.json.'


class RecoverySessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        self.spec = recovery_spec()
        initialize(self.root, self.spec)
        self.session = RECOVERY.RecoverySession(self.spec, self.root)

    def execute(self, name, args, round_index, delegate=None, session=None):
        if delegate is None:
            delegate = lambda: native_read(self.root) if name == 'read_file_lines' else native_edit(self.root, args)
        return (session or self.session).execute(name, args, delegate, round_index=round_index)

    def recover(self, *, edit_delegate=None, verify=True):
        self.execute('read_file_lines', READ, 0)
        suppressed = Mock(side_effect=AssertionError('first edit must be injected, not dispatched'))
        conflict = self.execute('replace_file', EDIT, 1, suppressed)
        suppressed.assert_not_called()
        self.execute('read_file_lines', READ, 2)
        self.execute('replace_file', EDIT, 3, edit_delegate)
        if verify:
            self.execute('read_file_lines', READ, 4)
        return conflict

    def grade(self, session=None):
        return R.grade(self.spec, [final_turn()], self.root,
                       execution=self.session if session is None else session)

    def test_complete_recovery_passes_using_independent_source_edit(self):
        initial = (self.root / TARGET).read_bytes()
        protected = (self.root / PROTECTED).read_bytes()
        conflict = self.recover()
        self.assertIn('conflict', conflict.lower())
        self.assertEqual((self.root / TARGET).read_bytes(), initial.replace(b'"retries": 3', b'"retries": 5', 1))
        self.assertEqual((self.root / PROTECTED).read_bytes(), protected)
        self.assertEqual(self.session.verify(self.spec, self.root).status, 'pass')
        self.assertEqual(self.grade().reward, 1.0)

    def test_first_targeted_edit_is_an_error_without_file_or_delegate_effect(self):
        before = (self.root / TARGET).read_bytes()
        delegate = Mock(side_effect=AssertionError('unexpected write'))
        observation = self.execute('replace_file', json.dumps(EDIT), 0, delegate)
        self.assertIn('conflict', observation.lower())
        self.assertEqual((self.root / TARGET).read_bytes(), before)
        delegate.assert_not_called()
        self.assertEqual(self.grade().reward, 0.0)

    def test_noop_and_false_success_cannot_complete_recovery(self):
        self.recover(edit_delegate=lambda: 'Applied 1 edit(s) to config/worker.json.')
        self.assertEqual(self.grade().reward, 0.0)
        self.assertIn(b'"retries": 3', (self.root / TARGET).read_bytes())

    def test_wrong_artifact_cannot_pass_despite_success_observation(self):
        def wrong_edit():
            path = self.root / TARGET
            path.write_bytes(path.read_bytes().replace(b'"retries": 3', b'"retries": 4'))
            return 'Applied 1 edit(s) to config/worker.json.'
        self.recover(edit_delegate=wrong_edit)
        self.assertEqual(self.grade().reward, 0.0)

    def test_protected_file_change_fails_after_otherwise_complete_recovery(self):
        self.recover()
        (self.root / PROTECTED).write_text('Unrelated content removed.', encoding='utf-8')
        self.assertEqual(self.grade().reward, 0.0)

    def test_unrelated_read_cannot_replace_verified_target_fingerprint(self):
        self.recover()
        target = self.root / TARGET
        target.write_bytes(target.read_bytes().replace(b'"timeout_seconds": 20', b'"timeout_seconds": 21'))
        self.execute('read_file_lines', {'path': PROTECTED}, 5,
                     lambda: '\n'.join((self.root / PROTECTED).read_text(encoding='utf-8').splitlines()))
        self.assertEqual(self.session.verify(self.spec, self.root).status, 'fail')

    def test_out_of_scope_tool_is_blocked_and_cannot_be_laundered_by_later_recovery(self):
        delegate = Mock(side_effect=AssertionError('blocked tool dispatched'))
        self.execute('write_file', {'path': TARGET, 'content': 'forged'}, 0, delegate)
        delegate.assert_not_called()
        self.recover()
        self.assertEqual(self.grade().reward, 0.0)

    def test_correct_artifact_without_recovery_session_is_invalid_not_a_score(self):
        native_edit(self.root)
        with self.assertRaises(R.InvalidReward):
            R.grade(self.spec, [final_turn()], self.root)

    def test_receipt_dict_is_not_replayable_as_trusted_execution(self):
        self.recover()
        with self.assertRaises(R.InvalidReward):
            self.grade(self.session.receipt())

    def test_session_cannot_be_replayed_for_another_workspace_or_spec(self):
        self.recover()
        other = Path(self.temp.name) / 'other'
        initialize(other, self.spec)
        native_edit(other)
        with self.assertRaises(R.InvalidReward):
            R.grade(self.spec, [final_turn()], other, execution=self.session)
        changed = copy.deepcopy(self.spec)
        changed['prompt'] += ' Altered task definition.'
        with self.assertRaises(R.InvalidReward):
            R.grade(changed, [final_turn()], self.root, execution=self.session)

    def test_dispatch_exception_is_infrastructure_invalidity(self):
        def unavailable():
            raise OSError('invented file-tool transport failure')
        with self.assertRaises(R.InvalidReward):
            self.execute('read_file_lines', READ, 0, unavailable)

    def test_dispatch_nonstring_result_is_invalid(self):
        with self.assertRaises(R.InvalidReward):
            self.execute('read_file_lines', READ, 0, lambda: {'forged': 'observation'})

    def test_read_backend_error_string_invalidates_attempt(self):
        with self.assertRaises(R.InvalidReward):
            self.execute('read_file_lines', READ, 0,
                         lambda: RECOVERY.ERROR_PREFIX + 'ENOENT: transport cannot access workspace')
        with self.assertRaises(R.InvalidReward):
            self.grade()

    def test_write_backend_error_string_invalidates_attempt(self):
        self.execute('replace_file', EDIT, 0)
        self.execute('read_file_lines', READ, 1)
        with self.assertRaises(R.InvalidReward):
            self.execute('replace_file', EDIT, 2,
                         lambda: RECOVERY.ERROR_PREFIX + 'backend disconnected')
        with self.assertRaises(R.InvalidReward):
            self.grade()

    def test_bad_request_with_unchanged_bytes_allows_later_recovery(self):
        self.execute('replace_file', EDIT, 0)
        self.execute('read_file_lines', READ, 1)
        initial = (self.root / TARGET).read_bytes()
        self.execute('replace_file', EDIT, 2,
                     lambda: RECOVERY.ERROR_PREFIX + 'BAD_REQUEST: oldText mismatch')
        self.assertEqual((self.root / TARGET).read_bytes(), initial)
        self.assertFalse(self.session.receipt()['invalid'])
        self.execute('read_file_lines', READ, 3)
        self.execute('replace_file', EDIT, 4)
        self.execute('read_file_lines', READ, 5)
        self.assertEqual(self.grade().reward, 1.0)

    def test_unknown_edit_success_observation_is_invalid(self):
        self.execute('replace_file', EDIT, 0)
        self.execute('read_file_lines', READ, 1)
        with self.assertRaises(R.InvalidReward):
            self.execute('replace_file', EDIT, 2, lambda: 'Success! Everything is fixed.')

    def test_list_or_dict_path_is_learner_argument_failure_without_dispatch(self):
        for malformed in ([TARGET], {'path': TARGET}):
            delegate = Mock(side_effect=AssertionError('malformed arguments dispatched'))
            observation = self.execute('read_file_lines', {'path': malformed}, 0, delegate)
            self.assertIn('BAD_REQUEST:', observation)
            delegate.assert_not_called()
            self.assertFalse(self.session.receipt()['invalid'])
        self.recover()
        self.assertEqual(self.grade().reward, 1.0)

    def test_invalid_read_number_types_are_learner_errors_without_dispatch(self):
        for start in (True, '1', 1.5):
            delegate = Mock(side_effect=AssertionError('invalid numeric arguments dispatched'))
            observation = self.execute('read_file_lines', {'path': TARGET, 'startLine': start}, 0, delegate)
            self.assertIn('BAD_REQUEST:', observation)
            delegate.assert_not_called()
            self.assertFalse(self.session.receipt()['invalid'])
        self.recover()
        self.assertEqual(self.grade().reward, 1.0)

    def test_integral_float_read_arguments_can_prove_recovery_without_mutation(self):
        self.execute('replace_file', EDIT, 0)
        lines_args = {'path': TARGET, 'startLine': 1.0, 'endLine': 100.0}
        before = copy.deepcopy(lines_args)
        self.execute('read_file_lines', lines_args, 1)
        self.assertEqual(lines_args, before)
        self.execute('replace_file', EDIT, 2)
        chars_args = {'path': TARGET, 'offset': 0.0, 'length': 1000.0}
        before = copy.deepcopy(chars_args)
        self.execute('read_file_chars', chars_args, 3,
                     lambda: (self.root / TARGET).read_text(encoding='utf-8'))
        self.assertEqual(chars_args, before)
        self.assertEqual(self.grade().reward, 1.0)

    def test_same_round_prequeued_read_and_retry_cannot_earn_recovery_credit(self):
        self.execute('replace_file', EDIT, 1)
        self.execute('read_file_lines', READ, 1)
        self.execute('replace_file', EDIT, 1)
        self.execute('read_file_lines', READ, 1)
        self.assertEqual(self.grade().reward, 0.0)

    def test_read_after_retry_in_same_round_does_not_count_as_verification(self):
        self.execute('replace_file', EDIT, 0)
        self.execute('read_file_lines', READ, 1)
        self.execute('replace_file', EDIT, 2)
        self.execute('read_file_lines', READ, 2)
        self.assertEqual(self.grade().reward, 0.0)

    def test_correct_edit_without_later_readback_is_not_complete(self):
        self.recover(verify=False)
        self.assertIn(b'"retries": 5', (self.root / TARGET).read_bytes())
        self.assertEqual(self.grade().reward, 0.0)

    def test_prior_read_does_not_replace_post_conflict_inspection(self):
        self.execute('read_file_lines', READ, 0)
        self.execute('replace_file', EDIT, 1)
        self.execute('replace_file', EDIT, 2)
        self.execute('read_file_lines', READ, 3)
        self.assertEqual(self.grade().reward, 0.0)

    def test_deterministic_receipts_and_separate_session_progress(self):
        self.recover()
        receipt = self.session.receipt()
        other = Path(self.temp.name) / 'independent'
        initialize(other, self.spec)
        independent = RECOVERY.RecoverySession(self.spec, other)
        self.assertEqual(independent.verify(self.spec, other).status, 'fail')
        for number, name, args in ((0, 'read_file_lines', READ), (1, 'replace_file', EDIT),
                                   (2, 'read_file_lines', READ), (3, 'replace_file', EDIT),
                                   (4, 'read_file_lines', READ)):
            independent.execute(name, args, lambda name=name: native_read(other) if name == 'read_file_lines' else native_edit(other),
                                round_index=number)
        self.assertEqual(independent.receipt(), receipt)
        json.dumps(receipt, allow_nan=False)
        receipt['events'].clear()
        self.assertEqual(self.session.verify(self.spec, self.root).status, 'pass')


class RecoveryEpisodeIntegrationTests(unittest.TestCase):
    def test_actual_episode_wraps_tools_and_grades_complete_mocked_recovery(self):
        spec = recovery_spec()
        schemas = [
            {'type': 'function', 'function': {'name': 'replace_file', 'parameters': {'type': 'object',
                'properties': {'path': {'type': 'string'}, 'edits': {'type': 'array'}}}}},
            {'type': 'function', 'function': {'name': 'read_file_lines', 'parameters': {'type': 'object',
                'properties': {'path': {'type': 'string'}}}}},
            {'type': 'function', 'function': {'name': 'read_file_chars', 'parameters': {'type': 'object',
                'properties': {'path': {'type': 'string'}, 'offset': {'type': 'integer'}, 'length': {'type': 'integer'}}}}},
        ]
        def call(name, args):
            params = ''.join('<parameter=' + key + '>' + (value if isinstance(value, str) else json.dumps(value))
                             + '</parameter>' for key, value in args.items())
            return '</think><tool_call><function=' + name + '>' + params + '</function></tool_call>'
        replies = [call('read_file_lines', READ), call('replace_file', EDIT), call('read_file_lines', READ),
                   call('replace_file', EDIT), call('read_file_lines', READ), '</think>ANSWER: ' + TARGET]
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            class Workspace:
                def __init__(self, root, runner):
                    self.root = Path(root)
                def reset(self):
                    self.root.mkdir(parents=True, exist_ok=False)
            class Turn(SimpleNamespace):
                def __init__(self, **kwargs):
                    defaults = dict(action=None, extra=[], truncated=False, content='', resp_text='')
                    defaults.update(kwargs)
                    super().__init__(**defaults)
            def execute(ws, name, args, counters):
                parsed = json.loads(args) if isinstance(args, str) else args
                return native_read(ws.root) if name == 'read_file_lines' else native_edit(ws.root, parsed)
            def messages(system, user, rounds):
                result = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
                for step in rounds:
                    for executed in step['calls']:
                        result.append({'role': 'tool', 'content': executed['result']})
                return result
            native = SimpleNamespace(native_tools=lambda: schemas, to_api_messages=messages,
                system_prompt=Mock(side_effect=AssertionError('legacy prompt called')),
                _BLOCK_RE=re.compile(r'<tool_call>(.*?)</tool_call>', re.S),
                _FN_RE=re.compile(r'<function=([^>]+)>'),
                _PARAM_RE=re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.S),
                _js_parse=json.loads, _js_stringify=json.dumps,
                recover_from_reasoning=lambda *args: None, execute=Mock(side_effect=execute))
            M = dict(native=native, Workspace=Workspace, Runner=lambda _: object(),
                     Config=lambda: SimpleNamespace(episode=SimpleNamespace(data_root=data_root)),
                     Turn=Turn, Action=lambda name, args: SimpleNamespace(name=name, arguments=args),
                     Task=SimpleNamespace, dialect=SimpleNamespace(prepare=lambda cfg, task, ws: task))
            tok = SimpleNamespace(decode=lambda ids, **kwargs: replies[ids[0]])
            def render(tokenizer, msgs, tools):
                captured.append(copy.deepcopy(msgs))
                return [100 + len(captured)]
            completions = [dict(gen_ids=[i], lp_q4=[0.0], tps=0.0, stop_type='eos') for i in range(len(replies))]
            with ExitStack() as stack:
                stack.enter_context(patch.object(Q, 'MAX_ROUNDS', len(replies)))
                stack.enter_context(patch.object(Q, 'PROMPT_PROFILE', 'work-contract-v1'))
                stack.enter_context(patch.object(Q, 'render', side_effect=render))
                stack.enter_context(patch.object(Q, 'complete', side_effect=completions))
                stack.enter_context(patch.dict(Q.os.environ, {'GRPO_RUNNER': 'offline-fake'}))
                stack.enter_context(patch.object(Q.subprocess, 'Popen', side_effect=AssertionError('process launch')))
                stack.enter_context(patch.object(Q.requests, 'post', side_effect=AssertionError('network request')))
                result = Q.episode(M, tok, Q.contract_tasks.FAMILY, spec, 'recovery-fixture')
            self.assertEqual(result['reward'], 1.0)
            self.assertEqual(result['stop'], 'answer')
            self.assertEqual(sum(call.args[1] == 'replace_file' for call in native.execute.call_args_list), 1)
            self.assertTrue(any('conflict' in message['content'].lower() for message in captured[2]
                                if message['role'] == 'tool'))
            final_file = data_root / 'workspaces/grpo-recovery-fixture' / TARGET
            self.assertEqual(final_file.read_bytes(), spec['files'][TARGET].encode().replace(b'"retries": 3', b'"retries": 5', 1))


if __name__ == '__main__':
    unittest.main()
