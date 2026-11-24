"""Synthetic result accounting and mocked runner checks; no candidate/model runs."""
import ast
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from evals import eval_runner as runner
from evals import result_contract as R


def receipt():
    return dict(verified=True, served_model='fixture-model', base_url='http://127.0.0.1:8000',
                artifacts_sha256={'fixture.gguf': 'a' * 64}, verification_method='synthetic offline fixture',
                verified_at='2026-10-26T12:00:00+00:00')


def row(item='a', attempt=1, passed=True, **extra):
    return dict(item_id=item, attempt=attempt, passed=passed, bucket='fixture', run_id='run-fixture', **extra)


class ResultAccountingTests(unittest.TestCase):
    def test_complete_two_attempt_score_and_no_hardcoded_pass4(self):
        rows = [row('a', 1, False), row('a', 2, True), row('b', 1, False), row('b', 2, False)]
        result = R.summarize(rows, ['a', 'b'], 2, 'fixture', 'run-fixture')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['k'], 2)
        self.assertEqual(result['pass_at_k'], 0.5)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['per_attempt_acc'], 0.25)
        self.assertNotIn('pass4_items', result)

    def test_missing_attempt_invalidates_headline_even_if_item_already_solved(self):
        result = R.summarize([row()], ['a'], 2, 'fixture', 'run-fixture')
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['pass_at_k'])
        self.assertIsNone(result['per_item']['a'])
        self.assertEqual(result['missing_attempts'], 1)
        self.assertEqual(result['observed_solved_items'], 1)

    def test_request_errors_are_incomplete_not_zero(self):
        result = R.summarize([row(passed=None, note='request-error')], ['a'], 1, 'fixture', 'run-fixture')
        self.assertEqual(result['invalid_attempts'], 1)
        self.assertEqual(result['valid_attempts'], 0)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['accuracy'])

    def test_empty_log_is_incomplete_not_zero(self):
        result = R.summarize([], ['a'], 4, 'fixture', 'run-fixture')
        self.assertEqual(result['missing_attempts'], 4)
        self.assertIsNone(result['accuracy'])

    def test_foreign_duplicate_conflicting_and_malformed_rows_fail_closed(self):
        bad_sets = [[row('foreign')], [row(), row()], [row(), row(passed=False)],
                    [row(attempt=True)], [row(attempt=0)], [row(attempt=2)],
                    [row(passed=1)], [row(passed='true')], [row(passed=None)],
                    [{**row(), 'run_id': 'other'}], [{**row(), 'bucket': 'other'}],
                    [{**row(), 'error': 'unavailable'}], ['not-a-row']]
        for rows in bad_sets:
            with self.subTest(rows=rows), self.assertRaises(R.ResultIntegrityError):
                R.summarize(rows, ['a'], 1, 'fixture', 'run-fixture')

    def test_selected_ids_are_exact_unique_and_public(self):
        for items in ([], [{'id': 'a'}, {'id': 'a'}], [{'uid': 'a'}], [{'id': ' a'}],
                      [{'id': 'a', 'split': 'holdout'}]):
            with self.subTest(items=items), self.assertRaises(R.ResultIntegrityError):
                R.selected_ids(items)

    def test_raw_logs_do_not_skip_malformed_or_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            for text in ('{broken', json.dumps(row()) + '\n\n', '{"passed":true,"passed":false}'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(R.ResultIntegrityError):
                    R.read_rows(path, ['a'], 1, 'fixture', 'run-fixture')

    def test_unicode_line_separators_inside_json_strings_remain_one_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            value = row(note='model text\u2028continues\u2029here')
            path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
            records = R.read_rows(path, ['a'], 1, 'fixture', 'run-fixture')
            self.assertEqual(records[('a', 1)]['note'], value['note'])

    def test_forged_duplicate_json_verdict_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            text = json.dumps(row())[:-1] + ', "passed": false}\n'
            path.write_text(text, encoding='utf-8')
            with self.assertRaises(R.ResultIntegrityError):
                R.read_rows(path, ['a'], 1, 'fixture', 'run-fixture')


class IdentityTests(unittest.TestCase):
    def identity(self, **changes):
        kwargs = dict(dataset_bytes=b'{"fixture":1}', selected={'fixture': ['a']},
                      rendered={'a': {'messages': [{'content': 'question'}], 'tools': []}},
                      grader_sources={'grader.py': b'actual grader source'},
                      settings={'attempts': 1, 'sampling': {'temperature': 0.7}}, served_identity=receipt())
        kwargs.update(changes)
        return R.make_identity(**kwargs)

    def test_every_identity_input_changes_run_digest(self):
        baseline = self.identity()
        changes = [dict(dataset_bytes=b'{"fixture":2}'), dict(selected={'fixture': ['b']}),
                   dict(rendered={'a': {'messages': [], 'tools': [{'name': 'changed'}]}}),
                   dict(grader_sources={'grader.py': b'new grader'}),
                   dict(settings={'attempts': 2, 'sampling': {'temperature': 0.7}}),
                   dict(served_identity={**receipt(), 'artifacts_sha256': {'fixture.gguf': 'b' * 64}})]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(self.identity(**change)['run_id'], baseline['run_id'])

    def test_serving_identity_requires_verified_alias_endpoint_and_weight_hashes(self):
        for change in ({'verified': False}, {'served_model': 'nearby-base'}, {'base_url': 'http://127.0.0.1:9'},
                       {'artifacts_sha256': {}}, {'artifacts_sha256': {'x': 'not-sha'}},
                       {'verification_method': ''}, {'verified_at': '2026-10-26'}):
            with self.subTest(change=change), self.assertRaises(R.ResultIntegrityError):
                R.validate_served_identity({**receipt(), **change}, 'fixture-model', 'http://127.0.0.1:8000')

    def test_resume_refuses_unbound_folder_identity_change_and_foreign_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            identity = self.identity()
            with self.assertRaises(R.ResultIntegrityError):
                R.validate_resume(path, identity, {'fixture': ['a']}, 1)
            (path / 'run.identity.json').write_text(json.dumps(identity), encoding='utf-8')
            with self.assertRaises(R.ResultIntegrityError):
                R.validate_resume(path, self.identity(dataset_bytes=b'changed'), {'fixture': ['a']}, 1)
            (path / 'raw').mkdir()
            (path / 'raw/foreign.jsonl').write_text('', encoding='utf-8')
            with self.assertRaises(R.ResultIntegrityError):
                R.validate_resume(path, identity, {'fixture': ['a']}, 1)

    def test_second_writer_is_refused_and_owner_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            identity = self.identity()
            with R.locked_run(path, identity, {'fixture': ['a']}, 1):
                with self.assertRaises(R.ResultIntegrityError):
                    with R.locked_run(path, identity, {'fixture': ['a']}, 1):
                        self.fail('second writer acquired lock')
                self.assertTrue((path / '.run.lock').exists())
            self.assertFalse((path / '.run.lock').exists())


class RunnerOfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('eval_runner.py', 'graders.py', 'result_contract.py', 'code_executor.py'):
            (self.root / name).write_text('synthetic identity source ' + name, encoding='utf-8')
        self.data = self.root / 'public.json'
        self.data.write_text(json.dumps({'items': {'mhqa': [dict(id='fixture-a', dataset='mhqa',
            prompt='What is the capital of France?', answer={'answer': 'Paris', 'aliases': []})]},
            'buckets': {'mhqa': {'max_tokens': 50}}}), encoding='utf-8')
        self.served = self.root / 'served.json'
        self.served.write_text(json.dumps(receipt()), encoding='utf-8')
        self.args = ['eval_runner', '--model', 'fixture-model', '--slug', 'fixture',
                     '--set', str(self.data), '--served-identity', str(self.served),
                     '--attempts', '2', '--budget', '8192', '--allow-model-run']

    def invoke(self, extra=(), output=('ANSWER: Paris', 0.1, 4), health=True):
        with patch.object(runner, 'ROOT', self.root), patch.object(sys, 'argv', self.args + list(extra)), \
             patch.object(runner, 'chat', return_value=output) as chat, \
             patch.object(runner, 'wait_healthy', return_value=health) as wait:
            runner.main()
        return chat, wait

    def summary(self):
        return json.loads((self.root / 'results/fixture/mhqa.summary.json').read_text(encoding='utf-8'))

    def test_mocked_end_to_end_and_complete_resume_make_no_new_requests(self):
        chat, _ = self.invoke()
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(self.summary()['accuracy'], 1.0)
        chat, health = self.invoke()
        chat.assert_not_called()
        health.assert_not_called()

    def test_resume_fills_only_missing_slot(self):
        self.invoke()
        raw = self.root / 'results/fixture/raw/mhqa.jsonl'
        rows = raw.read_text(encoding='utf-8').split('\n')
        raw.write_text(rows[0] + '\n', encoding='utf-8')
        chat, _ = self.invoke()
        self.assertEqual(chat.call_count, 1)
        parsed = [json.loads(line) for line in raw.read_text(encoding='utf-8').split('\n') if line]
        self.assertEqual({r['attempt'] for r in parsed}, {1, 2})
        self.assertEqual(len(parsed), 2)
        self.assertEqual(self.summary()['accuracy'], 1.0)

    def test_resume_identity_mismatch_fails_before_request_or_write(self):
        self.invoke()
        raw = self.root / 'results/fixture/raw/mhqa.jsonl'
        before = raw.read_bytes()
        with patch.object(runner, 'ROOT', self.root), patch.object(sys, 'argv', self.args + ['--sampling', '{"temperature":0.2}']), \
             patch.object(runner, 'chat', side_effect=AssertionError('request')) as chat, \
             patch.object(runner, 'wait_healthy', side_effect=AssertionError('health')) as health:
            with self.assertRaises(R.ResultIntegrityError):
                runner.main()
        self.assertEqual(raw.read_bytes(), before)
        chat.assert_not_called()
        health.assert_not_called()

    def test_changed_executor_image_refuses_resume_and_is_recorded(self):
        with patch.dict(runner.os.environ, {'GRPO_CODE_SANDBOX_IMAGE': 'python@sha256:' + 'a' * 64}):
            self.invoke()
        settings = self.summary()['identity']['settings']
        self.assertEqual(settings['code_sandbox_image'], 'python@sha256:' + 'a' * 64)
        self.assertEqual(settings['python_version'], sys.version)
        self.assertEqual(settings['executor_profile'], runner.code_executor.EXECUTOR_PROFILE)
        with patch.dict(runner.os.environ, {'GRPO_CODE_SANDBOX_IMAGE': 'python@sha256:' + 'b' * 64}):
            with self.assertRaises(R.ResultIntegrityError):
                self.invoke()

    def test_resume_foreign_rows_fail_before_any_request(self):
        self.invoke()
        raw = self.root / 'results/fixture/raw/mhqa.jsonl'
        rows = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
        rows[0]['item_id'] = 'foreign'
        raw.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
        with patch.object(runner, 'ROOT', self.root), patch.object(sys, 'argv', self.args), \
             patch.object(runner, 'wait_healthy', side_effect=AssertionError('health')):
            with self.assertRaises(R.ResultIntegrityError):
                runner.main()

    def test_request_or_grader_errors_produce_incomplete_summary(self):
        self.invoke(output=(None, 0.1, 0))
        self.assertIsNone(self.summary()['accuracy'])
        self.assertEqual(self.summary()['invalid_attempts'], 2)
        chat, _ = self.invoke()
        chat.assert_not_called()  # immutable failed observations are not silently replaced

    def test_grader_infrastructure_error_is_not_a_wrong_answer(self):
        with patch.object(runner.graders, 'grade', side_effect=runner.graders.InvalidGrading('sandbox unavailable')):
            self.invoke()
        self.assertEqual(self.summary()['status'], 'incomplete')
        self.assertIsNone(self.summary()['accuracy'])

    def test_code_preflight_failure_precedes_health_and_generation(self):
        self.data.write_text(json.dumps({'items': {'code': [dict(id='code-fixture', dataset='code',
            prompt='Echo stdin.', tests={'kind': 'stdin', 'cases': [{'input': 'a', 'output': 'a'}]})]},
            'buckets': {'code': {'max_tokens': 50}}}), encoding='utf-8')
        with patch.object(runner, 'ROOT', self.root), patch.object(sys, 'argv', self.args), \
             patch.object(runner.graders, 'preflight', side_effect=runner.graders.InvalidGrading('image not configured')) as preflight, \
             patch.object(runner, 'chat', side_effect=AssertionError('generation')) as chat, \
             patch.object(runner, 'wait_healthy', side_effect=AssertionError('health')) as health:
            with self.assertRaises(runner.graders.InvalidGrading):
                runner.main()
        preflight.assert_called_once()
        chat.assert_not_called()
        health.assert_not_called()

    def test_no_model_opt_in_refuses_before_dataset_or_identity_read(self):
        args = [a for a in self.args if a != '--allow-model-run']
        with patch.object(sys, 'argv', args), patch.object(Path, 'read_bytes', side_effect=AssertionError('input read')):
            with self.assertRaisesRegex(SystemExit, 'Model execution is disabled'):
                runner.main()

    def test_private_options_and_paths_are_refused_without_opening(self):
        for extra in (['--sealed'], ['--holdout-seal', 'invented'], ['--set', 'private/tasks.json'],
                      ['--set', 'holdout.json']):
            with self.subTest(extra=extra), patch.object(sys, 'argv', self.args + extra), \
                 patch.object(Path, 'read_bytes', side_effect=AssertionError('private input read')):
                with self.assertRaises(R.ResultIntegrityError):
                    runner.main()

    def test_unknown_bucket_is_not_silently_filtered(self):
        with self.assertRaises(R.ResultIntegrityError):
            self.invoke(extra=['--buckets', 'mhqa,missing'])
        self.assertFalse((self.root / 'results').exists())

    def test_lcb_stdin_delegates_to_trusted_executor(self):
        cases = [{'input': '1\n', 'output': '2\n'}]
        with patch.object(runner.graders, 'run_stdin', return_value=(True, 'pass:1/1')) as execute:
            self.assertEqual(runner.grade_lcb({'tests': cases}, '```python\nprint(2)\n```'), (True, 'pass:1/1'))
        execute.assert_called_once_with('print(2)\n', cases)


class TauComparisonTests(unittest.TestCase):
    def test_temperature_thinking_and_step_cap_changes_refuse_comparison(self):
        # Importing Tau2 would import model integrations; execute only its pure
        # comparison definition and key tuple from the actual source instead.
        source = Path(__file__).resolve().parents[1] / 'tau2-bridge/run_tau2.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compare'
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMPARE_KEYS' for t in n.targets)]
        namespace = dict(json=json, Path=Path, sys=sys)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.json', Path(directory) / 'b.json'
            settings = dict(sim_model='fixture', sim_temperature=0, judge_llm=None, tau2='fixture',
                            tasks=['a'], budget=8192, trials=2, agent_temperature=0.7, thinking=True, max_steps=10)
            a.write_text(json.dumps({'settings': settings}), encoding='utf-8')
            for key, value in (('agent_temperature', 0.2), ('thinking', False), ('max_steps', 20)):
                b.write_text(json.dumps({'settings': {**settings, key: value}}), encoding='utf-8')
                with self.subTest(key=key), self.assertRaisesRegex(SystemExit, key):
                    namespace['compare']([str(a), str(b)])


if __name__ == '__main__':
    unittest.main()
