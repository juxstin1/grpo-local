import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evals import code_executor as engine
from evals import graders


class EndpointContracts(unittest.TestCase):
    def setUp(self):
        self.cases = [{'input': '3', 'output': '6'}, {'input': '4', 'output': '8'}]
        self.record = {'tests': {'kind': 'functional', 'entry': 'Solution.twice', 'cases': self.cases}}

    def run_fixture(self, job, timeout):
        """Only tests' own harmless fixture code; never data/model candidate code."""
        out = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(job))), contextlib.redirect_stdout(out):
            exec(engine.WORKER, {})
        return {'status': 'ok', 'stdout': out.getvalue()}

    def test_correct_function_control(self):
        with patch.object(engine, '_run_container', side_effect=self.run_fixture):
            for body in ('x*2', 'x+x'):
                result = graders.grade_code(self.record, f'```python\nclass Solution:\n def twice(self,x): return {body}\n```')
                self.assertTrue(result[0])

    def test_wrong_function_control(self):
        with patch.object(engine, '_run_container', side_effect=self.run_fixture):
            self.assertFalse(graders.grade_code(self.record, '```python\nclass Solution:\n def twice(self,x): return 6\n```')[0])

    def test_no_answers_or_verdict_in_candidate_job(self):
        with patch.object(engine, '_run_container', return_value={'status': 'ok', 'stdout': '{"outputs":[6,8]}'}) as run:
            self.assertEqual(graders.run_functional('pass', 'Solution.twice', self.cases)['passed'], 2)
        job = run.call_args.args[0]
        self.assertEqual(set(job), {'kind', 'code', 'entry', 'inputs'})
        self.assertEqual(job['inputs'], [[3], [4]])
        self.assertNotIn('output', job)

    def test_counterfeit_verdict_and_malformed_outputs_rejected(self):
        for text in ('{"passed":2,"total":2}', '{"outputs":[6,8],"passed":2}',
                     '{"outputs":[6]}', '{"outputs":[6,8]}\n{"outputs":[6,8]}',
                     '{"outputs":[6,8],"outputs":[6,8]}', '{"outputs":[NaN,8]}'):
            with self.subTest(text=text), patch.object(engine, '_run_container', return_value={'status':'ok','stdout':text}):
                self.assertIn('error', graders.run_functional('pass', 'Solution.twice', self.cases))

    def test_parent_comparison_rejects_boolean_numeric_confusion(self):
        with patch.object(engine, '_run_container', return_value={'status':'ok','stdout':'{"outputs":[true]}' }):
            res = graders.run_functional('pass', 'Solution.test', [{'input':'1','output':'1'}])
            self.assertEqual(res['passed'], 0)

    def test_correct_stdin_control(self):
        with patch.object(engine, '_run_container', side_effect=self.run_fixture):
            self.assertTrue(graders.run_stdin('print(int(input())*2)', self.cases)[0])

    def test_crash_and_unavailable_executor_are_invalid(self):
        with patch.object(engine, '_run_container', side_effect=engine.InvalidGrading('nonzero sandbox exit')):
            with self.assertRaises(engine.InvalidGrading):
                graders.run_stdin("print(6)\nraise RuntimeError()", self.cases)

    def test_empty_suites_invalid_before_execution(self):
        with patch.object(engine, '_run_container') as run:
            with self.assertRaises(engine.InvalidGrading):
                graders.run_stdin('pass', [])
            with self.assertRaises(engine.InvalidGrading):
                graders.run_functional('pass', 'Solution.test', [])
            run.assert_not_called()

    def test_assert_suite_refused_without_running_candidate(self):
        with patch.object(engine, '_run_container') as run:
            with self.assertRaises(engine.InvalidGrading):
                graders.run_functional('pass', 'Solution.test', [], kind='assert', source='expected_secret')
            run.assert_not_called()

    def test_bad_test_data_invalid(self):
        with self.assertRaises(engine.InvalidGrading):
            graders.run_functional('pass', 'Solution.test', [{'input':'?','output':'6'}])
        with self.assertRaises(engine.InvalidGrading):
            graders.run_stdin('pass', [{'input':''}])

    def test_missing_or_mutable_image_refused_before_process(self):
        for image in ('', 'python:latest', 'python@sha256:bad'):
            with patch.dict(os.environ, {'GRPO_CODE_SANDBOX_IMAGE': image}), patch.object(engine.subprocess, 'Popen') as popen:
                with self.assertRaises(engine.InvalidGrading):
                    engine.execute('pass', kind='stdin', stdin='')
                popen.assert_not_called()

    def test_container_command_has_isolation_and_no_host_or_gpu_access(self):
        args = engine.command('python@sha256:'+'a'*64, 'grpo-public-grade-fixture')
        for required in ('--pull=never','--network=none','--read-only','--cap-drop=ALL',
                         '--security-opt=no-new-privileges','--user=65534:65534','--pids-limit=32','--memory=256m'):
            self.assertIn(required,args)
        for prohibited in ('--gpus','--privileged','--volume','-v','--mount'):
            self.assertNotIn(prohibited,args)
        self.assertNotIn('expected', engine.WORKER)

    def test_missing_executable_invalid(self):
        with patch.dict(os.environ, {'GRPO_CODE_SANDBOX_IMAGE':'python@sha256:'+'a'*64}), patch.object(engine.subprocess,'Popen',side_effect=FileNotFoundError):
            with self.assertRaises(engine.InvalidGrading):
                engine.execute('pass',kind='stdin',stdin='')

    def test_transport_exit_and_output_limits(self):
        for exit_code, output, expectation in ((0, b'6\n', 'ok'), (1, b'6\n', 'invalid'),
                                                (125, b'', 'invalid'), (0, b'x'*100, 'candidate-error')):
            proc = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(output), stderr=io.BytesIO(),
                                   returncode=exit_code, poll=lambda: exit_code)
            with self.subTest(exit_code=exit_code, expectation=expectation), \
                    patch.dict(os.environ, {'GRPO_CODE_SANDBOX_IMAGE':'python@sha256:'+'a'*64}), \
                    patch.object(engine.subprocess, 'Popen', return_value=proc), patch.object(engine, 'MAX_OUTPUT', 20):
                if expectation == 'invalid':
                    with self.assertRaises(engine.InvalidGrading):
                        engine.execute('pass',kind='stdin',stdin='')
                else:
                    self.assertEqual(engine.execute('pass',kind='stdin',stdin='')['status'], expectation)

    def test_large_integer_comparison_preserves_exactness(self):
        self.assertTrue(graders._same(10**500, 10**500))
        self.assertFalse(graders._same(10**500, 10**500+1))

    def test_mhqa_only_final_nonempty_answer(self):
        record = {'answer': {'answer': 'Oslo'}}
        self.assertTrue(graders.grade_mhqa(record,'ANSWER: Oslo')[0])
        for reply in ('ANSWER: Oslo\nI retract that answer.', 'ANSWER: Oslo\nANSWER:', 'ANSWER: Oslo\nANSWER: Bergen'):
            self.assertFalse(graders.grade_mhqa(record, reply)[0])

    def test_chain_control_and_incorrect_intermediate(self):
        record={'answer':{'chain':['Alice','Works','Oslo']}}
        self.assertTrue(graders.grade_mhqa(record,'ANSWER: Alice -> Works -> Oslo')[0])
        self.assertFalse(graders.grade_mhqa(record,'ANSWER: Alice -> Other -> Oslo')[0])

    def test_math_box_requires_final_delivery(self):
        self.assertEqual(graders.last_boxed(r'answer \boxed{42}'), '42')
        self.assertIsNone(graders.last_boxed(r'\boxed{42} I retract this.'))

    def test_duplicate_pool_ids_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pool.jsonl'
            p.write_text('{"uid":"same","prompt":"one"}\n{"uid":"same","prompt":"two"}\n')
            with self.assertRaises(engine.InvalidGrading):
                graders.load_pool(p)

    def test_preflight_refuses_before_execution(self):
        with patch.object(engine, '_run_container') as run, patch.dict(os.environ, {'GRPO_CODE_SANDBOX_IMAGE':''}):
            with self.assertRaises(engine.InvalidGrading):
                graders.preflight([{'dataset':'code', **self.record}])
            with self.assertRaises(engine.InvalidGrading):
                graders.preflight([{'dataset':'code', 'tests':{'kind':'assert','source':'pass'}}])
            graders.preflight([{'dataset':'math','answer':'42'}])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
