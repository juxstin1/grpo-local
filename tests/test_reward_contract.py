import copy
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
import contract_tasks as tasks
import reward_contract as reward
import real_tasks


def turn(text='', **kwargs):
    return SimpleNamespace(content=text, resp_text=text, action=None, extra=[], truncated=False, **kwargs)


def solve(spec, variant=False):
    """Independent transformations of inputs; never copy the contract's expected answer."""
    files = dict(spec['files'])
    name = spec['name']
    if name.endswith('rename-callers'):
        for path in ('src/pricing.py', 'src/receipt.py'):
            files[path] = files[path].replace('apply_discount', 'apply_reduction')
            if variant:
                files[path] = '# Renamed public API; behavior retained.\n' + files[path].replace("'", '"')
    elif name.endswith('reconcile-decisions'):
        source = json.loads(files['records/decisions.json'])
        chosen = {}
        for row in sorted(source['decisions'], key=lambda r: r['revision']):
            if row['status'] == 'approved':
                chosen[row['field']] = row['value']
        files['reports/decision.json'] = json.dumps({'project': source['project'], **chosen},
                                                    indent=2 if variant else None, sort_keys=variant)
    elif name.endswith('open-orders'):
        latest = {}
        for row in csv.DictReader(io.StringIO(files['data/orders.csv'])):
            key = row['order_id']
            if key not in latest or int(row['revision']) > int(latest[key]['revision']):
                latest[key] = row
        orders, total = [], Decimal('0')
        for key, row in sorted(latest.items()):
            if row['status'] == 'cancelled':
                continue
            value = int(row['quantity']) * Decimal(row['unit_usd'])
            total += value
            orders.append(dict(order_id=key, total_usd=f'{value:.2f}'))
        files['reports/open-orders.json'] = json.dumps(dict(orders=orders, total_usd=f'{total:.2f}'),
                                                       indent=2 if variant else None, sort_keys=variant)
    else:
        before, after = files['docs/agenda.md'].split('## Beta\n', 1)
        files['docs/agenda.md'] = before + '## Beta\n' + after.replace('Draft status: pending\n', '', 1)
    return files


class OutcomeContracts(unittest.TestCase):
    def setUp(self):
        # Stateful recovery has separate dispatch/observation controls in
        # test_recovery_contract; these controls exercise artifact-only roots.
        self.tasks = [s for s in tasks.specs() if 'recovery' not in s['contract']]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def check(self, spec, files, turns=None):
        self.serial += 1
        folder = self.root / str(self.serial)
        folder.mkdir()
        real_tasks.write_files(files, folder)
        if turns is None:
            turns = [turn('ANSWER: ' + spec['contract']['final_answer'])]
        return reward.grade(spec, turns, folder)

    def test_one_correct_refactor_control(self):
        spec = self.tasks[0]
        self.assertEqual(self.check(spec, solve(spec)).reward, 1)

    def test_independent_reference_transformations(self):
        for spec in self.tasks:
            for variant in (False, True):
                with self.subTest(task=spec['name'], variant=variant):
                    self.assertEqual(self.check(spec, solve(spec, variant)).reward, 1)

    def test_original_noop_fails_static_roots(self):
        for spec in self.tasks:
            with self.subTest(task=spec['name']):
                self.assertEqual(self.check(spec, spec['files']).reward, 0)

    def test_comment_only_rename_fails(self):
        spec = self.tasks[0]
        files = {k: v + '\n# apply_reduction\n' for k, v in spec['files'].items()}
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_partial_cross_file_rename_fails(self):
        spec = self.tasks[0]
        files = solve(spec)
        files['src/receipt.py'] = spec['files']['src/receipt.py']
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_changed_algorithm_fails(self):
        spec = self.tasks[0]
        files = solve(spec)
        files['src/pricing.py'] = 'def apply_reduction(price, percent):\n    return 0\n'
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_invalid_python_fails(self):
        spec = self.tasks[0]
        files = solve(spec)
        files['src/pricing.py'] = '# apply_reduction\nthis is not valid python !\n'
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_encoding_cookie_cannot_hide_executable_statement(self):
        spec = self.tasks[0]
        files = solve(spec)
        files['src/pricing.py'] = "# coding: utf-7\n# +AAo-raise RuntimeError('injected')\n" + files['src/pricing.py']
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_deleted_or_corrupted_sources_fail(self):
        for spec in self.tasks:
            source = next(k for k in spec['files'] if k not in spec['contract']['outputs'])
            for delete in (False, True):
                files = solve(spec)
                if delete:
                    del files[source]
                else:
                    files[source] = 'destroyed\n'
                self.assertEqual(self.check(spec, files).reward, 0)

    def test_new_forged_success_file_fails(self):
        spec = self.tasks[0]
        files = solve(spec)
        files['test-results.txt'] = 'All tests passed\n'
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_draft_proposal_is_not_approved_fact(self):
        spec = self.tasks[1]
        files = solve(spec)
        files['reports/decision.json'] = files['reports/decision.json'].replace('21500', '30000')
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_negated_fact_prose_cannot_replace_structured_artifact(self):
        spec = self.tasks[1]
        files = solve(spec)
        files['reports/decision.json'] = 'None of these is correct: Quay Renewal; 21500; Ivo; 2026-11-18.'
        self.assertEqual(self.check(spec, files).reward, 0)

    def test_duplicate_keys_nan_and_extra_fields_fail(self):
        spec = self.tasks[1]
        for tail in (',"budget_usd":21500', ',"other":NaN', ',"claim":"sent"'):
            files = solve(spec)
            files['reports/decision.json'] = files['reports/decision.json'][:-1] + tail + '}'
            self.assertEqual(self.check(spec, files).reward, 0)

    def test_cancelled_order_and_stale_revision_fail(self):
        spec = self.tasks[2]
        for old, new in [('30.75', '20.50'), ('31.45', '45.45')]:
            files = solve(spec)
            files['reports/open-orders.json'] = files['reports/open-orders.json'].replace(old, new)
            self.assertEqual(self.check(spec, files).reward, 0)

    def test_wrong_section_or_all_occurrences_fail(self):
        spec = self.tasks[3]
        for count in (1, 2):
            files = dict(spec['files'])
            files['docs/agenda.md'] = files['docs/agenda.md'].replace('Draft status: pending\n', '', count)
            self.assertEqual(self.check(spec, files).reward, 0)

    def test_stale_answer_blank_retraction_and_irrelevant_question_fail(self):
        spec = self.tasks[1]
        earlier = turn('ANSWER: ' + spec['contract']['final_answer'])
        earlier.action = object()
        for text in ('', 'I failed; disregard my previous reply.', 'Would you like a sandwich?', 'ANSWER: wrong.txt'):
            self.assertEqual(self.check(spec, solve(spec), [earlier, turn(text)]).reward, 0)

    def test_false_sent_claim_fails_even_with_delivery_line(self):
        spec = self.tasks[1]
        response = 'This email has been sent.\nANSWER: ' + spec['contract']['final_answer']
        self.assertEqual(self.check(spec, solve(spec), [turn(response)]).reward, 0)

    def test_truncation_and_unexecuted_action_fail(self):
        spec = self.tasks[0]
        for field in ('truncated', 'action'):
            last = turn('ANSWER: ' + spec['contract']['final_answer'])
            setattr(last, field, True)
            self.assertEqual(self.check(spec, solve(spec), [last]).reward, 0)

    def test_workspace_unavailable_is_invalid(self):
        spec = self.tasks[0]
        with self.assertRaises(reward.InvalidReward):
            reward.grade(spec, [turn('ANSWER: ' + spec['contract']['final_answer'])], self.root / 'missing')

    def test_legacy_prose_tasks_are_unqualified_not_zero(self):
        with self.assertRaises(reward.UnqualifiedTask):
            reward.validate_spec(dict(name='office-legacy', family='office_draft'))
        with self.assertRaises(reward.UnqualifiedTask):
            real_tasks.make_task({})

    def test_bad_contract_configuration_is_invalid(self):
        for rule in (None, [], {'kind': 'python_ast', 'expected': []},
                     {'kind': 'python_ast', 'expected': 'broken !'},
                     {'kind': 'json_exact', 'expected': float('nan')}):
            spec = copy.deepcopy(self.tasks[0])
            spec['contract']['outputs']['src/pricing.py'] = rule
            with self.assertRaises(reward.InvalidReward):
                reward.validate_spec(spec)

    def test_same_candidate_repeats_identically(self):
        for spec in self.tasks:
            files = solve(spec)
            results = [self.check(spec, files) for _ in range(3)]
            self.assertEqual(results, [reward.Verdict('pass', 'verified-outcome')] * 3)

    def test_boolean_is_not_integer_json_fact(self):
        self.assertFalse(reward._json_equal({'n': True}, {'n': 1}))

    def test_all_sampled_families_reported(self):
        result = reward.summarize([dict(family='synthetic', reward=1), dict(family='bionic_real', reward=0)])
        self.assertEqual(result['by_family'], {'bionic_real': [0], 'synthetic': [1]})
        self.assertEqual(result['reward_mean'], 0.5)

    def test_no_valid_samples_or_invalid_reward_cannot_score(self):
        for episodes in ([], [dict(family='x', reward=None)], [dict(family='x', reward=float('nan'))]):
            with self.assertRaises(reward.InvalidReward):
                reward.summarize(episodes)

    def test_missing_legacy_verifier_raises_invalid(self):
        with patch('subprocess.run', side_effect=FileNotFoundError('injected')):
            with self.assertRaises(reward.InvalidReward):
                real_tasks.run_check(self.root, {'cmd': ['missing-verifier']})

    def test_contract_paths_cannot_escape_workspace(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'file:stream', 'a\\b'):
            with self.assertRaises(reward.InvalidReward):
                real_tasks.write_files({name: 'x'}, self.root)


if __name__ == '__main__':
    unittest.main()
