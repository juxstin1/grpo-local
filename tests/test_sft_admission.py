import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sft import admission as A
from sft.audit_teacher import audit, sha


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.task = {'uid': 'fixture', 'root_id': 'fixture-root', 'dataset': 'mhqa',
                     'prompt': 'Who owns the invented fixture?', 'answer': {'answer': 'Ivo'}}
        self.registry = SimpleNamespace(sha256='registry', validate_training_rows=lambda _: None)
        self.index = {'fixture': self.task}
        self.row = {'id': 'win', 'uid': 'fixture', 'root_id': 'fixture-root', 'split': 'train',
                    'grade': {'pass': True, 'status': 'valid'}, 'dataset_registry_sha256': 'registry',
                    'messages': [{**m, 'train': False} for m in A.graders.messages(self.task)],
                    'prompt_ids': [1, 2], 'completion_ids': [3, 4],
                    'completion_text': 'ANSWER: Ivo<eos>', 'verification_response': 'ANSWER: Ivo'}
        self.bind()
        self.tok = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2],
                                   decode=lambda ids, skip_special_tokens: 'ANSWER: Ivo' + ('' if skip_special_tokens else '<eos>'))

    def bind(self):
        self.row['admission_evidence'] = A.make_evidence(self.task, self.row, self.registry.sha256)

    def test_actual_registered_answer_replayed(self):
        self.assertEqual(A.verify_tokens_and_replay(self.row, self.tok, self.registry, self.index)['status'], 'replayed')

    def test_forged_win_even_rehashed_fails_replay(self):
        self.row.update(completion_text='ANSWER: Maya<eos>', verification_response='ANSWER: Maya')
        self.bind()
        self.tok.decode = lambda ids, skip_special_tokens: 'ANSWER: Maya' + ('' if skip_special_tokens else '<eos>')
        with self.assertRaisesRegex(A.AdmissionError, 'independent replay'):
            A.verify_tokens_and_replay(self.row, self.tok, self.registry, self.index)

    def test_changed_token_target_rejected(self):
        self.row['completion_text'] = 'another target'
        self.bind()
        with self.assertRaisesRegex(A.AdmissionError, 'completion IDs'):
            A.verify_tokens_and_replay(self.row, self.tok, self.registry, self.index)

    def test_changed_prompt_ids_rejected(self):
        self.row['prompt_ids'] = [9]
        self.bind()
        with self.assertRaisesRegex(A.AdmissionError, 'prompt token IDs'):
            A.verify_tokens_and_replay(self.row, self.tok, self.registry, self.index)

    def test_hint_rewrite_cannot_rehash_into_admission(self):
        self.row['messages'][1]['content'] += '\nHint: Ivo'
        self.bind()
        with self.assertRaisesRegex(A.AdmissionError, 'instructions'):
            A.check_record(self.row, self.registry, self.index)

    def test_transformation_flags_blocked(self):
        for key in ('hint_stripped', 'prompt_refreshed', 'context_transformations', 'hinted'):
            row = copy.deepcopy(self.row)
            row[key] = True
            with self.assertRaisesRegex(A.AdmissionError, 'separate replay'):
                A.check_record(row, self.registry, self.index)

    def test_provenance_replay_flag_cannot_be_hidden_by_pass(self):
        self.row['provenance'] = {'requires_replay': True}
        self.bind()
        with self.assertRaisesRegex(A.AdmissionError, 'separate replay'):
            A.check_record(self.row, self.registry, self.index)

    def test_missing_receipt_and_changed_verifier(self):
        self.row.pop('admission_evidence')
        with self.assertRaisesRegex(A.AdmissionError, 'missing reproducible'):
            A.check_record(self.row, self.registry, self.index)
        self.bind()
        with patch.object(A, 'verifier_identity', return_value={'changed': True}), self.assertRaisesRegex(A.AdmissionError, 'grader/runtime'):
            A.check_record(self.row, self.registry, self.index)

    def test_registry_task_change_and_unregistered_tasks(self):
        self.index['fixture'] = {**self.task, 'answer': 'Maya'}
        with self.assertRaises(A.AdmissionError):
            A.check_record(self.row, self.registry, self.index)
        with self.assertRaises(A.AdmissionError):
            A.require_registered_tasks([self.task], self.index)

    def test_replay_infrastructure_error_never_passes(self):
        with patch.object(A.graders, 'grade', side_effect=A.InvalidGrading('unavailable')), self.assertRaisesRegex(A.AdmissionError, 'unavailable'):
            A.verify_tokens_and_replay(self.row, self.tok, self.registry, self.index)

    def test_duplicate_rows_rejected(self):
        with self.assertRaises(A.AdmissionError):
            A.check_dataset([self.row, self.row], self.registry, self.index)


class TeacherAuditTests(unittest.TestCase):
    def test_provenance_and_root_quarantine_without_content_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ['{"invented":"first"}', '{"invented":"second"}']
            (root / 'source.jsonl').write_text('\n'.join(source), encoding='utf-8')
            rows = [{'id': str(i), 'root_id': 'same-root', 'teacher_model': 'fixture-teacher',
                     'teacher_trace_id': str(i), 'completion_ids': [i]} for i in range(2)]
            manifest = {'admitted': [{'model': 'fixture-teacher', 'id': str(i), 'file': 'source.jsonl',
                         'line': i+1, 'row_sha256': sha(source[i].encode()), 'hinted': i == 0} for i in range(2)]}
            train, metadata = root / 'train.jsonl', root / 'manifest.json'
            train.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            metadata.write_text(json.dumps(manifest), encoding='utf-8')
            result = audit(train, metadata, root)
            self.assertEqual((result['hinted_rows'], result['hinted_root_rows'], result['admitted_rows']), (1, 2, 0))
            self.assertNotIn('completion_ids', result['dispositions'][0])
            (root / 'source.jsonl').write_text('tampered\n' + source[1], encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                audit(train, metadata, root)


if __name__ == '__main__':
    unittest.main()
