"""Invented token records only: loss-mask boundaries and transformation lineage."""
import builtins
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from sft import sft_schema as schema
from sft import from_gym as converter


EOS = 2
SHA = 'a' * 64


def render_fixture(tokenizer, messages, thinking):
    assert thinking is True
    return [10, len(messages[0]['content']), 20]


class TokenizerFixture:
    def decode(self, tokens):
        return 'fixture tokens: ' + ','.join(map(str, tokens))


def record(system='Follow the task.\n'):
    messages = [{'role': 'system', 'content': system, 'train': False},
                {'role': 'user', 'content': 'Do the task.', 'train': False}]
    prompt = render_fixture(None, messages, True)
    segments = [{'kind': 'prompt', 'ids': prompt, 'train': False},
                {'kind': 'assistant', 'ids': [40, EOS], 'train': True},
                {'kind': 'obs', 'ids': [31], 'train': False},
                {'kind': 'assistant', 'ids': [41, EOS], 'train': True}]
    return dict(id='item-a1', root_id='domain/task', bucket='fixture', source='synthetic', split='train',
                messages=messages, prompt_ids=prompt, segments=segments, completion_ids=[40, EOS, 41, EOS],
                completion_text='fixture', n_tokens=8, grade={'pass': True, 'detail': 'fixture pass'},
                model='fixture-model', tokenizer_template_sha=SHA,
                sampling={'temperature': 0.7, 'top_p': 0.8, 'max_tokens': 100})


class SchemaBoundaryTests(unittest.TestCase):
    def test_valid_segmented_record_masks_context_and_observation(self):
        value = record()
        schema.validate(value, EOS)
        ids, mask = schema.training_tokens_and_mask(value, EOS)
        self.assertEqual(mask, [False, False, False, True, True, False, True, True])
        labels = [token if train else -100 for token, train in zip(ids, mask)]
        self.assertEqual(labels, [-100, -100, -100, 40, EOS, -100, 41, EOS])

    def test_valid_single_turn_requires_context_and_eos(self):
        value = record()
        del value['segments']
        value['n_tokens'] = len(value['prompt_ids']) + len(value['completion_ids'])
        schema.validate(value, EOS)
        ids, mask = schema.training_tokens_and_mask(value, EOS)
        self.assertEqual(ids, value['prompt_ids'] + value['completion_ids'])
        self.assertEqual(mask, [False, False, False, True, True, True, True])
        for field in ('prompt_ids', 'completion_ids', 'messages'):
            bad = copy.deepcopy(value)
            bad[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                schema.validate(bad, EOS)

    def test_grade_pass_requires_exact_boolean_true(self):
        for flag in (1, 0, 'true', 'false', None, False):
            value = record()
            value['grade']['pass'] = flag
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        value['grade']['pass'] = False
        schema.validate_structure(value, EOS)

    def test_message_flags_cannot_train_or_use_truthy_coercion(self):
        for flag in (True, 1, 0, 'false', None):
            value = record()
            value['messages'][0]['train'] = flag
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        value = record()
        del value['messages'][0]['train']
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)

    def test_every_context_kind_is_forbidden_from_training(self):
        for kind in ('obs', 'observation', 'tool', 'user', 'system', 'context'):
            value = record()
            value['segments'][2] = {'kind': kind, 'ids': [31, EOS], 'train': True}
            value['completion_ids'] = [40, EOS, 31, EOS, 41, EOS]
            value['n_tokens'] = 9
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                schema.validate(value, EOS)

    def test_unknown_kind_or_contradictory_role_is_not_assumed_trainable(self):
        for train in (True, False):
            value = record()
            value['segments'][1]['kind'] = 'unrecognized'
            value['segments'][1]['train'] = train
            with self.subTest(train=train), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        value = record()
        value['segments'][1]['role'] = 'user'
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)

    def test_segment_flags_are_exact_booleans(self):
        for flag in (1, 0, None, 'true', 'false'):
            value = record()
            value['segments'][1]['train'] = flag
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                schema.validate(value, EOS)

    def test_prompt_must_precede_assistant_and_match_prompt_ids(self):
        value = record()
        value['segments'][0]['kind'] = 'assistant'
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)
        value = record()
        value['prompt_ids'] = [999]
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)
        value = record()
        value['segments'][1]['kind'] = 'tool'
        value['segments'][1]['train'] = False
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)

    def test_completion_must_equal_nonempty_trainable_assistant_segments(self):
        value = record()
        value['completion_ids'] = [41, EOS]
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)
        for segment in value['segments']:
            segment['train'] = False
        value['completion_ids'] = []
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)

    def test_each_trainable_assistant_turn_requires_eos(self):
        value = record()
        value['segments'][1]['ids'][-1] = 9
        value['completion_ids'][1] = 9
        with self.assertRaises(ValueError):
            schema.validate(value, EOS)

    def test_tokens_counts_and_limits_require_exact_integers(self):
        for token in (True, -1, 1.5, '3'):
            value = record()
            value['prompt_ids'][0] = token
            with self.subTest(token=token), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        for count in (True, 0, -1, 8.0, 7):
            value = record()
            value['n_tokens'] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        for limit in (True, 0, -1, 8.0, 7):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                schema.validate(record(), EOS, max_tokens=limit)

    def test_explicit_large_budget_inspects_without_changing_global_limit(self):
        value = record()
        value['segments'][0]['ids'] = value['prompt_ids'] = [10] * 10000
        value['n_tokens'] = 10005
        old = schema.MAX_TOKENS_MULTITURN
        with patch.object(schema, 'MAX_TOKENS_MULTITURN', 8192):
            with self.assertRaises(ValueError):
                schema.validate(value, EOS)
            schema.validate(value, EOS, max_tokens=32768)
            self.assertEqual(schema.MAX_TOKENS_MULTITURN, 8192)
        self.assertEqual(schema.MAX_TOKENS_MULTITURN, old)

    def test_scalar_metadata_and_sampling_cannot_be_coerced(self):
        for key, bad in (('root_id', 73), ('id', ''), ('model', None), ('source', []),
                         ('tokenizer_template_sha', 'unverified'), ('completion_text', 1), ('sampling', [])):
            value = record()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                schema.validate(value, EOS)
        for key, bad in (('temperature', True), ('top_p', float('nan')), ('max_tokens', False), ('budget', 0)):
            value = record()
            value['sampling'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                schema.validate(value, EOS)


class StrictLoadTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_scalars_and_blank_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            for text in ('{"grade":{"pass":false,"pass":true}}', '{"n":NaN}', '{"n":Infinity}',
                         '{"n":-Infinity}', '{"n":1e999}', 'true', '{}\n\n'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(ValueError):
                    schema.load(path)

    def test_unicode_line_separators_do_not_split_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            value = {'text': 'first\u2028second\u2029third'}
            path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(schema.load(path), [value])


class ConversionProvenanceTests(unittest.TestCase):
    def test_hint_removal_preserves_hinted_history_and_invalidates_grade(self):
        original = record('Follow the task.\nHint: inspect Beta.\n')
        before = copy.deepcopy(original)
        transformed = converter.strip_hint(original, None, 'Hint: inspect Beta.', render_chat=render_fixture)
        self.assertEqual(original, before)
        self.assertTrue(transformed['hinted'])
        self.assertTrue(transformed['hint_stripped'])
        self.assertFalse(transformed['grade']['pass'])
        event = transformed['provenance']['prompt_transformations'][0]
        self.assertTrue(event['original_grade']['pass'])
        self.assertNotEqual(event['original_prompt_sha256'], event['new_prompt_sha256'])
        self.assertNotEqual(event['original_prompt_ids_sha256'], event['new_prompt_ids_sha256'])
        schema.validate_structure(transformed, EOS)
        with self.assertRaises(ValueError):
            schema.validate(transformed, EOS)

    def test_finish_refresh_is_quarantined_and_composed_history_is_preserved(self):
        original = record(converter.OLD_FINISH + 'Hint: inspect Beta.\n')
        stripped = converter.strip_hint(original, None, 'Hint: inspect Beta.', render_chat=render_fixture)
        refreshed = converter.refresh_prompt(stripped, None, render_chat=render_fixture)
        self.assertEqual(len(refreshed['provenance']['prompt_transformations']), 2)
        self.assertTrue(refreshed['hinted'])
        self.assertFalse(refreshed['grade']['pass'])
        schema.validate_structure(refreshed, EOS)

    def test_transformation_refuses_unverified_original_prompt_tokens(self):
        original = record(converter.OLD_FINISH)
        original['segments'][0]['ids'] = [99]
        with self.assertRaises(ValueError):
            converter.refresh_prompt(original, None, render_chat=render_fixture)

    def test_converter_preserves_existing_root_and_does_not_regrade(self):
        original = record()
        value = converter.convert_record(original, 'new-source', TokenizerFixture(), SHA, EOS)
        self.assertEqual(value['root_id'], original['root_id'])
        self.assertEqual(value['grade'], original['grade'])
        self.assertEqual(original['source'], 'synthetic')

    def test_integer_root_requires_explicit_mapping_and_requalification(self):
        original = record()
        original['root_id'] = 73
        with self.assertRaises(ValueError):
            converter.convert_record(original, 'source', TokenizerFixture(), SHA, EOS)
        value = converter.convert_record(original, 'source', TokenizerFixture(), SHA, EOS,
                                         root_mapping={'item-a1': 'mapped/task'})
        self.assertEqual(value['root_id'], 'mapped/task')
        self.assertEqual(value['provenance']['original_root_id'], 73)
        self.assertFalse(value['grade']['pass'])
        schema.validate_structure(value, EOS)

    def test_conversion_cannot_silently_repair_inconsistent_tokens_or_template(self):
        for field, bad in (('n_tokens', 1), ('completion_ids', [99, EOS]),
                           ('prompt_ids', [99]), ('tokenizer_template_sha', 'b' * 64)):
            original = record()
            original[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                converter.convert_record(original, 'source', TokenizerFixture(), SHA, EOS)

    def test_output_is_exclusive_utf8_lf_and_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            converter.write_new_records(path, [record()])
            before = path.read_bytes()
            self.assertTrue(before.endswith(b'\n'))
            self.assertNotIn(b'\r\n', before)
            with self.assertRaises(FileExistsError):
                converter.write_new_records(path, [{'replacement': True}])
            self.assertEqual(path.read_bytes(), before)

    def test_import_and_cli_without_opt_in_do_not_load_model_or_encoding(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'transformers', 'torch', 'encoding'}:
                raise AssertionError('model or gym encoding import before opt-in')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            importlib.reload(converter)
            with patch.object(sys, 'argv', ['from_gym', 'missing.jsonl', '--source', 'fixture', '--out', 'missing-out.jsonl']), \
                 patch.object(Path, 'read_text', side_effect=AssertionError('input read before opt-in')):
                with self.assertRaisesRegex(SystemExit, 'Model execution is disabled'):
                    converter.main()


if __name__ == '__main__':
    unittest.main()
