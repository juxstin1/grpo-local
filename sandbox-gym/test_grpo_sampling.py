"""Behavior checks for bounded, error-safe dynamic group selection."""

import random
import unittest
from collections import Counter
from types import SimpleNamespace

from grpo_sampling import DynamicGroupSampler


def task(family, index):
    return SimpleNamespace(family=family, name=f"{family}-{index}")


def group(passes, size=8):
    return [SimpleNamespace(reward=float(i < passes)) for i in range(size)]


class DynamicSamplingTests(unittest.TestCase):
    def test_refills_all_family_quotas_without_reusing_a_task(self):
        quotas = {"find_h": 4, "edit": 2, "mhqa": 2}
        tasks = [task(family, i) for family, n in
                 (("find_h", 16), ("edit", 10), ("mhqa", 10)) for i in range(n)]
        sampler = DynamicGroupSampler(tasks, quotas, 8, 2.0, 36, random.Random(17))
        seen = set()
        while not sampler.ready:
            candidates = sampler.next_candidates()
            self.assertTrue(candidates, "unexpected exhaustion of the candidate budget")
            for t in candidates:
                key = (t.family, t.name)
                self.assertNotIn(key, seen)
                seen.add(key)
                index = int(t.name.rsplit("-", 1)[1])
                if index % 5 == 0:
                    outputs = group(0)
                elif index % 5 == 1:
                    outputs = group(8)
                elif index % 5 == 2:
                    outputs = group(4)[:-1] + ["request timeout"]
                else:
                    outputs = group(4)
                sampler.observe(t, outputs)
        selected = sampler.selected_groups()
        self.assertEqual(Counter(t.family for t, _ in selected), quotas)
        self.assertEqual(len(selected), 8)
        self.assertGreater(sampler.resampled_groups, 0)
        self.assertEqual(sampler.candidate_groups, len(seen))
        self.assertGreater(sampler.dead_groups, 0)
        self.assertGreater(sampler.error_groups, 0)

    def test_all_pass_groups_exhaust_pool_without_claiming_a_live_batch(self):
        sampler = DynamicGroupSampler([task("find_h", i) for i in range(8)],
                                      {"find_h": 8}, 8, 2.0, 16, random.Random(1))
        for t in sampler.next_candidates():
            self.assertEqual(sampler.observe(t, group(8)), "dead")
        self.assertFalse(sampler.ready)
        self.assertEqual(sampler.next_candidates(), [])
        self.assertEqual(sampler.all_pass_groups, 8)
        self.assertEqual(sampler.dead_groups, 8)
        with self.assertRaises(RuntimeError):
            sampler.selected_groups()

    def test_request_error_does_not_become_a_zero_reward(self):
        t = task("mhqa", 1)
        sampler = DynamicGroupSampler([t], {"mhqa": 1}, 8, 2.0, 8, random.Random(1))
        self.assertEqual(sampler.next_candidates(), [t])
        self.assertEqual(sampler.observe(t, group(7)[:-1] + ["bridge down"]), "error")
        self.assertEqual(sampler.rewards, [])
        self.assertEqual(sampler.dead_groups, 0)
        self.assertEqual(sampler.error_groups, 1)
        self.assertFalse(sampler.ready)

    def test_eight_positive_rewards_are_dead_even_if_values_vary(self):
        t = task("mhqa", 2)
        sampler = DynamicGroupSampler([t], {"mhqa": 1}, 8, 2.0, 8, random.Random(1))
        sampler.next_candidates()
        varied = [SimpleNamespace(reward=0.5 + i / 100) for i in range(8)]
        self.assertEqual(sampler.observe(t, varied), "dead")
        self.assertEqual(sampler.all_pass_groups, 1)
        self.assertFalse(sampler.ready)


if __name__ == "__main__":
    unittest.main()
