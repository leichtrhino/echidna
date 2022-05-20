
import unittest
import pathlib
import tempfile
import math
import itertools
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt

from echidna.datasets import sampler

from .source_list import source_list_a, source_list_b

class TestSampler(unittest.TestCase):

    def test_sample_a_num_sources_a(self):
        # 1. num_sources == number of categories in dataset
        #    category repetition: not allowed
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories=None,
            num_sources=2,
            category_repetition=False,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 2)

    def test_sample_a_num_sources_b(self):
        # 2. num_sources > number of categories in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_a,
                sr=16000,
                duration=1.0,
                source_categories=None,
                num_sources=3,
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True

        self.assertTrue(is_error, 'mepit in this condition should raise error')
        is_error = False

    def test_sample_a_num_sources_c(self):
        # 3. num_sources > number of categories in dataset
        #    category repetition: allowed
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories=None,
            num_sources=3,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 3)

    def test_sample_a_category_set_a(self):
        # 4.a source_categories is given as set
        #     category repetition: not allowed
        #     num_caetgories is given
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster'},
            num_sources=2,
            category_repetition=False,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            set(source_list_a[s]['category'] for s in sources),
            {'dog', 'rooster'}
        )

        # 4.b source_categories is given as set
        #     category repetition: not allowed
        #     num_caetgories not given
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster'},
            category_repetition=False,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            set(source_list_a[s]['category'] for s in sources),
            {'dog', 'rooster'}
        )

    def test_sample_a_category_set_b(self):
        # 5. source_categories is given as set
        #    and some are not in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_a,
                sr=16000,
                duration=1.0,
                source_categories={'dog', 'rooster', 'piano'},
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit on this configuration should fail')

    def test_sample_a_category_set_c(self):
        # 6. source_categories is given as set
        #    num_categories > number of categories in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_a,
                sr=16000,
                duration=1.0,
                source_categories={'dog', 'rooster'},
                num_sources=3,
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit on this configuration should fail')


    def test_sample_a_category_set_d(self):
        # 7. source_categories is given as set
        #    num_categories > number of categories in dataset
        #    category repetition: allowed
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster'},
            num_sources=3,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 3)
        self.assertEqual(
            set(source_list_a[s]['category'] for s in source_list_a),
            {'dog', 'rooster'}
        )

    def test_sample_a_category_list_a(self):
        # 8. source_categories is given as list
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories=['dog', 'rooster'],
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(
            [source_list_a[s]['category'] for s in sources],
            ['dog', 'rooster']
        )

        # error on num_sources is fed
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_a,
                sr=16000,
                duration=1.0,
                source_categories=['dog', 'rooster'],
                num_sources=2,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')


    def test_sample_a_category_list_b(self):
        # 9. source_categories is given as list
        #    the list includes category that the dataset does not include
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_a,
                sr=16000,
                duration=1.0,
                source_categories=['dog', 'rooster', 'piano'],
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_a_category_list_c(self):
        # 10. source_categories is given as list
        #     the list contains repetition
        mepit = sampler.Sampler(
            source_list_a,
            sr=16000,
            duration=1.0,
            source_categories=['dog', 'rooster', 'dog'],
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(
            [source_list_a[s]['category'] for s in sources],
            ['dog', 'rooster', 'dog']
        )

    def test_sample_b_num_sources_a(self):
        # 1. num_sources = 2: pass (dog, rooster)
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories=None,
            num_sources=2,
            category_repetition=False,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            len(set(source_list_b[s]['category'] for s in sources)), 2
        )

    def test_sample_b_num_sources_b(self):
        # 2. num_sources = 3: fail
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_b,
                sr=16000,
                duration=1.0,
                source_categories=None,
                num_sources=3,
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True

        self.assertTrue(is_error, 'mepit in this condition should raise error')
        is_error = False

    def test_sample_b_num_sources_c(self):
        # 3. num_sources = 4 with rep.: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories=None,
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 4)
        self.assertLessEqual(
            len(set(source_list_b[s]['category'] for s in sources)), 4
        )

    def test_sample_b_category_set_a(self):
        # 4. category_set = {'dog', 'rooster', 'cow'}: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            category_repetition=False,
            splits=None,
        )

        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            num_sources=2,
            category_repetition=False,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            len(set(source_list_b[s]['category'] for s in sources)),
            2
        )

    def test_sample_b_category_set_b(self):
        # 5. category_set = {'dog', 'rooster', 'pig'}: fail
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_b,
                sr=16000,
                duration=1.0,
                source_categories={'dog', 'rooster', 'pig'},
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit on this configuration should fail')

    def test_sample_b_category_set_c(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 4)
        self.assertLessEqual(
            len(set(source_list_b[s]['category'] for s in sources)),
            4
        )

    def test_sample_b_category_set_d(self):
        # 7. category_set = {'dog',}, ns = 2 with rep.: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog',},
            num_sources=2,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(
            [source_list_b[s]['category'] for s in sources],
            ['dog', 'dog']
        )

    def test_sample_b_category_set_e(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 4)
        self.assertLessEqual(
            len(set(source_list_b[s]['category'] for s in sources)),
            4
        )

    def test_sample_b_category_set_f(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            category_weight={'dog': 1.0, 'rooster': 2.0, 'cow': 2.0},
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(len(sources), 4)
        self.assertLessEqual(
            len(set(source_list_b[s]['category'] for s in sources)),
            4
        )

    def test_sample_b_category_set_g(self):
        # 8. category_set = {'dog',}, ns = 3 with rep.: fail
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_b,
                sr=16000,
                duration=1.0,
                source_categories={'dog',},
                num_sources=3,
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit on this configuration should fail')


    def test_sample_b_category_list_a(self):
        # 9. category_list = ['dog', 'rooster', 'dog']: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories=['dog', 'rooster', 'dog'],
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(
            [source_list_b[s]['category'] for s in sources],
            ['dog', 'rooster', 'dog']
        )

    def test_sample_b_category_list_b(self):
        # 10. category_list = ['dog', 'rooster', 'cow']: pass
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        )

        sources = mepit.sample_source()
        self.assertEqual(
            [source_list_b[s]['category'] for s in sources],
            ['dog', 'rooster', 'cow']
        )

    def test_sample_b_category_list_c(self):
        # 11. category_list = ['dog', 'rooster', 'pig']: fail
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_b,
                sr=16000,
                duration=1.0,
                source_categories=['dog', 'rooster', 'pig'],
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_b_category_list_d(self):
        # 12. category_list = ['dog', 'dog', 'dog']: fail
        is_error = False
        try:
            mepit = sampler.Sampler(
                source_list_b,
                sr=16000,
                duration=1.0,
                source_categories=['dog', 'dog', 'dog'],
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_fetch(self):
        duration = 4.
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=duration,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        ).time_stretch_range((0.8, 1.2))
        data, sources = mepit.sample_data()

        # test data
        waveform, midi = data['waveform'], data['midi']
        self.assertEqual(midi, None)
        self.assertEqual(
            waveform.shape, torch.Size((3, int(16000*duration))))

        # test metadata
        self.assertEqual(
            [s['category'] for s in sources], ['dog', 'rooster', 'cow'])
        # same tracks should have same start, end, and transform parameters
        self.assertEqual(sources[0]['track'], sources[1]['track'])
        self.assertEqual(
            round(sources[0]['start'], 2), round(sources[1]['start'], 2))
        self.assertEqual(
            round(sources[0]['end'], 2), round(sources[1]['end'], 2))
        self.assertEqual(
            round(sources[0]['tf_params']['time_stretch_rate'], 2),
            round(sources[1]['tf_params']['time_stretch_rate'], 2))
        # different tracks should have different start, end, and params
        self.assertNotEqual(
            round(sources[0]['start'], 2), round(sources[2]['start'], 2))
        self.assertNotEqual(
            round(sources[0]['end'], 2), round(sources[2]['end'], 2))
        self.assertNotEqual(
            round(sources[0]['tf_params']['time_stretch_rate'], 2),
            round(sources[2]['tf_params']['time_stretch_rate'], 2))

    def test_fetch_dogornot(self):
        duration = 4.
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=duration,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        ).time_stretch_range((0.8, 1.2))
        data, sources = mepit.sample_data()

        # test data
        waveform, midi = data['waveform'], data['midi']
        self.assertEqual(midi, None)
        self.assertEqual(
            waveform.shape, torch.Size((3, int(16000*duration))))

        # test metadata
        self.assertEqual(
            [s['category'] for s in sources], ['dog', 'rooster', 'cow'])
        # same tracks should have same start, end, and transform parameters
        self.assertEqual(sources[0]['track'], sources[1]['track'])
        self.assertEqual(
            round(sources[0]['start'], 3), round(sources[1]['start'], 3))
        self.assertEqual(
            round(sources[0]['end'], 3), round(sources[1]['end'], 3))
        self.assertEqual(
            round(sources[0]['tf_params']['time_stretch_rate'], 3),
            round(sources[1]['tf_params']['time_stretch_rate'], 3))
        # different tracks should have different start, end, and params
        self.assertNotEqual(
            round(sources[0]['start'], 3), round(sources[2]['start'], 3))
        self.assertNotEqual(
            round(sources[0]['end'], 3), round(sources[2]['end'], 3))
        self.assertNotEqual(
            round(sources[0]['tf_params']['time_stretch_rate'], 3),
            round(sources[2]['tf_params']['time_stretch_rate'], 3))

    def test_freeze(self):
        mepit = sampler.Sampler(
            source_list_b,
            sr=16000,
            duration=5.,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            mepit.freeze(tmpdirname, 5, 4)

            # returns metadata
            fmepit = sampler.FrozenSamples(tmpdirname)
            loader = torch.utils.data.DataLoader(
                fmepit,
                2,
                collate_fn=sampler.collate_samples
            )
            d, s = next(iter(loader))
            w, m = d['waveform'], d['midi']
            self.assertEqual(m, None)
            self.assertEqual(w.shape, torch.Size((2, 3, int(16000*5.))))
            self.assertEqual(len(s), 2)
            self.assertEqual(len(s[0]), 3)
            self.assertEqual(
                [_s['category'] for _s in s[0]], ['dog', 'rooster', 'cow'])
