
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

class TestSampleValidator(unittest.TestCase):

    def test_sample_a_num_sources_a(self):
        # 1. num_sources == number of categories in dataset
        #    category repetition: not allowed
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories=None,
                num_sources=2,
                category_repetition=False,
                splits=None,
            )
        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

    def test_sample_a_num_sources_b(self):
        # 2. num_sources > number of categories in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_a,
                    source_categories=None,
                    num_sources=3,
                    category_repetition=False,
                    splits=None,
                )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')
        is_error = False

    def test_sample_a_num_sources_c(self):
        # 3. num_sources > number of categories in dataset
        #    category repetition: allowed
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories=None,
                num_sources=3,
                category_repetition=True,
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

    def test_sample_a_category_set_a(self):
        # 4.a source_categories is given as set
        #     category repetition: not allowed
        #     num_caetgories is given
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories={'dog', 'rooster'},
                num_sources=2,
                category_repetition=False,
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

        # 4.b source_categories is given as set
        #     category repetition: not allowed
        #     num_caetgories not given
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories={'dog', 'rooster'},
                category_repetition=False,
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

    def test_sample_a_category_set_b(self):
        # 5. source_categories is given as set
        #    and some are not in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_a,
                    source_categories={'dog', 'rooster', 'piano'},
                    category_repetition=False,
                    splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_a_category_set_c(self):
        # 6. source_categories is given as set
        #    num_categories > number of categories in dataset
        #    category repetition: not allowed
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_a,
                    source_categories={'dog', 'rooster'},
                    num_sources=3,
                    category_repetition=False,
                    splits=None,
                )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')


    def test_sample_a_category_set_d(self):
        # 7. source_categories is given as set
        #    num_categories > number of categories in dataset
        #    category repetition: allowed
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories={'dog', 'rooster'},
                num_sources=3,
                category_repetition=True,
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

    def test_sample_a_category_list_a(self):
        # 8. source_categories is given as list
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories=['dog', 'rooster'],
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

        # error on num_sources is fed
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_a,
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
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_a,
                    source_categories=['dog', 'rooster', 'piano'],
                    splits=None,
                )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_a_category_list_c(self):
        # 10. source_categories is given as list
        #     the list contains repetition
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_a,
                source_categories=['dog', 'rooster', 'dog'],
                splits=None,
            )

        self.assertEqual(source_list, source_list_a)
        self.assertEqual(track_category_dict.keys(),
                         set([(None, 'dog'), (None, 'rooster')]))

    def test_sample_b_num_sources_a(self):
        # 1. num_sources = 2: pass (dog, rooster)
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_b,
                source_categories=None,
                num_sources=2,
                category_repetition=False,
                splits=None,
            )

        self.assertEqual(source_list, source_list_b)
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             ('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_num_sources_b(self):
        # 2. num_sources = 3: fail
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_b,
                    source_categories=None,
                    num_sources=3,
                    category_repetition=False,
                    splits=None,
                )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_b_num_sources_c(self):
        # 3. num_sources = 4 with rep.: pass
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_b,
                source_categories=None,
                num_sources=4,
                category_repetition=True,
                splits=None,
            )

        self.assertEqual(source_list, source_list_b)
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             ('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_set_a(self):
        # 4. category_set = {'dog', 'rooster', 'cow'}: pass
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_b,
                source_categories={'dog', 'rooster', 'cow'},
                category_repetition=False,
                splits=None,
            )

        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_b,
                source_categories={'dog', 'rooster', 'cow'},
                num_sources=2,
                category_repetition=False,
                splits=None,
            )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster', 'cow')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_set_b(self):
        # 5. category_set = {'dog', 'rooster', 'pig'}: fail
        is_error = False
        try:
            source_list, track_category_dict =\
                sampler.validate_source_list(
                    source_list_b,
                    source_categories={'dog', 'rooster', 'pig'},
                    category_repetition=False,
                    splits=None,
                )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_b_category_set_c(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        source_list, track_category_dict =\
            sampler.validate_source_list(
                source_list_b,
                source_categories={'dog', 'rooster', 'cow'},
                num_sources=4,
                category_repetition=True,
                splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster', 'cow')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_set_d(self):
        # 7. category_set = {'dog',}, ns = 2 with rep.: pass
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_b,
            source_categories={'dog',},
            num_sources=2,
            category_repetition=True,
            splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog',)))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             #('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             #(None, 'cow'),
                         ]))

    def test_sample_b_category_set_e(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_b,
            source_categories={'dog', 'rooster', 'cow'},
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster', 'cow')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_set_f(self):
        # 6. category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_b,
            source_categories={'dog', 'rooster', 'cow'},
            category_weight={'dog': 1.0, 'rooster': 2.0, 'cow': 2.0},
            num_sources=4,
            category_repetition=True,
            splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster', 'cow')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_set_g(self):
        # 8. category_set = {'dog',}, ns = 3 with rep.: fail
        is_error = False
        try:
            source_list, track_category_dict = sampler.validate_source_list(
                source_list_b,
                source_categories={'dog',},
                num_sources=3,
                category_repetition=False,
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')


    def test_sample_b_category_list_a(self):
        # 9. category_list = ['dog', 'rooster', 'dog']: pass
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_b,
            source_categories=['dog', 'rooster', 'dog'],
            splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             #(None, 'cow'),
                         ]))

    def test_sample_b_category_list_b(self):
        # 10. category_list = ['dog', 'rooster', 'cow']: pass
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_b,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_b.items()
                              if v['category'] in ('dog', 'rooster', 'cow')))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'), ('B', 'dog'),
                             ('A', 'rooster'), ('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                         ]))

    def test_sample_b_category_list_c(self):
        # 11. category_list = ['dog', 'rooster', 'pig']: fail
        is_error = False
        try:
            source_list, track_category_dict = sampler.validate_source_list(
                source_list_b,
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
            source_list, track_category_dict = sampler.validate_source_list(
                source_list_b,
                source_categories=['dog', 'dog', 'dog'],
                splits=None,
            )
        except:
            is_error = True
        self.assertTrue(is_error, 'mepit in this condition should fail')

    def test_sample_c_category_list_a(self):
        # 13. category_list = ['dog', 'rooster', 'sheep']: pass
        source_list_c = {
            pathlib.Path('~/Desktop/ESC-50-master/audio/xxxxxxxx.wav').expanduser(): {
                'category': 'sheep',
                'split': '1',
                'track': 'A',
            },
        }
        source_list_c.update(source_list_b)
        source_list, track_category_dict = sampler.validate_source_list(
            source_list_c,
            source_categories=['dog', 'rooster', 'sheep', 'cow'],
            splits=None,
            check_track_strictly=False,
        )

        self.assertEqual(source_list,
                         dict((k, v) for k, v in source_list_c.items()
                              if v['category'] in ('dog', 'rooster', 'cow', 'sheep')
                              and (v['track'] == 'A' or v['track'] is None)))
        self.assertEqual(track_category_dict.keys(),
                         set([
                             ('A', 'dog'),
                             #('B', 'dog'),
                             ('A', 'rooster'),
                             #('B', 'rooster'),
                             #('C', 'pig'),
                             (None, 'cow'),
                             ('A', 'sheep'),
                         ]))

