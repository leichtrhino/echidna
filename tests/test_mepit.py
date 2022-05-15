import unittest
import pathlib
import tempfile
import math
import itertools
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from chimerau import datasets as ds

# dataset A: two categories, no track
source_list_a = {
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-100032-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '1',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-110389-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '1',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-114280-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '2',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-114587-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '2',
        'track': None,
    },

    pathlib.Path('~/Desktop/ESC-50-master/audio/1-26806-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '1',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-27724-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '1',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-100786-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '2',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-65750-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '2',
        'track': None,
    },
}



# dataset B: some have categories, few have track
# num_sources = 3: pass (dog, rooster, cow).expanduser()
# num_sources = 4: fail
# num_sources = 4 with rep.: pass
# category_set = {'dog', 'rooster', 'cow'}: pass
# category_set = {'dog', 'rooster', 'pig'}: fail
# category_set = {'dog', 'rooster', 'cow'}, ns = 4 with rep.: pass
# category_set = {'dog',}, ns = 2 with rep.: pass
# category_set = {'dog',}, ns = 3 with rep.: fail
# category_list = ['dog', 'rooster', 'dog']: pass
# category_list = ['dog', 'rooster', 'cow']: pass
# category_list = ['dog', 'rooster', 'pig']: fail
# category_list = ['dog', 'dog', 'dog']: fail
source_list_b = {
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-100032-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '1',
        'track': 'A',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-110389-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '1',
        'track': 'A',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-114280-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '2',
        'track': 'B',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-114587-A-0.wav').expanduser(): {
        'category': 'dog',
        'split': '2',
        'track': 'B',
    },

    pathlib.Path('~/Desktop/ESC-50-master/audio/1-26806-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '1',
        'track': 'A',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-27724-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '1',
        'track': 'A',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-100786-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '2',
        'track': 'B',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/2-65750-A-1.wav').expanduser(): {
        'category': 'rooster',
        'split': '2',
        'track': 'B',
    },

    pathlib.Path('~/Desktop/ESC-50-master/audio/1-208757-A-2.wav').expanduser(): {
        'category': 'pig',
        'split': '1',
        'track': 'C',
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-208757-B-2.wav').expanduser(): {
        'category': 'pig',
        'split': '1',
        'track': 'C',
    },

    pathlib.Path('~/Desktop/ESC-50-master/audio/1-16568-A-3.wav').expanduser(): {
        'category': 'cow',
        'split': '1',
        'track': None,
    },
    pathlib.Path('~/Desktop/ESC-50-master/audio/1-202111-A-3.wav').expanduser(): {
        'category': 'cow',
        'split': '1',
        'track': None,
    },
}

class TestMergeActivation(unittest.TestCase):
    def test_merge_activation_create(self):
        base_list = [
            (0*16000, 25*16000, []),
        ]
        x = torch.zeros(25*16000)
        x[3*16000:6*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 6*16000-3*16000))
        x[9*16000:12*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 12*16000-9*16000))
        x[16*16000:19*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 19*16000-16*16000))

        ds.merge_activation(base_list, x, tag='x')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 25*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [[], ['x'], [], ['x'], [], ['x'], []])

    def test_merge_activation_update(self):
        base_list = [
            (0*16000, 5*16000, []),
            (5*16000, 10*16000, ['a']),
            (10*16000, 15*16000, []),
            (15*16000, 20*16000, ['a']),
            (20*16000, 25*16000, []),
        ]
        x = torch.zeros(25*16000)
        x[3*16000:6*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 6*16000-3*16000))
        x[9*16000:12*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 12*16000-9*16000))
        x[16*16000:19*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 19*16000-16*16000))

        y = torch.zeros(25*16000)
        y[14*16000:21*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 21*16000-14*16000))

        ds.merge_activation(base_list, x, tag='x')
        ds.merge_activation(base_list, y, tag='y')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 25*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [[], ['x'], ['a', 'x'], ['a'], ['a', 'x'], ['x'],
                          [], ['y'], ['a', 'y'], ['a', 'x', 'y'],
                          ['a', 'y'], ['y'], []])

    def test_merge_activation_boundary(self):
        base_list = [
            (0*16000, 5*16000, []),
        ]
        x = torch.zeros(5*16000)
        x[0*16000:1*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 1*16000-0*16000))
        x[4*16000:5*16000] = \
            torch.sin(torch.linspace(0, 2*torch.pi, 5*16000-4*16000))

        ds.merge_activation(base_list, x, tag='x')

        self.assertEqual(base_list[0][0], 0)
        self.assertEqual(base_list[-1][1], 5*16000)
        self.assertEqual([b[0] for b in base_list[1:]],
                         [b[1] for b in base_list[:-1]])
        self.assertEqual([b[2] for b in base_list],
                         [['x'], [], ['x']])

class TestSampleValidator(unittest.TestCase):

    def test_sample_a_num_sources_a(self):
        # 1. num_sources == number of categories in dataset
        #    category repetition: not allowed
        source_list, track_category_dict =\
            ds.validate_source_list(
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
                ds.validate_source_list(
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
            ds.validate_source_list(
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
            ds.validate_source_list(
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
            ds.validate_source_list(
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
                ds.validate_source_list(
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
                ds.validate_source_list(
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
            ds.validate_source_list(
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
            ds.validate_source_list(
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
                ds.validate_source_list(
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
                ds.validate_source_list(
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
            ds.validate_source_list(
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
            ds.validate_source_list(
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
                ds.validate_source_list(
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
            ds.validate_source_list(
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
            ds.validate_source_list(
                source_list_b,
                source_categories={'dog', 'rooster', 'cow'},
                category_repetition=False,
                splits=None,
            )

        source_list, track_category_dict =\
            ds.validate_source_list(
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
                ds.validate_source_list(
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
            ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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
            source_list, track_category_dict = ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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
            source_list, track_category_dict = ds.validate_source_list(
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
            source_list, track_category_dict = ds.validate_source_list(
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
        source_list, track_category_dict = ds.validate_source_list(
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


class TestMEPIT(unittest.TestCase):

    def test_sample_a_num_sources_a(self):
        # 1. num_sources == number of categories in dataset
        #    category repetition: not allowed
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
            source_list_b,
            sr=16000,
            duration=1.0,
            source_categories={'dog', 'rooster', 'cow'},
            category_repetition=False,
            splits=None,
        )

        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
            mepit = ds.MEPIT(
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
        mepit = ds.MEPIT(
            source_list_b,
            sr=16000,
            duration=duration,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
            time_stretch_range=(0.8, 1.2),
        )
        data, sources, partition = mepit.sample_data()

        # test data
        self.assertEqual(data.shape, torch.Size((3, int(16000*duration))))

        # test partition
        self.assertEqual(len(partition), 3-1)

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
        partition_algorithm = lambda x, y: \
            ds.partition_voice_or_not(x, y, {'dog'})
        mepit = ds.MEPIT(
            source_list_b,
            sr=16000,
            duration=duration,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
            time_stretch_range=(0.8, 1.2),
            partition_algorithm=partition_algorithm
        )
        data, sources, partition = mepit.sample_data()

        # test data
        self.assertEqual(data.shape, torch.Size((3, int(16000*duration))))

        # test partition
        self.assertEqual(set(partition),
                         {((0,), (1,)), ((0,), (2,)), ((0,), (1, 2))})

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
        mepit = ds.MEPIT(
            source_list_b,
            sr=16000,
            duration=5.,
            source_categories=['dog', 'rooster', 'cow'],
            splits=None,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            mepit.freeze(tmpdirname, 5, 4)

            # 1. no metadata
            fmepit = ds.FrozenMEPIT(tmpdirname,
                                    out_partition_metadata=False,
                                    out_source_metadata=False)
            self.assertEqual(len(fmepit), 5)
            loader = torch.utils.data.DataLoader(fmepit, 2)
            d = next(iter(loader))
            self.assertEqual(d.shape, torch.Size((2, 3, int(16000*5.))))

            # 2. returns metadata
            fmepit = ds.FrozenMEPIT(tmpdirname,
                                    out_partition_metadata=True,
                                    out_source_metadata=True)
            loader = torch.utils.data.DataLoader(
                fmepit,
                2,
                collate_fn=lambda l: (
                    torch.stack([m[0] for m in l], dim=0),
                    [m[1] for m in l],
                    [m[2] for m in l],
                )
            )
            d, s, p = next(iter(loader))
            self.assertEqual(d.shape, torch.Size((2, 3, int(16000*5.))))
            self.assertEqual(len(p), 2)
            self.assertEqual(len(p[0]), 3-1)
            self.assertEqual(len(s), 2)
            self.assertEqual(len(s[0]), 3)
            self.assertEqual(
                [_s['category'] for _s in s[0]], ['dog', 'rooster', 'cow'])

            # 3. mix
            fmepit = ds.FrozenMEPITMixByPartition(tmpdirname)
            self.assertEqual(len(fmepit), (3-1)*5)
            loader = torch.utils.data.DataLoader(fmepit, 2)
            d = next(iter(loader))
            self.assertEqual(d.shape, torch.Size((2, 2, int(16000*5.))))


if __name__ == '__main__':
    unittest.main()
