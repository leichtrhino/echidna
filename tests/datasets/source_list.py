import pathlib

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

