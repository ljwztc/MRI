from pathlib import Path
import random

data_path = '/hdd2/lj/fastMRI/knee'

preprocess_files = list(Path(data_path, 'multicoil_train').iterdir()) + list(Path(data_path, 'multicoil_val').iterdir())

random.shuffle(preprocess_files)
print(len(preprocess_files))
size = len(preprocess_files) // 10

groups = []
for _ in range(10):
    groups.append([])

for i in range(10):
    if i < 6:
        filename = f'train_{i}.txt'
    elif i in [6, 7]:
        i -= 6
        filename = f'val_{i}.txt'
    elif i in [8, 9]:
        i -= 8
        filename = f'test_{i}.txt'
    with open(filename, 'w') as f:
        for _ in range(size): 
            elem = preprocess_files.pop()
            elem = '/'.join(str(elem).split('/')[-2:])
            f.write(elem + '\n') 