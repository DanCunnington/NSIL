from examples.hitting_sets.templates import templates
from os.path import join
import ast
folders = ['hs', 'chs']
files = ['fashion_mnist_train.csv', 'fashion_mnist_validation.csv', 'fashion_mnist_test.csv']

new_header = 'i1,i2,i3,i4,label,template'
for fol in folders:
    for fil in files:
        new_lines = [new_header]
        with open(join('old_data', fol, fil), 'r') as f:
            csv_f = f.readlines()[1:]
        for line in csv_f:
            sets, label = line.rstrip().split('|')
            sets = ast.literal_eval(sets)
            target_template = '{'
            for subset in sets:
                target_template += '{'
                for element in subset:
                    target_template += '_,'
                target_template = target_template[:-1] + '}|'
            target_template = target_template[:-1] + '}'
            template_idx = templates.index(target_template)
            image_idxs = [item for sublist in sets for item in sublist]
            image_idxs = ','.join(str(i) for i in image_idxs)
            new_line = f'{image_idxs},{label},{template_idx}'
            new_lines.append(new_line)

        with open(join('data', fol, fil), 'w') as outf:
            for l in new_lines:
                outf.write(f'{l}\n')
