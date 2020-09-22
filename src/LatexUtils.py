from tabulate import tabulate
import re

def print_summary_table(
        method_keys, metric_names, comp_metric_name,
        dframes_manager, file2name, **kwargs):
    data = []
    headers = ['name'] + metric_names
    cmp_index = metric_names.index(comp_metric_name) + 1
    for key in method_keys:
        name = file2name[key]
        row = [name] + [dframes_manager.get_metric(key,k) for k in metric_names]
        data.append(row)
    data.sort(key=lambda r: -r[cmp_index])
    print(tabulate(data, headers=headers, **kwargs))
    

_CELL_NUM_REGEX = re.compile(r'(?<=\&)\s+(\\[a-z]+\{)*\d*\.?\d+\}*(?=\s+(\&|\\))')
_NUM_REGEX = re.compile(r'\d*\.?\d+')

def highlight_topk_column_values(latex_table_string, n_cols, k, command='bold'):
    # print (latex_table_string)
    # print("------------- command = %s ------------" % command)
    if command == 'bold':
        latex_command = ' \\textbf{{{}}} '
    else:
        assert command == 'underline'
        latex_command = ' \\underline{{{}}} '
    is_max = k > 0
    k = abs(k)
    tokens = []
    values = []
    spans = []
    for m in _CELL_NUM_REGEX.finditer(latex_table_string):
        token = m.group().strip()
        tokens.append(token)
        value = float(_NUM_REGEX.search(token).group())
        values.append(value)
        # print(token, ' ', value)
        spans.append(m.span())
    n_rows = len(tokens) // n_cols
    col_topk_values = [set() for _ in range(n_cols)]
    for c in range(n_cols):
        uniq = set()
        for r in range(n_rows):
            uniq.add(values[n_cols * r + c])
        uniq_list = list(uniq)
        uniq_list.sort(reverse=is_max)
        for i in range(min(k, len(uniq_list))):
            col_topk_values[c].add(uniq_list[i])
    output = ''
    offset = 0
    i = 0
    for r in range(n_rows):
        for c in range(n_cols):
            start, end = spans[i]
            output += latex_table_string[offset:start]
            if values[i] in col_topk_values[c]:
                output += latex_command.format(tokens[i])
            else:
                output += ' {} '.format(tokens[i])
            i += 1
            offset = end
    output += latex_table_string[offset:]
    return output
    
    
def print_best_experiments_summary_table(
        best_keys, best_exp_names, dframes_manager,
        metric_names=(),
        metric_aliases=None,
        bold_top_x=None,
        underline_top_x=None,
        sortby=None,
        **kwargs):
    
    assert len(best_keys) == len(best_exp_names)
    metric_aliases = metric_aliases or metric_names

    if sortby:
        def _get_value(index):
            exp_key = best_keys[index]
            return dframes_manager.get_metric(exp_key, sortby)
        indexes = list(range(len(best_keys)))
        indexes.sort(key=_get_value, reverse=True)
        best_keys = [best_keys[i] for i in indexes]
        best_exp_names = [best_exp_names[i] for i in indexes]
    
    data = []
    headers = ['name']
    headers.extend(metric_aliases)
            
    for dom_key, dom_name in zip(best_keys, best_exp_names):
        row = [dom_name]
        for name in metric_names:
            row.append(dframes_manager.get_metric(dom_key, name))
        data.append(row)

    tab = tabulate(data, headers=headers, **kwargs).replace(' 0.', '  .')
    
    should_bold = bold_top_x is not None
    should_underline = underline_top_x is not None
    if should_bold or should_underline:
        n_cols = len(metric_names)
        if bold_top_x is not None:
            tab = highlight_topk_column_values(tab, n_cols, bold_top_x, command='bold')
        if underline_top_x is not None:
            tab = highlight_topk_column_values(tab, n_cols, underline_top_x, command='underline')
    print(tab)