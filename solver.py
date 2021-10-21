import re
from collections import OrderedDict
import argparse
import sys

bic, imp, neg, con, dis = "<=>", "=>", "!", "&", "|"
verbose, debug = False, False
precedence = {bic: 1, imp: 2, dis: 3, con: 4, neg: 5}
symbol = re.compile(r"([a-zA-Z0-9]|_)+")
operator = re.compile(f"<=>|=>|!|&|\\|")
left_brackets = re.compile(r"\(|\[|\{")
right_brackets = re.compile(r"\)|\]|\}")
brackets = re.compile(r"\(|\[|\{|\)|\]|\}")
mode = None


def print_verbose(*arg, **kwargs):
    if verbose:
        print(*arg, **kwargs)


def print_debug(*arg, **kwargs):
    if debug:
        print(*arg, **kwargs)


class TreeNode:
    def __init__(self, children=None):
        self.children = children

    def apply_to_children(self, func, *args, **kwargs):
        for c in self.children:
            func(c, *args, **kwargs)

    def is_leaf(self):
        return False


class Binary(TreeNode):
    def __init__(self, left, right):
        super().__init__(children=[left, right])


class Unary(TreeNode):
    def __init__(self, operand):
        super().__init__(children=operand)


class Atom(TreeNode):
    def __init__(self, var_symbol):
        super().__init__()
        self.symbol = var_symbol

    def eval(self, truth_values):
        return truth_values[self.symbol]

    def is_leaf(self):
        return True


class Biconditional(Binary):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.symbol = bic

    def eval(self, truth_values):
        if self.children[0].eval(truth_values) == self.children[1].eval(truth_values):
            return True
        return False


class Implication(Binary):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.symbol = imp

    def eval(self, truth_values):
        if self.children[0].eval(truth_values) and not self.children[1].eval(truth_values):
            return False
        return True


class Negation(Unary):
    def __init__(self, operand):
        super().__init__(operand)
        self.symbol = neg

    def apply_to_children(self, func, *args, **kwargs):
        func(self.children, *args, **kwargs)

    def eval(self, truth_values):
        return not self.children.eval(truth_values)


class Disjunction(Binary):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.symbol = dis

    def eval(self, truth_values):
        return self.children[0].eval(truth_values) or self.children[1].eval(truth_values)


class Conjunction(Binary):
    def __init__(self, left, right):
        super().__init__(left, right)
        self.symbol = con

    def eval(self, truth_values):
        return self.children[0].eval(truth_values) and self.children[1].eval(truth_values)


def is_cnf(node):
    if node.is_leaf() or (node.symbol == neg and node.children.is_leaf()):
        return True
    else:
        cnf = True
        if node.symbol == neg:
            return False  # The negation is not next to a symbol
        for child_node in node.children:
            if not child_node.is_leaf() and node.symbol == dis and child_node.symbol == con:
                return False  # If we find a disjunction over a conjunction then return false
            cnf = cnf and is_cnf(child_node)
        return cnf


def inorder_traversal(root_node, level=0):
    if root_node.is_leaf():
        print_verbose(root_node.symbol, end=' ')
    elif issubclass(type(root_node), Unary):
        print_verbose(root_node.symbol, end='')
        inorder_traversal(root_node.children, level=level + 1)
    else:
        if level != 0:
            print_verbose("(", end='')
        inorder_traversal(root_node.children[0], level=level + 1)
        print_verbose(root_node.symbol, end=' ')
        inorder_traversal(root_node.children[1], level=level + 1)
        if level != 0:
            print_verbose(")", end='')


def print_cnf_clauses(root_node):
    if root_node.is_leaf():
        print(root_node.symbol, end='')
    elif issubclass(type(root_node), Unary):
        print(root_node.symbol, end='')
        print_cnf_clauses(root_node.children)
    elif re.match(con, root_node.symbol):
        print_cnf_clauses(root_node.children[0])
        print("\n", end='')
        print_cnf_clauses(root_node.children[1])
    elif re.match(dis, root_node.symbol):
        print_cnf_clauses(root_node.children[0])
        print(" ", end='')
        print_cnf_clauses(root_node.children[1])
    elif operator.match(root_node.symbol):
        print(f"ERROR {inorder_traversal(root_node)} not in CNF")
    else:
        print("Something went wrong!")


def make_parse_tree(exp):
    _tokens = infix_to_prefix(exp)[::-1]

    def parse_prefix_exp():
        node = None
        if symbol.match(_tokens[-1]):
            return Atom(_tokens.pop())
        elif operator.match(_tokens[-1]):
            op = _tokens.pop()
            if op == neg:
                a = parse_prefix_exp()
                node = Negation(a)
            else:
                a = parse_prefix_exp()
                b = parse_prefix_exp()
                if op == imp:
                    node = Implication(a, b)
                elif op == bic:
                    node = Biconditional(a, b)
                elif op == con:
                    node = Conjunction(a, b)
                elif op == dis:
                    node = Disjunction(a, b)
            return node

    return parse_prefix_exp()


def tokenize(tokens):
    _tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '<' and tokens[i + 1] == '=' and tokens[i + 2] == '>':
            _tokens.append(bic)
            i += 2
        elif tokens[i] == '=' and tokens[i + 1] == '>':
            _tokens.append(imp)
            i += 1
        elif operator.match(tokens[i]) or brackets.match(tokens[i]):
            _tokens.append(tokens[i])
        elif symbol.match(tokens[i]):
            current_token = []
            while symbol.match(tokens[i]):
                current_token.append(tokens[i])
                i += 1
            _tokens.append("".join(current_token))
            continue
        elif tokens[i] in [' ', '\n']:
            # If we are in dpll mode, then ' ' is an operator and we catch it earlier so we don't go into this branch
            i += 1
            continue
        else:
            print(f"ERROR: Token Error {tokens[i]} is not a valid token")
            if mode == 'dpll':
                print("ERROR: input might not be in CNF. Use 'solver' mode to get a solution for BNF")
            sys.exit(1)
        i += 1
    return _tokens


def is_balanced(exp):
    match = {'(': ')', '[': ']', '{': '}'}
    s = []
    for c in exp:
        if c in ['(', '{', '[']:
            s.append(c)
        elif c in [')', '}', ']']:
            if len(s) == 0 or match[s.pop()] != c:
                return False
    if len(s) == 0:
        return True
    return False


def validate_expression(exp):
    # this function validates the input infix expression
    symbol = "([a-zA-Z0-9_]+)"
    pre = f"({neg}|\(| )*"
    post = "[)]*"
    op = f"({imp}|{con}|\\{dis}|{bic})"
    spaces = "[ ]*"
    valid_expression = re.compile(f"^{pre}{spaces}{symbol}{spaces}{post}{spaces}({op}{spaces}{pre}{spaces}{symbol}{spaces}{post})*{spaces}$")
    return valid_expression.match(exp)


# infix to prefix converter
def infix_to_prefix(tokens):
    prefix_expression = []
    operator_stack = []
    if not is_balanced(tokens):
        print("ERROR: Parentheses are not balanced")
        sys.exit(1)
    if not validate_expression("".join(tokens)):
        print("ERROR: Not a valid expression")
        sys.exit(1)
    _tokens = tokens[::-1]  # Reverse the string of tokens
    for tok in _tokens:
        if tok == '\n' or (
                tok == ' ' and dis != ' '):  # This check is just in case we are in dpll mode and ' ' means or
            continue
        elif symbol.match(tok):
            prefix_expression.append(tok)
        elif operator.match(tok) and not operator_stack:
            operator_stack.append(tok)
        elif operator_stack:
            if right_brackets.match(tok):
                operator_stack.append(tok)
            elif left_brackets.match(tok):
                while operator_stack and not right_brackets.match(operator_stack[-1]):
                    prefix_expression.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()
            elif right_brackets.match(operator_stack[-1]) or precedence[tok] >= precedence[operator_stack[-1]]:
                operator_stack.append(tok)
            else:
                while operator_stack and not right_brackets.match(operator_stack[-1]) and precedence[tok] < precedence[operator_stack[-1]]:
                    prefix_expression.append(operator_stack.pop())
                operator_stack.append(tok)
    while operator_stack:
        prefix_expression.append(operator_stack.pop())
    return prefix_expression[::-1]


def remove_biconditionals(current_sub_tree, parent):
    if current_sub_tree.is_leaf():
        return
    elif re.match(bic, current_sub_tree.symbol):
        current_node = current_sub_tree
        remove_biconditionals(current_node.children[0], current_node)
        remove_biconditionals(current_node.children[1], current_node)
        c1 = current_node.children[0]
        c2 = current_node.children[1]
        parent.children[parent.children.index(current_node)] = Conjunction(Implication(c1, c2), Implication(c2, c1))
    else:
        current_sub_tree.apply_to_children(remove_biconditionals, current_sub_tree)


def remove_implication(current_sub_tree, parent):
    if current_sub_tree.is_leaf():
        return
    elif imp == current_sub_tree.symbol:
        current_node = current_sub_tree
        remove_implication(current_node.children[0], current_node)
        remove_implication(current_node.children[1], current_node)
        c1 = current_node.children[0]
        c2 = current_node.children[1]
        parent.children[parent.children.index(current_node)] = Disjunction(Negation(c1), c2)
    else:
        current_sub_tree.apply_to_children(remove_implication, current_sub_tree)


def move_negations(current_sub_tree, parent):
    if current_sub_tree.is_leaf():
        return
    elif neg == current_sub_tree.symbol:
        if current_sub_tree.children.is_leaf():  # The negations is right next to a literal
            return
        # Test for double negation
        if neg == current_sub_tree.children.symbol:
            if type(parent) is not Negation:
                parent.children[parent.children.index(current_sub_tree)] = current_sub_tree.children.children
            else:
                parent.children = current_sub_tree.children.cihldren
            current_sub_tree = current_sub_tree.children.children
            if current_sub_tree.is_leaf():  # The negations is right next to a literal
                return
        else:
            if con == current_sub_tree.children.symbol:
                new_current_sub_tree = Disjunction(
                    Negation(current_sub_tree.children.children[0]),
                    Negation(current_sub_tree.children.children[1])
                )
                parent.children[parent.children.index(current_sub_tree)] = new_current_sub_tree
                current_sub_tree = new_current_sub_tree
            elif dis == current_sub_tree.children.symbol:
                new_current_sub_tree = Conjunction(
                    Negation(current_sub_tree.children.children[0]),
                    Negation(current_sub_tree.children.children[1])
                )
                parent.children[parent.children.index(current_sub_tree)] = new_current_sub_tree
                current_sub_tree = new_current_sub_tree
        current_sub_tree.apply_to_children(move_negations, current_sub_tree)
    else:
        current_sub_tree.apply_to_children(move_negations, current_sub_tree)


def distribute_disjunctions(current_sub_tree, parent):
    if current_sub_tree.is_leaf():
        return
    elif dis == current_sub_tree.symbol:
        current_sub_tree.apply_to_children(distribute_disjunctions, current_sub_tree)
        i = 0
        distribute = False
        for c in current_sub_tree.children:
            if type(c) is not str and re.match(con, c.symbol):
                distribute = True
                break
            i += 1
        if distribute:
            a = current_sub_tree.children[1 - i]
            b = current_sub_tree.children[i].children[0]
            c = current_sub_tree.children[i].children[1]
            _current_sub_tree = Conjunction(Disjunction(a, b), Disjunction(a, c))
            parent.children[parent.children.index(current_sub_tree)] = _current_sub_tree
            current_sub_tree = _current_sub_tree
    else:
        current_sub_tree.apply_to_children(distribute_disjunctions, current_sub_tree)


def bnf_to_cnf(bnf):
    root = TreeNode([bnf])
    print_verbose("Full Sentence Before CNF transformation")
    inorder_traversal(bnf)
    print_verbose()
    # Step 1 remove biconditionals
    remove_biconditionals(root.children[0], root)
    print_verbose("After Removing Biconditionals:")
    inorder_traversal(bnf)
    print_verbose()
    # Step 2 remove implications
    remove_implication(root.children[0], root)
    print_verbose("After removing Implications:")
    inorder_traversal(bnf)
    print_verbose()
    # Step 3 move negations in
    move_negations(root.children[0], root)
    print_verbose("After Moving Negations inwards:")
    inorder_traversal(bnf)
    print_verbose()
    # Step 4 distribute disjunctions
    while not is_cnf(root.children[0]):
        distribute_disjunctions(root.children[0], root)
    print_verbose("After Distributing the Disjunctions until we have a CNF expression:")
    inorder_traversal(bnf)
    print_verbose()
    return root.children[0]


def get_clauses(expression):
    # There is probably a better way of getting the clauses from the expression tree
    # but this is the best I could come up with in the amount of time we had.
    if not is_cnf(expression):
        print("ERROR: The expression given to the dpll solver is not in CNF")
        return None

    clauses = []

    def rec_clause_helper(node):
        # Helper function to find all the nodes where the top level node is a conjunction
        if node.is_leaf():
            return
        elif operator.match(node.symbol):
            if node.symbol == con:
                for child in node.children:
                    if child.is_leaf() or child.symbol == dis:
                        clauses.append(child)
            node.apply_to_children(rec_clause_helper)

    def literal_finder(node):
        # Helper function to find the literals in the tree
        if node.is_leaf():
            return [node.symbol]
        elif node.symbol == neg and node.children.is_leaf():
            return [f"{node.symbol}{node.children.symbol}"]
        else:
            return literal_finder(node.children[0]) + literal_finder(node.children[1])

    rec_clause_helper(expression)
    result = []
    for clause in clauses:
        result.append(set(literal_finder(clause)))
    return result


def find_easy_case(clauses, symbols):
    # Find a pure literal
    literals = set([s for clause in clauses for s in clause])
    for k in symbols.keys():
        if k in literals and f"{neg}{k}" not in literals:
            print_debug(f"Easy case: {k} is a pure literal")
            return k
        if k not in literals and f"{neg}{k}" in literals:
            print_debug(f"Easy case: {neg}{k} is a pure literal")
            return f"{neg}{k}"
    # Find a singleton
    for c in clauses:
        if len(c) == 1:
            print_debug(f"Easy case: {list(c)[0]} is a singleton")
            return list(c)[0]
    return None


def remove_and_simplify(clauses, symbol_to_rm, truth_value=True):
    to_remove = []
    not_symbol = f"{neg}{symbol_to_rm}"
    if not truth_value:
        symbol_to_rm = f"{neg}{symbol_to_rm}"
        not_symbol = symbol_to_rm.replace(f"{neg}", '')
    for i in range(0, len(clauses)):
        if symbol_to_rm in clauses[i]:
            # remove the clause from the set of clauses
            to_remove.append(i)
        if not_symbol in clauses[i]:
            # simplify
            print_debug(f"\tsimplify {clauses[i]} to get {clauses[i].difference({not_symbol})}")
            clauses[i] = clauses[i].difference({not_symbol})
    for l, idx in enumerate(to_remove):
        print_debug(f"\tThe clause {clauses[idx - l]} is satisfied")
        del clauses[idx - l]
    print_debug(f"clauses: {clauses}")
    return clauses


def print_clauses(clauses):
    for clause in clauses:
        c = []
        for symbol in clause:
            c.append(symbol)
        print_verbose(" ".join(c))


# Implementation of DPLL solver
def dpll_solver(cnf_expression):
    clauses = get_clauses(cnf_expression)
    if not clauses:
        return False  # if clauses is None the input is not in CNF

    def get_symbols(clause_list):
        symbols = set()
        for sym in [symbol for clause in clause_list for symbol in clause]:
            symbols.add(sym.replace(f"{neg}", ''))
        return symbols

    def dpll_helper(clause_list, symbol_dict):
        while True:
            if len(clause_list) == 0:
                return True, symbol_dict
            print_clauses(clause_list)
            for c in clause_list:
                if len(c) == 0:
                    return False, OrderedDict()
            easy_case = find_easy_case(clause_list, symbol_dict)
            while easy_case:
                if neg not in easy_case:
                    print_verbose(f"easyCase: {easy_case} = True")
                    symbol_dict[easy_case] = True
                else:
                    print_verbose(f"easyCase: {easy_case.replace(neg, '')} = False")
                    symbol_dict[easy_case.replace(neg, '')] = False
                # find all the clauses in which easy_case or neg easy_case appears
                clause_list = remove_and_simplify(clause_list, easy_case.replace(neg, ''),
                                                  symbol_dict[easy_case.replace(neg, '')])
                print_debug(f"The truth values are: {symbol_dict}")
                x = len(clause_list)
                if len(clause_list) == 0:
                    return True, symbol_dict
                print_clauses(clause_list)
                for c in clause_list:
                    if len(c) == 0:
                        print_verbose(f"{easy_case} contradiction")
                        print_verbose("fail|", end='')
                        return False, OrderedDict()
                easy_case = find_easy_case(clause_list, symbol_dict)
            # Try a hard case
            for k, v in symbol_dict.items():
                if v is None:
                    for guess in [True, False]:
                        print_debug(f"Hard Case: guess {k} = {guess}")
                        print_verbose(f"hard case, guess: {k} = {guess}")
                        symbol_dict[k] = guess
                        print_debug(f"The truth values are: {symbol_dict}")
                        answer, solution = dpll_helper(remove_and_simplify(clause_list.copy(), k, guess),
                                                       symbol_dict.copy())
                        if answer:
                            return True, solution
                    if not answer:
                        # We need to backtrack if we made a bad guess
                        print_debug(f"Backtracking!")
                        return False, OrderedDict()

    # add the symbols in lexicographic order to a dictionary
    sym_dict = OrderedDict()
    for s in sorted(get_symbols(clauses)):
        sym_dict[s] = None
    answer, truth_values = dpll_helper(clauses, sym_dict)
    if answer:
        for k, v in truth_values.items():
            if truth_values[k] is None:
                truth_values[k] = False
                print_verbose(f"unbound {k} = {truth_values[k]}")
        for k, v in truth_values.items():
            print(f"{k} = {truth_values[k]}")
        return True
    print("NO VALID ASSIGNMENT")
    return False


def read_file(file_name):
    tokens = []
    with open(file_name) as in_file:
        for line in in_file.readlines():
            if line == '\n':
                continue
            l = line
            if mode == 'dpll' and re.search(r"=>|<=>|\||&", l):
                print("ERROR: input file is not in CNF. Try using solver mode")
                sys.exit(1)
            if mode == 'dpll' and "|" in line:
                l = line.replace(" ", '').replace("|",'')
            tokens.append(f"({l.strip()})")
    return tokenize("&".join(tokens))


if __name__ == '__main__':
    # solver [-v] -mode $mode $input-file
    parser = argparse.ArgumentParser(description='DPLL Solver: A program that transforms BNF sentences into CNF and '
                                                 'solves BNF and CNF sentences using the DPLL algorithm')
    parser.add_argument('-mode', nargs=1, type=str, required=True,
                        help="Mode of the solver. It can be 'cnf', 'dpll', or 'solver'. When the mode is set to "
                             "dpll, the program will expect an input file with sentences in CNF and in the the form:"
                             " 'A B !C' where space means disjunction and there are no other operators")
    parser.add_argument('-v', action='store_true', required=False, help='Optional flag for verbose mode')
    parser.add_argument('-d', action='store_true', required=False, help='Optional flag for debug mode')
    parser.add_argument('filename', help='Path to file with CNF or BNF sentences depending on the mode')
    args = parser.parse_args(sys.argv[1:])
    verbose = args.v
    if len(args.mode) > 1:
        sys.exit(1)
    mode = args.mode[0]
    if mode == 'dpll':
        dis = ' '
        operator = re.compile(f"{imp}|{neg}|{con}|{dis}|{bic}")
        precedence = {bic: 1, imp: 2, dis: 3, con: 4, neg: 5}
    if mode not in ['dpll', 'cnf', 'solver']:
        print("ERROR: Mode not recognized")
        sys.exit(2)
    filename = args.filename
    debug = args.d
    expression = read_file(filename)
    root = make_parse_tree(expression)
    if mode in ['cnf', 'solver']:
        root = bnf_to_cnf(root)
        if mode == 'cnf':
            print("CNF clauses: ")
            print_cnf_clauses(root)
            print()
    if mode in ['solver', 'dpll']:
        if not is_cnf(root):
            print("ERROR: input file is not in CNF")
            sys.exit(1)
        print("Solution:")
        dpll_solver(root)
