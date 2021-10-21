# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import re
from collections import OrderedDict

bic = "<=>"
imp = "=>"
neg = "~"
con = "&"
dis = "|"

precedence = {
    bic: 1,
    imp: 2,
    dis: 3,
    con: 4,
    neg: 5
}

# We are going to need a clause class
symbol = re.compile(r"[a-zA-Z]")
operator = re.compile(r"=>|\||&|~|<=>")
left_brackets = re.compile(r"\(|\[")
right_brackets = re.compile(r"\)|\]")
test = "(P & ~ Q => W <=> A | B & C) & (A => B) & (C | B & A)"
test = "(A <=> B <=> C) & (A | ~B)"
test = "(B => E) & (E => ~G) & (~(~B & ~C) <=> A) & (~(~E | ~G) => A) & (A & B => G) & (F => B & D)"
# test = "(P | Q | ~R) & (P | ~Q | R) & (~P | Q) & (~Q | R | W) & (~R | W) & (~W | ~X) & (~W | X)"


# test = "~(~B & ~C) <=> A"


class TreeNode:
    def __init__(self, children=None):
        self.children = children
        if self.children is None:
            self.children = []

    def apply_to_children(self, func, *args, **kwargs):
        for c in self.children:
            func(c, *args, **kwargs)


class Biconditional(TreeNode):
    def __init__(self, left, right):
        super().__init__(children=[left, right])
        self.symbol = bic


class Implication(TreeNode):
    def __init__(self, left, right):
        super().__init__(children=[left, right])
        self.symbol = imp


class Negation(TreeNode):
    def __init__(self, operand):
        super().__init__(children=operand)
        self.symbol = neg

    def apply_to_children(self, func, *args, **kwargs):
        func(self.children, *args, **kwargs)


class Disjunction(TreeNode):
    def __init__(self, left, right):
        super().__init__(children=[left, right])
        self.symbol = dis


class Conjunction(TreeNode):
    def __init__(self, left, right):
        super().__init__(children=[left, right])
        self.symbol = con


def is_cnf(root):
    if type(root) == str:
        return True
    else:
        cnf = True
        for c in root.children:
            if type(c) is not str and root.symbol == dis and c.symbol == con:
                return False
            cnf = cnf and is_cnf(c)
        return cnf


def inorder_traversal(root):
    if type(root) == str:
        print(root, end='')
    elif type(root) == Negation:
        print("(", end='')
        print(root.symbol, end='')
        inorder_traversal(root.children)
        print(") ", end='')
    elif re.match(con, root.symbol):
        print(" (", end='')
        inorder_traversal(root.children[0])
        print(root.symbol, end='')
        inorder_traversal(root.children[1])
        print(") ", end='')
    elif operator.match(root.symbol):
        print(" (", end='')
        inorder_traversal(root.children[0])
        print(root.symbol, end='')
        inorder_traversal(root.children[1])
        print(") ", end='')
    else:
        print("Error")


def get_clauses(root):
    clauses = []

    def rec_clause_helper(node):
        if type(node) == str:
            return
        elif operator.match(node.symbol):
            if node.symbol == con:
                for c in node.children:
                    if type(c) is not str and c.symbol == dis:
                        clauses.append(c)
            node.apply_to_children(rec_clause_helper)

    def literal_finder_in_conjunctions(node):
        if type(node) == str:
            return [node]
        elif node.symbol == neg and type(node.children) == str:
            return [f"{node.symbol}{node.children}"]
        elif node.symbol == dis:
            x = literal_finder_in_conjunctions(node.children[0]) + literal_finder_in_conjunctions(node.children[1])
            return x

    rec_clause_helper(root)
    result = []
    for c in clauses:
        result.append(set(literal_finder_in_conjunctions(c)))
    return result


def remove_biconditionals(current_sub_tree, parent):
    if type(current_sub_tree) == str:
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
    if type(current_sub_tree) == str:
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
    if type(current_sub_tree) == str:
        return
    elif neg == current_sub_tree.symbol:
        if type(current_sub_tree.children) is str:
            # The negations is right next to a literal
            return
        # Test for double negation
        if neg == current_sub_tree.children.symbol:
            if type(parent) is not Negation:
                parent.children[parent.children.index(current_sub_tree)] = current_sub_tree.children.children
            else:
                parent.children = current_sub_tree.children.cihldren
            current_sub_tree = current_sub_tree.children.children
            if type(current_sub_tree) is str:
                # The negations is right next to a literal
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
    if type(current_sub_tree) == str:
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
    print(f"\nBefore CNF conversion:")
    inorder_traversal(root.children[0])
    # Step 1 remove biconditionals
    remove_biconditionals(root.children[0], root)
    print(f"\nAfter removing biconditionals:")
    inorder_traversal(root.children[0])
    # Step 2 remove implications
    remove_implication(root.children[0], root)
    print(f"\nAfter removing implications:")
    inorder_traversal(root.children[0])
    # Step 3 move negations in
    move_negations(root.children[0], root)
    print(f"\nAfter moving negations:")
    inorder_traversal(root.children[0])
    # Step 4 distribute disjunctions
    while not is_cnf(root.children[0]):
        distribute_disjunctions(root.children[0], root)
    print(f"\nAfter distributing:")
    inorder_traversal(root.children[0])
    print(is_cnf(root.children[0]))
    return root.children[0]


def tokenize(tokens):
    tokens = tokens.strip().replace(" ", '')
    _tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '<' and tokens[i + 1] == '=' and tokens[i + 2] == '>':
            _tokens.append(bic)
            i += 2
        elif tokens[i] == '=' and tokens[i + 1] == '>':
            _tokens.append(imp)
            i += 1
        elif symbol.match(tokens[i]) or operator.match(tokens[i]) or left_brackets.match(
                tokens[i]) or right_brackets.match(tokens[i]):
            _tokens.append(tokens[i])
        else:
            print(f"Token Error. {tokens[i]} is not a valid token")
        i += 1
    return _tokens


# infix to prefix converter
def infix_to_prefix(tokens):
    prefix_expression = []
    operator_stack = []
    _tokens = tokenize(tokens)[::-1]  # Reverse the string of tokens
    for tok in _tokens:
        if tok == '\n' or tok == ' ':
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


def make_parse_tree(tokens):
    _tokens = tokens[::-1]

    def parse_prefix_exp():
        if symbol.match(_tokens[-1]):
            return _tokens.pop()
        elif operator.match(_tokens[-1]):
            op = _tokens.pop()
            if op == imp:
                return Implication(parse_prefix_exp(), parse_prefix_exp())
            elif op == bic:
                return Biconditional(parse_prefix_exp(), parse_prefix_exp())
            elif op == con:
                return Conjunction(parse_prefix_exp(), parse_prefix_exp())
            elif op == dis:
                return Disjunction(parse_prefix_exp(), parse_prefix_exp())
            elif op == neg:
                return Negation(parse_prefix_exp())

    return parse_prefix_exp()


def find_easy_case(clauses, symbols):
    # Find a singleton
    for c in clauses:
        if len(c) == 1:
            print(f"Easy case: {list(c)[0]} is a singleton")
            return list(c)[0]
    # Find a pure literal
    literals = set([s for clause in clauses for s in clause])
    for k in symbols.keys():
        if k in literals and f"{neg}{k}" not in literals:
            print(f"Easy case: {k} is a pure literal")
            return k
        if k not in literals and f"{neg}{k}" in literals:
            print(f"Easy case: {neg}{k} is a pure literal")
            return f"{neg}{k}"
    return None


def get_symbols(clauses):
    symbols = set()
    for x in [s for clause in clauses for s in clause]:
        symbols.add(x.replace(f"{neg}", ''))
    return symbols


def remove_and_simplify(clauses, symbol, truth_value=True):
    to_remove = []
    not_symbol = f"{neg}{symbol}"
    if not truth_value:
        symbol = f"{neg}{symbol}"
        not_symbol = symbol.replace(f"{neg}", '')
    for i in range(0, len(clauses)):
        if symbol in clauses[i]:
            # remove the clause from the set of clauses
            to_remove.append(i)
        if not_symbol in clauses[i]:
            # simplify
            print(f"\tsimplify {clauses[i]} to get {clauses[i].difference({not_symbol})}")
            clauses[i] = clauses[i].difference({not_symbol})
    for l, idx in enumerate(to_remove):
        print(f"\tThe clause {clauses[idx-l]} is satisfied")
        del clauses[idx-l]
    print(f"clauses: {clauses}")
    return clauses


# Implementation of DPLL solver
def dpll_solver(clauses, symbol_dict=None):
    # We need to backtrack if we made a bad guess
    if symbol_dict is None:
        symbol_dict = OrderedDict()
        for s in sorted(get_symbols(clauses)):
            symbol_dict[s] = None
    while True:
        if len(clauses) == 0:
            return True
        for c in clauses:
            if len(c) == 0:
                return False
        easy_case = find_easy_case(clauses, symbol_dict)
        while easy_case:
            # print(f"Easy Case {easy_case}")
            if neg not in easy_case:
                not_easy_case = f"{neg}{easy_case}"
                symbol_dict[easy_case] = True
            else:
                not_easy_case = easy_case.replace(f"{neg}", '')
                symbol_dict[not_easy_case] = False
            # find all the clauses in which easy_case or neg easy_case appears
            clauses = remove_and_simplify(clauses, easy_case)
            print(f"The truth values are: {symbol_dict}")
            if len(clauses) == 0:
                return True
            for c in clauses:
                if len(c) == 0:
                    return False
            easy_case = find_easy_case(clauses, symbol_dict)
        # Try a hard case
        for k, v in symbol_dict.items():
            if v is None:
                for guess in [True, False]:
                    print(f"Hard Case: guess {k} = {guess}")
                    symbol_dict[k] = guess
                    print(f"The truth values are: {symbol_dict}")
                    sol = dpll_solver(remove_and_simplify(clauses.copy(), k, guess), symbol_dict.copy())
                    if sol:
                        return True
                if not sol:
                    print(f"Backtracking!")
                    return False

#TODO: Get the information from the command line
#TODO: Read file


if __name__ == '__main__':
    root = make_parse_tree(infix_to_prefix(test))
    # inorder_traversal(root)
    root = bnf_to_cnf(root)
    clauses = get_clauses(root)
    dpll_solver(clauses)
    # print("\n\n")
    # inorder_traversal(root)
