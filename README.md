# DPLL Solver

The Davis–Putnam–Logemann–Loveland (DPLL) algorithm is a complete, backtracking-based search algorithm for deciding the satisfiability of a propositional logic formula in conjunctive normal form (CNF). 

The SAT problem is important both from theoretical and practical points of view. In complexity theory it was the first problem proved to be NP-complete, and can appear in a broad variety of applications such as model checking, automated planning and scheduling, and diagnosis in artificial intelligence.

## BNF to CNF Converter

In order to solve formulas that are not in CNF, I also implemented a BNF to CNF converter.

# How To Run
You can run the solver program using:

```
python3 solver.py [-h] -mode MODE [-v] [-d] filename
```

Example:

If we want to find a solution for a CNF sentence in a file called 
input_file.txt we would run

```
python3 solver.py -v -mode cnf input_file.txt
```

For more information on how to use it, run: 
``` 
python3 solver.py -h
```

If the mode is set to dpll, the output will be the set of truth 
values that satisfy the given set of CNF clauses if there exists one. If not, 
the program will return "NO VALID ASSIGNMENT"
If the input file is not in CNF, the program will return an error message and exit. 

If the mode is set to cnf, the program wil transform the set of BNF clauses into 
CNF and print them out to the user. 

If the mode is set to solver, the program will return the set of CNF clauses and 
the set of truth values that satisfy the set of clauses. 

# Implementation Details
The supported operators in this implementation are: `<=>, =>, !, &, and |` 
for biconditional, implication, negation, conjunction and disjunction respectively.

Before parsing the sentences, we read the input file line by line 
and combine them into a single BNF or CNF sentence using conjunction between 
the sentences. Then we validate the expression using regex, making sure that the arity 
of the operators is followed and that the parentheses and brackets are balanced. 

In this program, the input is expected to be an infix expression. In order
to make the parse tree, we convert the infix expression into a prefix
expression. This approach allows us to deal with parenthesis and brackets. 
Once we have a prefix expression, we can go through it and make internal nodes 
for the operators and leaves for the symbols. 

If the mode is set to dpll, the program expects only expressions in CNF with disjunctions 
represented as spaces. If the expression uses "|" to represent disjunctions, the program 
will still work if the notation is consistent.

Once we have a parse tree. We can convert the expression into CNF by following the steps 
that were explained in class, except the rule for removing the clauses with contradictions 
since it is an optional rule. The DPLL solver is implemented in a recursive way. It first 
tries to find an easy case. This is done in the `find_easy_case function`. Here we look for 
a pure literal, and if we don't find one, we look for a singleton or clause with only one 
literal. If we are unable to find an easy case, we look for a hard case. Then we call the 
function recursively on the rest of the clauses and on a copy of the truth values for the
symbols. The function `remove_and_simplify` takes care of removing any clause that has been 
satisfied, and simplifying any clause that needs simplifying. 
