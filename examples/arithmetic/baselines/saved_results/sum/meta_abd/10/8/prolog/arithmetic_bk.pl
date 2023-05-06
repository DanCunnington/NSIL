:- use_module(library(clpfd)). %% use constraint logic program to abduce output
:- use_module('meta_abd_clp').
:- ['abduce_clp_score.pl'].

metagol:min_clauses(1).
metagol:max_clauses(2).

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Abductive BK
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
tail([_|T],T).
head([H|_],H).
empty([]).

body_pred(head/2).
body_pred(tail/2).
body_pred(add/2).
body_pred(mult/2).
body_pred(eq/2).
body_pred(empty/1).

:- dynamic
    eq/2,
    add/2,
    mult/2.

abducible(add/2).
abducible(mult/2).
abducible(eq/2).

%% metarules
metarule([P,Q], [P,A,B], [[Q,A,B]]).
metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).
metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]) :- freeze(Q,freeze(R,Q\=R)).
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,A,B]]) :- freeze(Q,freeze(R,Q\=R)).
metarule([P,Q,B], [P,A], [[Q,A,B]]).
metarule([P,Q], [P,A], [[Q,A]]).
metarule([P,Q,F], [P,A,B], [[Q,A,B,F]]).
metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Abduction
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
abduce(Atom, Abd, Abd, Score, Score) :-
    ground(Abd),
    member(Atom, Abd).
abduce(Atom, Abd, [Abduced|Abd], Score1, Score2) :-
    abduce_atom(Atom, Abduced, Score),
    Score2 is Score1 * Score.
abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [add, X, Y],
    abduce_add(X, Y, Abduced, Score).
abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [mult, X, Y],
    abduce_mult(X, Y, Abduced, Score).
abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [eq, X, Y],
    abduce_eq(X, Y, Abduced, Score).

%% Abduce CLP constraints
abduce_add([X,Y|T], [Z|T], Constraint, Score) :-
    (   \+ground(Z) ->
        metagol:new_var(Z)
    ;   \+is_list(Z)),
    atomics_to_string([X,'+',Y,'#=',Z], Constraint),
    Score = 0.1.

abduce_mult([X,Y|T], [Z|T], Constraint, Score) :-
    (   \+ground(Z) ->
        metagol:new_var(Z)
    ;   \+is_list(Z)),
    atomics_to_string([X,'*',Y,'#=',Z], Constraint),
    Score = 0.1.

abduce_eq([X], Z, Constraint, Score) :-
    (   \+ground(Z) ->
        metagol:new_var(Z)
    ;   \+is_list(Z)),
    atomics_to_string([X,'#=',Z], Constraint),
    Score = 0.1.
