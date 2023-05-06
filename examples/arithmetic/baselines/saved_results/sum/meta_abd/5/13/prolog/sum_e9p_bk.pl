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
mod(X,Y) :- Y is X mod 2.
add(X,Y,Z) :- Z is X + Y.
plus_nine(X,Y) :- Y is X + 9.
even(X) :- mod(X,0).
odd(X) :- mod(X,1).


body_pred(head/2).
body_pred(tail/2).
body_pred(add/3).
body_pred(eq/2).
body_pred(plus_nine/2).
body_pred(even/1).
body_pred(odd/1).

:- dynamic
    add/3,
    eq/2,
    head/2,
    tail/2,
    plus_nine/2,
    even/1,
    odd/1.


abducible(add/3).
abducible(eq/2).
abducible(head/2).
abducible(tail/2).
abducible(plus_nine/2).
abducible(even/1).
abducible(odd/1).


%% metarules
% For E9P rules
% f(A,B) :- head(A,C), tail(A,D), eq(D,B), even(C).
% f(A,B) :- head(A,C), tail(A,D), plus_nine(D,B), odd(C).

metarule([P,Q,R,S,T], [P,A,B], [[Q,A,C],[R,A,D],[S,D,B],[T,C]]).

% For addition rule
% f(A,B) :- head(A,C), tail(A,D), add(C,D,B).

metarule([P,Q,R,S], [P,A,B], [[Q,A,C],[R,A,D],[S,C,D,B]]).

% Old ones from recursive task
%metarule([P,Q], [P,A,B], [[Q,A,B]]).
%metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).
%metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]) :- freeze(Q,freeze(R,Q\=R)).
%metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,A,B]]) :- freeze(Q,freeze(R,Q\=R)).
%metarule([P,Q,B], [P,A], [[Q,A,B]]).
%metarule([P,Q], [P,A], [[Q,A]]).
%metarule([P,Q,F], [P,A,B], [[Q,A,B,F]]).
%metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).
%metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).


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
    Atom =.. [head, X, Y],
    abduce_head(X, Y, Abduced, Score).

abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [tail, X, Y],
    abduce_tail(X, Y, Abduced, Score).

abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [plus_nine, X, Y, Z],
    abduce_plus_nine(X, Y, Z, Abduced, Score).

abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [even, X],
    abduce_even(X, Abduced, Score).

abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [odd, X],
    abduce_odd(X, Abduced, Score).

abduce_atom(Atom, Abduced, Score) :-
    Atom =.. [eq, X, Y],
    abduce_eq(X, Y, Abduced, Score).

%% Abduce CLP constraints
abduce_add(X,Y,Z, Constraint, Score) :-
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
