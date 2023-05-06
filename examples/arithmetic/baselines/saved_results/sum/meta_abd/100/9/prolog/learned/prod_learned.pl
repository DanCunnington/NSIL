:- use_module(library(clpfd)).

tail([_|T],T).
head([H|_],H).
add([X,Y|T], [Z|T]) :-
    Z #= X + Y.
mult([X,Y|T], [Z|T]) :-
    Z #= X * Y.
eq([X], Y) :-
    X #= Y.

%% learned sum/2
f(A,B):-mult(A,C),f(C,B).
f(A,B):-eq(A,B).
