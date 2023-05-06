#pos(ex_0@10, { result(16) }, { }, {
        :- result(X), X != 16.
        start_list((9, (7, (0, empty)))).
}).
#pos(ex_1@10, { result(7) }, { }, {
        :- result(X), X != 7.
        start_list((3, (4, (0, empty)))).
}).
#pos(ex_2@10, { result(21) }, { }, {
        :- result(X), X != 21.
        start_list((5, (9, (7, empty)))).
}).
#pos(ex_3@10, { result(19) }, { }, {
        :- result(X), X != 19.
        start_list((8, (2, (3, (5, (1, empty)))))).
}).
#pos(ex_4@10, { result(26) }, { }, {
        :- result(X), X != 26.
        start_list((1, (4, (8, (5, (8, empty)))))).
}).
#pos(ex_5@10, { result(29) }, { }, {
        :- result(X), X != 29.
        start_list((7, (6, (3, (5, (8, empty)))))).
}).



% List definition in ASP
list(L) :- start_list(L).
list(T) :- list((_, T)).
head(L, H) :- list(L), L = (H, _).
tail(L, T) :- list(L), L = (_, T).
empty(empty).

% Arithmetic Knowledge - add, mult, eq
add(L, (X+Y, T)) :- list(L), L = (X, (Y, T)).
list(L) :- add(_, L).
mult(L, (X*Y, T)) :- list(L), L = (X, (Y, T)).
list(L) :- mult(_, L).
eq(L, ELT) :- list(L), L = (ELT, empty).

% Link learned f to result, ensure only one result
result(R) :- start_list(L), f(L, R).
:- result(X), result(Y), X < Y.


#predicate(base, head/2).
#predicate(base, tail/2).
#predicate(base, add/2).
#predicate(base, mult/2).
#predicate(base, eq/2).
#predicate(base, empty/1).
#predicate(target, f/2).

% Meta rules
P(A, B) :- Q(A, B), m1(P, Q).
P(A, B) :- Q(A, C), P(C, B), m2(P, Q).
P(A, B) :- Q(A, C), R(C, B), m3(P, Q, R), Q != R.
P(A, B) :- Q(A, B), R(A, B), m4(P, Q, R), Q != R.
P(A) :- Q(A, B), m5(P, Q, B).
P(A) :- Q(A), m6(P, Q).
P(A, B) :- Q(A), R(A, B), m7(P, Q, R).
P(A, B) :- Q(A, B), R(B), m8(P, Q, R).

#modem(2, m1(target/2, any/2)).
#modem(2, m2(target/2, any/2)).
#modem(3, m3(target/2, any/2, any/2)).
#modem(3, m4(target/2, any/2, any/2)).
#modem(2, m5(target/1, any/2)).
#modem(2, m6(target/1, any/1)).
#modem(3, m7(target/2, any/1, any/2)).
#modem(3, m8(target/2, any/2, any/1)).
