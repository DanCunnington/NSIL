/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Calculate the score of abduced pseudo-labels from probabilistic facts
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
best_labels(Cons, Score, Label) :-
    cons_str(Cons, Cons_S),
    get_labels(Labels),
    clp_range(Range),
    range_str(Labels, Range, Range_S),
    atomics_to_string([Cons_S, ", ", Range_S], Call_S),
    term_string(Call, Call_S, [variable_names(Eqs)]),
    (   call(Call), metagol:reset_newvars ->
        maximise_prob(Eqs, Score, Label)
    ;   fail).

list_to_tuple([C], C).
list_to_tuple([C|Cons], Call) :-
    list_to_tuple(Cons, Call1),
    Call = (Call1, C).

%% Generate clpfd constraint strings
cons_str(Cons, Cons_S) :-
    atomics_to_string(Cons, ',', Cons_S).
range_str(Labels, Range, Range_S) :-
    atomics_to_string(Labels, ',', Labels_S),
    atomics_to_string(['[',Labels_S,'] ins ', Range], Range_S).

%% Eqs is the list of clpfd variables with constraints
maximise_prob(Eqs, Score, Max_Label) :-
    get_labels(Labels),
    once(label_vars(Labels, Eqs, Vars)),
    findall(Vars, label(Vars), All_Values),
    once(max_label_prob(Labels, All_Values, [], -1, Max_Label, Score)).

%% find the variables corresponding to pseudo-labels
label_vars([], _Eqs, []).
label_vars([L|Labels], Eqs, [V|Vars]) :-
    member(L=V, Eqs),
    label_vars(Labels, Eqs, Vars).

max_label_prob(_, [], L, S, L, S).
max_label_prob(Labels, [V|Values], L0, S0, L, S) :-
    label_prob(Labels, V, S1),
    (   S1 > S0 ->
        max_label_prob(Labels, Values, V, S1, L, S)
    ;   max_label_prob(Labels, Values, L0, S0, L, S)).

label_prob([], [], 1.0).
label_prob([L|Labels], [A|Assignmets], Score) :-
    label_prob(Labels, Assignmets, Score1),
    nn(L, A, S),
    Score is Score1 * S.
