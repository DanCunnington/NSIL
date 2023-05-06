from global_config import CustomILPConfig

# Define background knowledge used in different configurations
standard_bk_knowledge = '''
    digit_type(0..9).
    even(X) :- digit_type(X), X \\ 2 = 0.
    plus_nine(X1,X2) :- digit_type(X1), X2=9+X1.
    
    :- digit(1,X0), digit(2,X1), result(Y1), result(Y2), Y1 != Y2.
    result(Y) :- digit(1,X0), digit(2,X1), solution(X0,X1,Y).
'''

small_range = '''
num(0..18).
'''

large_range = '''
num(-9..81).
'''

sf = '''
#bias("penalty(1, head(X)) :- in_head(X).").
#bias("penalty(1, body(X)) :- in_body(X).").
'''

increased_bk_knowledge = '''
    digit_type(0..9).
    even(X) :- digit_type(X), X \\ 2 = 0.
    plus_nine(X1,X2) :- digit_type(X1), X2=9+X1.
    
    :- digit(1,X0), digit(2,X1), result(Y1), result(Y2), Y1 != Y2.
    result(Y) :- digit(1,X0), digit(2,X1), solution(X0,X1,Y).
    plus_eight(X1,X2) :- digit_type(X1), X2=8+X1.
    plus_seven(X1,X2) :- digit_type(X1), X2=7+X1.
    plus_six(X1,X2) :- digit_type(X1), X2=6+X1.
    plus_five(X1,X2) :- digit_type(X1), X2=5+X1.
    plus_four(X1,X2) :- digit_type(X1), X2=4+X1.
    plus_three(X1,X2) :- digit_type(X1), X2=3+X1.
    plus_two(X1,X2) :- digit_type(X1), X2=2+X1.
    plus_one(X1,X2) :- digit_type(X1), X2=1+X1.
'''

config_1_md = '''
    #modeh(solution(var(digit_type),var(digit_type),var(num))).
    #modeb(var(num) = var(digit_type)).
    #modeb(var(num) = var(digit_type) + var(digit_type)).
    #modeb(plus_nine(var(digit_type),var(num))).
    #modeb(even(var(digit_type))).
    #modeb(not even(var(digit_type))).
    #maxv(3).
'''
config_2_3_md = '''
    #modeh(solution(var(digit_type),var(digit_type),var(num))).
    #modeb(var(num) = var(digit_type)).
    #modeb(var(num) = var(digit_type) + var(digit_type)).
    #modeb(var(num) = var(digit_type) - var(digit_type)).
    #modeb(var(num) = var(digit_type) * var(digit_type)).
    #modeb(var(num) = var(digit_type) / var(digit_type)).
    #modeb(plus_nine(var(digit_type),var(num))).
    #modeb(even(var(digit_type))).
    #modeb(not even(var(digit_type))).
    #maxv(3).
'''

config_4_5_md = '''
    #modeh(solution(var(digit_type),var(digit_type),var(num))).
    #modeb(var(num) = var(digit_type)).
    #modeb(plus_nine(var(digit_type),var(num))).
    #modeb(plus_eight(var(digit_type),var(num))).
    #modeb(plus_seven(var(digit_type),var(num))).
    #modeb(plus_six(var(digit_type),var(num))).
    #modeb(plus_five(var(digit_type),var(num))).
    #modeb(plus_four(var(digit_type),var(num))).
    #modeb(plus_three(var(digit_type),var(num))).
    #modeb(plus_two(var(digit_type),var(num))).
    #modeb(plus_one(var(digit_type),var(num))).
    #modeb(even(var(digit_type))).
    #modeb(not even(var(digit_type))).
    #maxv(3).
'''

config_6_md = '''
    #modeh(solution(var(digit_type),var(digit_type),var(num))).
    #modeb(var(num) = var(digit_type)).
    #modeb(var(num) = var(digit_type) + var(digit_type)).
    #modeb(plus_nine(var(digit_type),var(num))).
    #modeb(plus_eight(var(digit_type),var(num))).
    #modeb(plus_seven(var(digit_type),var(num))).
    #modeb(plus_six(var(digit_type),var(num))).
    #modeb(plus_five(var(digit_type),var(num))).
    #modeb(plus_four(var(digit_type),var(num))).
    #modeb(plus_three(var(digit_type),var(num))).
    #modeb(plus_two(var(digit_type),var(num))).
    #modeb(plus_one(var(digit_type),var(num))).
    #modeb(even(var(digit_type))).
    #modeb(not even(var(digit_type))).
    #maxv(3).
'''

config_7_md = '''
    #modeh(solution(var(digit_type),var(digit_type),var(num))).
    #modeb(var(num) = var(digit_type)).
    #modeb(var(num) = var(digit_type) + var(digit_type)).
    #modeb(var(num) = var(digit_type) - var(digit_type)).
    #modeb(var(num) = var(digit_type) * var(digit_type)).
    #modeb(var(num) = var(digit_type) / var(digit_type)).
    #modeb(plus_nine(var(digit_type),var(num))).
    #modeb(plus_eight(var(digit_type),var(num))).
    #modeb(plus_seven(var(digit_type),var(num))).
    #modeb(plus_six(var(digit_type),var(num))).
    #modeb(plus_five(var(digit_type),var(num))).
    #modeb(plus_four(var(digit_type),var(num))).
    #modeb(plus_three(var(digit_type),var(num))).
    #modeb(plus_two(var(digit_type),var(num))).
    #modeb(plus_one(var(digit_type),var(num))).
    #modeb(even(var(digit_type))).
    #modeb(not even(var(digit_type))).
    #maxv(3).
'''

extra_configs = {
    'config_1': CustomILPConfig(f'{small_range}{standard_bk_knowledge}', f'{config_1_md}{sf}'),
    'config_2': CustomILPConfig(f'{small_range}{standard_bk_knowledge}', f'{config_2_3_md}{sf}'),
    'config_3': CustomILPConfig(f'{large_range}{standard_bk_knowledge}', f'{config_2_3_md}{sf}'),
    'config_4': CustomILPConfig(f'{small_range}{increased_bk_knowledge}', f'{config_4_5_md}{sf}'),
    'config_5': CustomILPConfig(f'{large_range}{increased_bk_knowledge}', f'{config_4_5_md}{sf}'),
    'config_6': CustomILPConfig(f'{small_range}{increased_bk_knowledge}', f'{config_6_md}{sf}'),
    'config_7': CustomILPConfig(f'{small_range}{increased_bk_knowledge}', f'{config_7_md}{sf}')
}