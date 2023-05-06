from global_config import CustomILPConfig

default_bk = '''
    ss(1..4).
    hs_index(1..2).
    elt(1..5).
'''

base_md = '''
    #inject("
    1 { example_active(EG) : group(G, EG) } :- group(G, _).
    :~ not example_active(EG), weight(EG, W).[W@1, eg_weight, W, EG]
    :- #count { H : nge_HYP(H) } > 5.
    ").

    #modeha(hs(var(hs_index), var(elt))).
    #modeb(hs(var(hs_index), var(elt)),(positive)).
    #modeb(var(elt) != var(elt)).
    #modeb(ss_element(var(ss), var(elt)),(positive)).
    #modeh(hit(var(ss))).
    #modeb(hit(var(ss))).
    #bias(":- not lb(0), choice_rule.").
    #bias(":- not ub(1), choice_rule.").
    #bias(":- in_head(H1), in_head(H2), H1<H2.").

'''
config_1_md = f'''{base_md}
    #modeb(ss_element(var(ss), 4), (positive)).
    #modeb(ss_element(var(ss), 3), (positive)).
'''

config_2_md = f'''{base_md}
    #modeb(ss_element(var(ss), 3), (positive)).
    #modeb(ss_element(var(ss), 1), (positive)).
'''

config_3_md = f'''{base_md}
    #modeb(ss_element(var(ss), 4), (positive)).
    #modeb(ss_element(1, var(elt)), (positive)).
'''

config_4_md = f'''{base_md}
    #modeb(ss_element(var(ss), 4), (positive)).
    #modeb(ss_element(4, var(elt)), (positive)).
'''

config_5_md = f'''{base_md}
    #modeb(ss_element(4, var(elt)), (positive)).
    #modeb(ss_element(var(ss), 3), (positive)).
'''

config_6_md = f'''{base_md}
    #modeb(ss_element(2, var(elt)), (positive)).
    #modeb(ss_element(var(ss), 2), (positive)).
'''

config_7_md = f'''{base_md}
    #modeb(ss_element(3, var(elt)), (positive)).
    #modeb(ss_element(var(ss), 3), (positive)).
'''

config_8_md = f'''{base_md}
    #modeb(ss_element(3, var(elt)), (positive)).
    #modeb(ss_element(var(ss), 4), (positive)).
'''

config_9_md = f'''{base_md}
    #modeb(ss_element(4, var(elt)), (positive)).
    #modeb(ss_element(2, var(elt)), (positive)).
'''

config_10_md = f'''{base_md}
    #modeb(ss_element(1, var(elt)), (positive)).
    #modeb(ss_element(3, var(elt)), (positive)).
'''

extra_configs = {
    'config_1': CustomILPConfig(default_bk, config_1_md),
    'config_2': CustomILPConfig(default_bk, config_2_md),
    'config_3': CustomILPConfig(default_bk, config_3_md),
    'config_4': CustomILPConfig(default_bk, config_4_md),
    'config_5': CustomILPConfig(default_bk, config_5_md),
    'config_6': CustomILPConfig(default_bk, config_6_md),
    'config_7': CustomILPConfig(default_bk, config_7_md),
    'config_8': CustomILPConfig(default_bk, config_8_md),
    'config_9': CustomILPConfig(default_bk, config_9_md),
    'config_10': CustomILPConfig(default_bk, config_10_md)
}