from src.NeuralSymbolicReasoner.NeurASP import NeurASP


def api(method='NeurASP'):
    if method == 'NeurASP':
        return NeurASP
    else:
        return NotImplementedError
