import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variables de entrada
glucosa = ctrl.Antecedent(np.arange(0, 251, 1), 'glucosa')
presion = ctrl.Antecedent(np.arange(0, 251, 1), 'presion')

# Variable de salida
diagnostico = ctrl.Consequent(np.arange(0, 11, 1), 'diagnostico')

# Funciones de pertenencia
glucosa['baja'] = fuzz.trimf(glucosa.universe, [50, 100, 150])
glucosa['normal'] = fuzz.trimf(glucosa.universe, [100, 150, 200])
glucosa['alta'] = fuzz.trimf(glucosa.universe, [150, 200, 250])

presion['baja'] = fuzz.trimf(presion.universe, [50, 100, 150])
presion['normal'] = fuzz.trimf(presion.universe, [100, 150, 200])
presion['alta'] = fuzz.trimf(presion.universe, [150, 200, 250])

diagnostico['normal'] = fuzz.trimf(diagnostico.universe, [0, 0, 5])
diagnostico['prediabetes'] = fuzz.trimf(diagnostico.universe, [0, 5, 10])
diagnostico['diabetes'] = fuzz.trimf(diagnostico.universe, [5, 10, 10])

# Reglas difusas
regla1 = ctrl.Rule(glucosa['alta'] & presion['alta'], diagnostico['diabetes'])
regla2 = ctrl.Rule(glucosa['normal'] & presion['alta'], diagnostico['prediabetes'])
regla3 = ctrl.Rule(glucosa['baja'] & presion['baja'], diagnostico['normal'])

# Sistema de control
sistema = ctrl.ControlSystem([regla1, regla2, regla3])
diagnostico_sistema = ctrl.ControlSystemSimulation(sistema)

# Casos de prueba
casos_prueba = [
    {'glucosa': 80, 'presion': 100},
    {'glucosa': 130, 'presion': 170},
    {'glucosa': 180, 'presion': 220}
]

# Evaluación y resultados
for caso in casos_prueba:
    diagnostico_sistema.input['glucosa'] = caso['glucosa']
    diagnostico_sistema.input['presion'] = caso['presion']
    diagnostico_sistema.compute()
    print("Resultado para Glucosa={}, Presion={}: {}".format(caso['glucosa'], caso['presion'], diagnostico_sistema.output['diagnostico']))

diagnostico.view(sim=diagnostico_sistema)
