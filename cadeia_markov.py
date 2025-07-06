import numpy as np
from typing import List, Optional

# Definição dos estados do clima
states: List[str] = ["Ensolarado", "Nublado", "Chuvoso"]

# Matriz de transição de estados como numpy array
transition_matrix: np.ndarray = np.array([
    [0.8, 0.15, 0.05],  # Transições a partir de "Ensolarado"
    [0.2, 0.6, 0.2],    # Transições a partir de "Nublado"
    [0.25, 0.25, 0.5]   # Transições a partir de "Chuvoso"
])

def get_state_index(state: str) -> int:
    """Retorna o índice do estado na lista de estados."""
    return states.index(state)

def predict_weather(
    initial_state: str, 
    num_days: int, 
    seed: Optional[int] = None
) -> List[str]:
    """
    Prevê o clima para os próximos dias usando uma cadeia de Markov.
    
    Args:
        initial_state: Estado inicial do clima.
        num_days: Número de dias a prever.
        seed: Semente para reprodutibilidade (opcional).
    Returns:
        Lista com a previsão dos estados do clima.
    """
    if seed is not None:
        np.random.seed(seed)
    current_state = initial_state
    forecast = [current_state]

    for _ in range(num_days - 1):
        current_index = get_state_index(current_state)
        next_state = np.random.choice(
            states, 
            p=transition_matrix[current_index]
        )
        forecast.append(next_state)
        current_state = next_state

    return forecast

if __name__ == "__main__":
    initial_state = "Nublado"
    num_days = 10
    seed = 42  # Para reprodutibilidade, pode remover se não quiser

    forecast = predict_weather(initial_state, num_days, seed=seed)

    print(f"Estado inicial: {initial_state}")
    print("Previsão para os próximos dias:")
    for day, state in enumerate(forecast, start=1):
        print(f"Dia {day:2d}: {state}")