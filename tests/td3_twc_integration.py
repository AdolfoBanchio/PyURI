import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import gymnasium as gym
import torch
import numpy as np
from fiuri import PyUriTwc, build_fiuri_twc
from td3_flat.td3_flat import TD3Engine
from utils import SequenceBuffer
from mlp import TwinCritic

def test_full_system_integration():
    print("--- Iniciando Test de Integración Total (Actor-Buffer-Critic) ---")
    
    # 1. Configuración y Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 32
    state_dim = 2
    action_dim = 1
    
    # 2. Instanciación de Clases
    # Nota: Asegúrate de tener build_fiuri_twc y TWC_JSON en el scope
    actor = build_fiuri_twc() 
    actor.to(device)
    
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    critic.to(device)
    
    buffer = SequenceBuffer(capacity=10000, device=device)
    
    print(f"  Instancias creadas en: {device}")

    # 3. Llenado del Buffer con datos sintéticos
    # Simulamos un episodio de 50 pasos
    for t in range(50):
        obs = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = 0.1
        next_obs = np.random.randn(state_dim).astype(np.float32)
        terminated = False
        truncated = (t == 49) # El paso 50 marca el fin
        buffer.store(obs, action, reward, next_obs, terminated, truncated)
    
    print(f"  Buffer cargado. Total transiciones: {buffer.total_transitions}")

    # 4. Simulación de un paso de entrenamiento (Forward Path)
    try:
        # A. Sampleo de Secuencia
        batch = buffer.sample(batch_size, seq_len)
        s = batch['obs']      # (B, T, 2)
        a = batch['action']   # (B, T, 1)
        
        print(f"  Batch sampleado: S{s.shape}, A{a.shape}")

        # B. Flujo del Actor (BPTT)
        # Simulamos lo que haría el entrenamiento del Actor
        pred_actions, _ = actor.forward_bptt(s)
        print(f"  Actor Forward (BPTT) exitoso: {pred_actions.shape}")

        # C. Flujo del Crítico (Aplanamiento interno)
        # Simulamos la evaluación de Q(s, a)
        q1, q2 = critic(s, pred_actions)
        print(f"  Critic Forward exitoso: Q1{q1.shape}, Q2{q2.shape}")

        # 5. Verificaciones Finales de Shaping
        assert pred_actions.shape == (batch_size, seq_len, action_dim), "Shape de acción incorrecta"
        assert q1.shape == (batch_size, seq_len, 1), "Shape de Q1 incorrecta (debería ser 3D tras el crítico)"
        assert pred_actions.device.type == device.type, "El Actor no está en el dispositivo correcto"
        assert q1.device.type == device.type, "El Crítico no está en el dispositivo correcto"

        print("\n✅ ¡INTEGRACIÓN TOTAL EXITOSA!")
        print("El flujo de información entre el circuito biológico y los críticos MLP es consistente.")

    except Exception as e:
        print(f"\n❌ FALLÓ EL TEST DE INTEGRACIÓN:")
        import traceback
        traceback.print_exc()


def test_full_system_with_gradients():
    print("--- Iniciando Test de Integración y Flujo de Gradientes ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len, state_dim, action_dim = 4, 16, 2, 1
    
    # 1. Inicialización
    actor = build_fiuri_twc()
    actor.to(device)
    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    buffer = SequenceBuffer(capacity=1000, device=device)

    # 2. Llenar buffer con 1 episodio corto
    for _ in range(20):
        buffer.store(np.random.randn(2), np.random.randn(1), 0.1, np.random.randn(2), False, False)
    # Forzar cierre de episodio para que sea sampleable
    buffer.store(np.random.randn(2), np.random.randn(1), 0.1, np.random.randn(2), True, False)

    # 3. Sampleo
    batch = buffer.sample(batch_size, seq_len)
    states = batch['obs'] # (B, T, 2)

    # ---------------------------------------------------------
    # MINI TEST DE GRADIENTES
    # ---------------------------------------------------------
    print("\n  Verificando flujo de gradientes...")
    
    # Aseguramos que los gradientes estén en cero
    actor.zero_grad()
    critic.zero_grad()

    # Paso 1: Actor genera acciones para toda la secuencia (BPTT)
    # IMPORTANTE: Aquí NO usamos .detach() porque queremos que el gradiente fluya
    pred_actions, _ = actor.forward_bptt(states)
    
    # Paso 2: El Crítico evalúa esas acciones predichas
    q1_values = critic.q1_forward(states, pred_actions)
    
    # Paso 3: Definimos una "Loss" (queremos maximizar Q, o minimizar -Q)
    # Usamos mean() para obtener un escalar y poder llamar a backward()
    loss = -q1_values.mean()
    
    # Paso 4: Backpropagation
    loss.backward()

    # Paso 5: Verificación de gradientes en el Actor
    # Tomamos los pesos de las sinapsis químicas como referencia
    actor_grad_norm = actor.weights.grad.norm().item()
    critic_grad_norm = critic.l1_1.weight.grad.norm().item()

    print(f"    Norma del gradiente en Critic (L1): {critic_grad_norm:.6f}")
    print(f"    Norma del gradiente en Actor (Weights): {actor_grad_norm:.6f}")

    # Validaciones
    assert critic_grad_norm > 0, "El Crítico no recibió gradientes."
    assert actor_grad_norm > 0, "¡ERROR! El Actor no recibió gradientes del Crítico. El BPTT podría estar roto."
    
    if actor_grad_norm > 0 and critic_grad_norm > 0:
        print("✅ Gradientes fluyendo: La pérdida del Crítico está informando al Actor biológico.")
    
    print("\n--- TEST DE INTEGRACIÓN Y GRADIENTES COMPLETADO ---")


def test_engine_update_flow():
    print("--- Iniciando Test de Motor TD3 y Flujo de BPTT ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len, burn_in = 4, 32, 8
    
    # 1. Preparar Componentes
    actor = build_fiuri_twc()
    critic = TwinCritic(state_dim=2, action_dim=1)
    
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
    # Supongamos que estas son las dimensiones de MountainCarContinuous
    obs_space = gym.spaces.Box(low=-1.2, high=0.6, shape=(2,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    engine = TD3Engine(
        gamma=0.99, tau=0.005, 
        observation_space=obs_space, action_space=act_space,
        actor=actor, critic=critic,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        policy_delay=1, # Forzamos delay=1 para ver gradientes del actor de inmediato
        device=device
    )

    buffer = SequenceBuffer(capacity=1000, device=device)

    # 2. Llenar buffer con datos para poder samplear seq_len
    # Necesitamos al menos un episodio que sea más largo que seq_len
    for t in range(seq_len + 5):
        buffer.store(
            np.random.randn(2).astype(np.float32), 
            np.random.randn(1).astype(np.float32), 
            1.0, 
            np.random.randn(2).astype(np.float32), 
            False, False
        )
    buffer.store(np.random.randn(2), np.random.randn(1), 0.0, np.random.randn(2), True, False)

    # 3. Capturar pesos antes de la actualización
    # Usamos una copia de los pesos para comparar después del step
    with torch.no_grad():
        initial_weights = engine.actor.weights.clone()

    # 4. Ejecutar el Update Step
    print(f"  Ejecutando update_step_bptt (T={seq_len}, burn_in={burn_in})...")
    batch = buffer.sample(batch_size, seq_len)
    
    # Verificamos que no haya gradientes previos
    engine.actor_optimizer.zero_grad()
    engine.critic_optimizer.zero_grad()

    actor_loss, critic_loss = engine.update_step_bptt(batch, burn_in)

    # 5. Verificaciones
    print(f"    Actor Loss: {actor_loss:.6f}")
    print(f"    Critic Loss: {critic_loss:.6f}")

    # Comprobar si los pesos del actor cambiaron
    with torch.no_grad():
        weight_diff = torch.norm(engine.actor.weights - initial_weights).item()
        print(f"    Norma del cambio en pesos del Actor: {weight_diff:.6e}")

    # Validaciones Finales
    assert critic_loss > 0, "La pérdida del crítico no debería ser cero con datos aleatorios"
    assert weight_diff > 0, "¡ERROR! Los pesos del actor no cambiaron. El gradiente no llegó a los parámetros."
    
    # Comprobar si hay NaNs (común en BPTT si el gradiente explota)
    assert not torch.isnan(engine.actor.weights).any(), "¡Explosión de gradiente detectada! (NaNs en pesos)"

    print("\n✅ ¡TEST EXITOSO!")
    print("El TD3Engine procesó la secuencia, aplicó BPTT y actualizó los pesos biológicos.")


test_full_system_integration()
test_full_system_with_gradients()
test_engine_update_flow()