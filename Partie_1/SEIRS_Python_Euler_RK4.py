import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Callable
import time

# =============================================================================
# PARAMÈTRES GLOBAUX DU MODÈLE SEIRS
# =============================================================================

# Paramètres épidémiologiques (CONSTANTS)
BETA = 0.5              # Taux d'infection
SIGMA = 1.0 / 3.0       # Taux inverse incubation (≈ 0.3333)
GAMMA = 1.0 / 7.0       # Taux inverse infectieux (≈ 0.1429)
RHO = 1.0 / 365.0       # Taux inverse immunité (≈ 0.00274)

# Conditions initiales
S0 = 0.999              # 99.9% susceptibles
E0 = 0.0                # 0% exposés
I0 = 0.001              # 0.1% infectés
R0 = 0.0                # 0% remis

# Domaine d'intégration
T_INITIAL = 0.0         # Temps initial (jours)
T_FINAL = 730.0         # Temps final (2 ans = 730 jours)

# Vérification : S0 + E0 + I0 + R0 doit être égal à 1.0
assert abs((S0 + E0 + I0 + R0) - 1.0) < 1e-10, "Erreur : conditions initiales invalides"

# =============================================================================
# SYSTÈME D'ÉQUATIONS DIFFÉRENTIELLES SEIRS
# =============================================================================

def seirs_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    Définition du système d'ODE SEIRS.
    
    Paramètres :
    -----------
    t : float
        Temps (non utilisé, système autonome)
    y : np.ndarray, shape (4,)
        Vecteur d'état [S, E, I, R]
    
    Retourne :
    ----------
    dydt : np.ndarray, shape (4,)
        Dérivées [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = y
    
    dS_dt = -BETA * S * I + RHO * R
    dE_dt = BETA * S * I - SIGMA * E
    dI_dt = SIGMA * E - GAMMA * I
    dR_dt = GAMMA * I - RHO * R
    
    return np.array([dS_dt, dE_dt, dI_dt, dR_dt])

# =============================================================================
# MÉTHODE 1 : EULER EXPLICITE (FORWARD EULER)
# =============================================================================

def euler_method(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Méthode d'Euler explicite pour résoudre y' = f(t, y).
    
    Schéma : y(t+dt) = y(t) + dt * f(t, y(t))
    
    Paramètres :
    -----------
    f : Callable
        Fonction définissant le système d'ODE : f(t, y) -> dy/dt
    y0 : np.ndarray
        Conditions initiales
    t_span : Tuple[float, float]
        Intervalle de temps (t_initial, t_final)
    dt : float
        Pas de temps
    
    Retourne :
    ----------
    t_array : np.ndarray
        Vecteur temps
    y_array : np.ndarray, shape (n_steps, n_equations)
        Solution [S, E, I, R] à chaque pas de temps
    """
    t_initial, t_final = t_span
    n_steps = int((t_final - t_initial) / dt) + 1
    t_array = np.linspace(t_initial, t_final, n_steps)
    
    # Initialisation de la solution
    n_eq = len(y0)
    y_array = np.zeros((n_steps, n_eq))
    y_array[0, :] = y0
    
    # Boucle d'intégration
    y_current = y0.copy()
    for i in range(1, n_steps):
        t_current = t_array[i - 1]
        dy_dt = f(t_current, y_current)
        y_current = y_current + dt * dy_dt
        y_array[i, :] = y_current
    
    return t_array, y_array

# =============================================================================
# MÉTHODE 2 : RUNGE-KUTTA D'ORDRE 4 (RK4)
# =============================================================================

def rk4_method(f: Callable, y0: np.ndarray, t_span: Tuple[float, float], 
               dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Méthode de Runge-Kutta d'ordre 4 pour résoudre y' = f(t, y).
    
    Schéma :
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt/2, y + k1/2)
        k3 = dt * f(t + dt/2, y + k2/2)
        k4 = dt * f(t + dt, y + k3)
        y(t+dt) = y(t) + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    Paramètres :
    -----------
    f : Callable
        Fonction définissant le système d'ODE
    y0 : np.ndarray
        Conditions initiales
    t_span : Tuple[float, float]
        Intervalle de temps (t_initial, t_final)
    dt : float
        Pas de temps
    
    Retourne :
    ----------
    t_array : np.ndarray
        Vecteur temps
    y_array : np.ndarray
        Solution à chaque pas de temps
    """
    t_initial, t_final = t_span
    n_steps = int((t_final - t_initial) / dt) + 1
    t_array = np.linspace(t_initial, t_final, n_steps)
    
    # Initialisation
    n_eq = len(y0)
    y_array = np.zeros((n_steps, n_eq))
    y_array[0, :] = y0
    
    # Boucle d'intégration RK4
    y_current = y0.copy()
    for i in range(1, n_steps):
        t_current = t_array[i - 1]
        
        # Calcul des 4 étapes de Runge-Kutta
        k1 = dt * f(t_current, y_current)
        k2 = dt * f(t_current + dt / 2.0, y_current + k1 / 2.0)
        k3 = dt * f(t_current + dt / 2.0, y_current + k2 / 2.0)
        k4 = dt * f(t_current + dt, y_current + k3)
        
        # Mise à jour
        y_current = y_current + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y_array[i, :] = y_current
    
    return t_array, y_array

# =============================================================================
# FONCTIONS D'EXPORT ET ANALYSE
# =============================================================================

def export_to_csv(t_array: np.ndarray, y_array: np.ndarray, filename: str):
    """
    Exporte la solution dans un fichier CSV.
    
    Format : temps, S, E, I, R
    """
    df = pd.DataFrame({
        'temps': t_array,
        'S': y_array[:, 0],
        'E': y_array[:, 1],
        'I': y_array[:, 2],
        'R': y_array[:, 3]
    })
    df.to_csv(filename, index=False)
    print(f"✓ Données exportées : {filename}")

def compute_l2_error(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Calcule l'erreur L2 entre deux solutions.
    
    ||y1 - y2||_2 = sqrt(sum((y1 - y2)^2))
    """
    return np.linalg.norm(y1 - y2)

# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_seirs_dynamics(t: np.ndarray, y: np.ndarray, title: str, filename: str):
    """
    Graphique 1 : Dynamique SEIRS [S(t), E(t), I(t), R(t)].
    """
    plt.figure(figsize=(12, 7))
    plt.plot(t, y[:, 0], 'b-', linewidth=2, label='S (Susceptibles)')
    plt.plot(t, y[:, 1], 'orange', linewidth=2, label='E (Exposés)')
    plt.plot(t, y[:, 2], 'r-', linewidth=2, label='I (Infectés)')
    plt.plot(t, y[:, 3], 'g-', linewidth=2, label='R (Remis)')
    
    plt.xlabel('Temps (jours)', fontsize=14)
    plt.ylabel('Proportion de la population', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, T_FINAL])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure sauvegardée : {filename}")

def plot_comparison_infected(t_euler, y_euler, t_rk4, y_rk4, filename: str):
    """
    Graphique 2 : Comparaison I(t) entre Euler et RK4.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(t_euler, y_euler[:, 2], 'r--', linewidth=2, label='Euler - I(t)', alpha=0.7)
    plt.plot(t_rk4, y_rk4[:, 2], 'b-', linewidth=2, label='RK4 - I(t)')
    
    plt.xlabel('Temps (jours)', fontsize=14)
    plt.ylabel('Proportion d\'infectés I(t)', fontsize=14)
    plt.title('Comparaison Méthodes : Euler vs RK4', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, T_FINAL])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure sauvegardée : {filename}")

def convergence_analysis(dt_values: list, filename: str):
    """
    Graphique 4 : Analyse de convergence (log-log).
    
    Compare l'erreur entre Euler et RK4 en fonction du pas de temps.
    """
    errors_euler = []
    errors_rk4 = []
    
    # Solution de référence (RK4 avec dt très petit)
    t_ref, y_ref = rk4_method(seirs_ode, np.array([S0, E0, I0, R0]), 
                              (T_INITIAL, T_FINAL), dt=0.001)
    
    for dt in dt_values:
        # Euler
        t_euler, y_euler = euler_method(seirs_ode, np.array([S0, E0, I0, R0]), 
                                        (T_INITIAL, T_FINAL), dt)
        # Interpoler pour comparer avec la référence
        y_euler_interp = np.interp(t_ref, t_euler, y_euler[:, 2])
        error_euler = np.sqrt(np.mean((y_euler_interp - y_ref[:, 2])**2))
        errors_euler.append(error_euler)
        
        # RK4
        t_rk4, y_rk4 = rk4_method(seirs_ode, np.array([S0, E0, I0, R0]), 
                                  (T_INITIAL, T_FINAL), dt)
        y_rk4_interp = np.interp(t_ref, t_rk4, y_rk4[:, 2])
        error_rk4 = np.sqrt(np.mean((y_rk4_interp - y_ref[:, 2])**2))
        errors_rk4.append(error_rk4)
    
    # Tracé log-log
    plt.figure(figsize=(10, 7))
    plt.loglog(dt_values, errors_euler, 'ro-', linewidth=2, markersize=8, 
               label='Euler (ordre 1)')
    plt.loglog(dt_values, errors_rk4, 'bs-', linewidth=2, markersize=8, 
               label='RK4 (ordre 4)')
    
    # Lignes de référence théoriques
    plt.loglog(dt_values, np.array(dt_values)**1 * errors_euler[0] / dt_values[0]**1, 
               'r--', alpha=0.5, label='Pente théorique O(dt)')
    plt.loglog(dt_values, np.array(dt_values)**4 * errors_rk4[0] / dt_values[0]**4, 
               'b--', alpha=0.5, label='Pente théorique O(dt⁴)')
    
    plt.xlabel('Pas de temps dt (jours)', fontsize=14)
    plt.ylabel('Erreur RMSE sur I(t)', fontsize=14)
    plt.title('Convergence : Euler vs RK4', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Figure sauvegardée : {filename}")

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    """
    Programme principal - Partie 1 : Résolution numérique SEIRS.
    """
    print("="*80)
    print("PROJET M2 - PARTIE 1 : MODÈLE SEIRS - RÉSOLUTION NUMÉRIQUE (PYTHON)")
    print("="*80)
    print()
    
    # Conditions initiales
    y0 = np.array([S0, E0, I0, R0])
    print(f"Conditions initiales : S={S0}, E={E0}, I={I0}, R={R0}")
    print(f"Vérification : S+E+I+R = {np.sum(y0):.10f} ✓")
    print()
    
    # Paramètres
    dt = 0.1  # Pas de temps (jours)
    print(f"Pas de temps dt = {dt} jours")
    print(f"Domaine : [{T_INITIAL}, {T_FINAL}] jours")
    print()
    
    # -------------------------------------------------------------------------
    # RÉSOLUTION PAR EULER
    # -------------------------------------------------------------------------
    print("--- MÉTHODE EULER EXPLICITE ---")
    start_time = time.time()
    t_euler, y_euler = euler_method(seirs_ode, y0, (T_INITIAL, T_FINAL), dt)
    time_euler = time.time() - start_time
    print(f"✓ Résolution terminée en {time_euler:.4f} secondes")
    print(f"  Nombre de pas de temps : {len(t_euler)}")
    export_to_csv(t_euler, y_euler, 'seirs_python_euler.csv')
    print()
    
    # -------------------------------------------------------------------------
    # RÉSOLUTION PAR RK4
    # -------------------------------------------------------------------------
    print("--- MÉTHODE RUNGE-KUTTA 4 ---")
    start_time = time.time()
    t_rk4, y_rk4 = rk4_method(seirs_ode, y0, (T_INITIAL, T_FINAL), dt)
    time_rk4 = time.time() - start_time
    print(f"✓ Résolution terminée en {time_rk4:.4f} secondes")
    print(f"  Nombre de pas de temps : {len(t_rk4)}")
    export_to_csv(t_rk4, y_rk4, 'seirs_python_rk4.csv')
    print()
    
    # -------------------------------------------------------------------------
    # COMPARAISON EULER vs RK4
    # -------------------------------------------------------------------------
    print("--- COMPARAISON EULER vs RK4 ---")
    l2_error = compute_l2_error(y_euler, y_rk4)
    print(f"Erreur L2 : ||y_euler - y_rk4||_2 = {l2_error:.6e}")
    print()
    
    # -------------------------------------------------------------------------
    # VISUALISATIONS
    # -------------------------------------------------------------------------
    print("--- GÉNÉRATION DES GRAPHIQUES ---")
    plot_seirs_dynamics(t_rk4, y_rk4, 
                       'Dynamique SEIRS - Méthode RK4', 
                       'seirs_dynamics_rk4.png')
    
    plot_comparison_infected(t_euler, y_euler, t_rk4, y_rk4, 
                            'comparison_euler_rk4.png')
    
    # Analyse de convergence
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    print("Analyse de convergence en cours...")
    convergence_analysis(dt_values, 'convergence_analysis.png')
    
    print()
    print("="*80)
    print("✓ PARTIE 1 TERMINÉE AVEC SUCCÈS")
    print("="*80)
    print()
    print("Fichiers générés :")
    print("  - seirs_python_euler.csv")
    print("  - seirs_python_rk4.csv")
    print("  - seirs_dynamics_rk4.png")
    print("  - comparison_euler_rk4.png")
    print("  - convergence_analysis.png")

if __name__ == "__main__":
    main()