/*
=============================================================================
PROJET M2 - PARTIE 2 : MODÈLE MULTI-AGENT SEIRS (C OPTIMISÉ)
=============================================================================
Auteur : Projet Synthèse M2 HPC & Épidémiologie
Date : Janvier 2026
Langage : C99

OPTIMISATIONS :
- Grille d'occupation des infectés (comptage voisins en O(1) au lieu de O(N))
- Réduction des allocations dynamiques
- Calcul d'index linéaire pour accès rapide grille
- Flags de compilation optimaux

COMPILATION :
gcc -std=c99 -O3 -march=native -flto -DNDEBUG -Wall -o sma_c SMA_C_SEIRS_OPTIMIZED.c -lm

EXÉCUTION :
./sma_c
=============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* =============================================================================
 * PARAMÈTRES GLOBAUX DE SIMULATION
 * ========================================================================== */

#define N_AGENTS 20000        // Nombre total d'agents
#define N_INITIAL_INFECTED 20 // Nombre initial d'infectés
#define GRID_SIZE 300         // Taille de la grille (300×300)
#define N_ITERATIONS 730      // Nombre de jours de simulation
#define N_REPLICATIONS 30     // Nombre de réplications

// Paramètres épidémiologiques
#define MEAN_EXPOSED_DURATION 3.0     // Durée moyenne exposition (jours)
#define MEAN_INFECTED_DURATION 7.0    // Durée moyenne infectieux (jours)
#define MEAN_RECOVERED_DURATION 365.0 // Durée moyenne immunité (jours)
#define INFECTION_FORCE 0.5           // Force de l'infection (β)

// Dossier de sortie
#define OUTPUT_DIR "results_c"

/* =============================================================================
 * GÉNÉRATEUR DE NOMBRES ALÉATOIRES - MERSENNE TWISTER MT19937
 * ========================================================================== */

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

typedef struct
{
    unsigned long mt[MT_N];
    int mti;
} MT19937_State;

static MT19937_State mt_state;

// Initialisation du PRNG
void init_genrand(unsigned long seed)
{
    mt_state.mt[0] = seed & 0xffffffffUL;
    for (mt_state.mti = 1; mt_state.mti < MT_N; mt_state.mti++)
    {
        mt_state.mt[mt_state.mti] =
            (1812433253UL * (mt_state.mt[mt_state.mti - 1] ^
                             (mt_state.mt[mt_state.mti - 1] >> 30)) +
             mt_state.mti);
        mt_state.mt[mt_state.mti] &= 0xffffffffUL;
    }
}

// Génère un nombre aléatoire entier sur [0,0xffffffff]
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2] = {0x0UL, MT_MATRIX_A};

    if (mt_state.mti >= MT_N)
    {
        int kk;

        for (kk = 0; kk < MT_N - MT_M; kk++)
        {
            y = (mt_state.mt[kk] & MT_UPPER_MASK) | (mt_state.mt[kk + 1] & MT_LOWER_MASK);
            mt_state.mt[kk] = mt_state.mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++)
        {
            y = (mt_state.mt[kk] & MT_UPPER_MASK) | (mt_state.mt[kk + 1] & MT_LOWER_MASK);
            mt_state.mt[kk] = mt_state.mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt_state.mt[MT_N - 1] & MT_UPPER_MASK) | (mt_state.mt[0] & MT_LOWER_MASK);
        mt_state.mt[MT_N - 1] = mt_state.mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mt_state.mti = 0;
    }

    y = mt_state.mt[mt_state.mti++];

    // Tempering
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

// Génère un nombre aléatoire réel sur [0,1)
double genrand_real2(void)
{
    return genrand_int32() * (1.0 / 4294967296.0);
}

// Distribution exponentielle négative
double negExp(double inMean)
{
    double u = genrand_real2();
    if (u == 0.0)
        u = 1e-10; // Éviter log(0)
    return -inMean * log(1.0 - u);
}

// Entier aléatoire dans [low, high]
int randint(int low, int high)
{
    return low + (genrand_int32() % (high - low + 1));
}

/* =============================================================================
 * STRUCTURE AGENT
 * ========================================================================== */

typedef enum
{
    STATUS_S, // Susceptible
    STATUS_E, // Exposé
    STATUS_I, // Infecté
    STATUS_R  // Remis
} AgentStatus;

typedef struct
{
    int id;
    AgentStatus status;
    int x;
    int y;
    int cell_index; // Index linéaire dans la grille : y * GRID_SIZE + x
    int time_in_status;
    double dE; // Durée période exposition
    double dI; // Durée période infectieuse
    double dR; // Durée immunité
} Agent;

/* =============================================================================
 * STRUCTURE SIMULATION
 * ========================================================================== */

typedef struct
{
    Agent *agents;
    int n_agents;
    int grid_size;
    unsigned long seed;

    // OPTIMISATION : Grille d'occupation des infectés
    // infected_grid[cell_index] = nombre d'infectés dans cette cellule
    int *infected_grid; // Taille : grid_size * grid_size

    // Historique des statistiques
    int *history_S;
    int *history_E;
    int *history_I;
    int *history_R;
} Simulation;

/* =============================================================================
 * FONCTIONS DE SIMULATION
 * ========================================================================== */

/**
 * Initialisation des agents
 */
void init_agents(Simulation *sim)
{
    printf("  Initialisation des %d agents (seed=%lu)...\n", N_AGENTS, sim->seed);

    for (int i = 0; i < N_AGENTS; i++)
    {
        Agent *agent = &sim->agents[i];

        agent->id = i;
        agent->status = (i < N_INITIAL_INFECTED) ? STATUS_I : STATUS_S;

        // Position aléatoire
        agent->x = randint(0, sim->grid_size - 1);
        agent->y = randint(0, sim->grid_size - 1);
        agent->cell_index = agent->y * sim->grid_size + agent->x;
        agent->time_in_status = 0;

        // Tirer les durées exponentiellement
        agent->dE = negExp(MEAN_EXPOSED_DURATION);
        agent->dI = negExp(MEAN_INFECTED_DURATION);
        agent->dR = negExp(MEAN_RECOVERED_DURATION);

        // Si infecté initialement, incrémenter la grille
        if (agent->status == STATUS_I)
        {
            sim->infected_grid[agent->cell_index]++;
        }
    }

    printf("    ✓ %d infectés initiaux\n", N_INITIAL_INFECTED);
    printf("    ✓ %d susceptibles initiaux\n", N_AGENTS - N_INITIAL_INFECTED);
}

/**
 * OPTIMISATION : Compte les infectés dans le voisinage de Moore en O(1)
 * au lieu de O(N) grâce à la grille d'occupation
 */
int count_infected_in_neighborhood(Simulation *sim, int x, int y)
{
    int count = 0;

    // Voisinage de Moore (3×3 incluant la cellule centrale)
    for (int dx = -1; dx <= 1; dx++)
    {
        // Grille toroïdale (wrap-around)
        int nx = (x + dx + sim->grid_size) % sim->grid_size;

        for (int dy = -1; dy <= 1; dy++)
        {
            int ny = (y + dy + sim->grid_size) % sim->grid_size;
            int idx = ny * sim->grid_size + nx;

            // Accès direct O(1) au nombre d'infectés dans cette cellule
            count += sim->infected_grid[idx];
        }
    }

    return count;
}

/**
 * Tentative d'infection pour un agent susceptible
 * Probabilité : p = 1 - exp(-0.5 * Ni)
 */
void try_infection(Simulation *sim, Agent *agent)
{
    if (agent->status != STATUS_S)
        return;

    int Ni = count_infected_in_neighborhood(sim, agent->x, agent->y);

    if (Ni > 0)
    {
        double p = 1.0 - exp(-INFECTION_FORCE * Ni);
        if (genrand_real2() < p)
        {
            agent->status = STATUS_E;
            agent->time_in_status = 0;
        }
    }
}

/**
 * Mise à jour de l'état d'un agent (transitions temporelles)
 */
void update_agent_status(Agent *agent)
{
    if (agent->status == STATUS_E && agent->time_in_status >= (int)agent->dE)
    {
        agent->status = STATUS_I;
        agent->time_in_status = 0;
    }
    else if (agent->status == STATUS_I && agent->time_in_status >= (int)agent->dI)
    {
        agent->status = STATUS_R;
        agent->time_in_status = 0;
    }
    else if (agent->status == STATUS_R && agent->time_in_status >= (int)agent->dR)
    {
        agent->status = STATUS_S;
        agent->time_in_status = 0;
    }
}

/**
 * Enregistrer les statistiques de l'itération
 */
void record_statistics(Simulation *sim, int iteration)
{
    int counts[4] = {0, 0, 0, 0}; // S, E, I, R

    for (int i = 0; i < sim->n_agents; i++)
    {
        counts[sim->agents[i].status]++;
    }

    sim->history_S[iteration] = counts[STATUS_S];
    sim->history_E[iteration] = counts[STATUS_E];
    sim->history_I[iteration] = counts[STATUS_I];
    sim->history_R[iteration] = counts[STATUS_R];
}

/**
 * Exécuter une itération de la simulation (1 jour)
 */
void simulation_step(Simulation *sim, int iteration)
{
    // Créer un ordre aléatoire des agents
    int *indices = (int *)malloc(sim->n_agents * sizeof(int));
    for (int i = 0; i < sim->n_agents; i++)
    {
        indices[i] = i;
    }

    // Shuffle Fisher-Yates
    for (int i = sim->n_agents - 1; i > 0; i--)
    {
        int j = randint(0, i);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Traiter chaque agent dans l'ordre aléatoire
    for (int idx = 0; idx < sim->n_agents; idx++)
    {
        Agent *agent = &sim->agents[indices[idx]];

        // OPTIMISATION : Si infecté actuellement, retirer de la grille
        if (agent->status == STATUS_I)
        {
            sim->infected_grid[agent->cell_index]--;
        }

        // 1. Déplacement aléatoire (CORRECTION DU BUG : mettre à jour X ET Y)
        agent->x = randint(0, sim->grid_size - 1);
        agent->y = randint(0, sim->grid_size - 1);
        agent->cell_index = agent->y * sim->grid_size + agent->x;

        // 2. Tentative d'infection
        try_infection(sim, agent);

        // 3. Mise à jour de l'état (transitions)
        update_agent_status(agent);

        // OPTIMISATION : Si nouvel état infecté, ajouter à la grille
        if (agent->status == STATUS_I)
        {
            sim->infected_grid[agent->cell_index]++;
        }

        // 4. Incrémenter le temps dans l'état
        agent->time_in_status++;
    }

    free(indices);

    // Enregistrer les statistiques
    record_statistics(sim, iteration);
}

/**
 * Lancer la simulation complète
 */
void run_simulation(Simulation *sim)
{
    printf("  Simulation en cours (seed=%lu)...\n", sim->seed);

    clock_t start = clock();

    // État initial (iteration 0)
    record_statistics(sim, 0);

    // Boucle de simulation
    for (int iter = 1; iter < N_ITERATIONS; iter++)
    {
        simulation_step(sim, iter);

        // Affichage de progression tous les 100 jours
        if (iter % 100 == 0)
        {
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            printf("    Jour %d/%d (S=%d, E=%d, I=%d, R=%d) [%.1fs]\n",
                   iter, N_ITERATIONS,
                   sim->history_S[iter], sim->history_E[iter],
                   sim->history_I[iter], sim->history_R[iter], elapsed);
        }
    }

    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("  ✓ Simulation terminée en %.2f secondes\n", total_time);
}

/**
 * Exporter les résultats en CSV
 */
void export_csv(Simulation *sim, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Erreur : impossible d'ouvrir %s\n", filename);
        return;
    }

    // En-tête
    fprintf(file, "iteration,S,E,I,R\n");

    // Données
    for (int i = 0; i < N_ITERATIONS; i++)
    {
        fprintf(file, "%d,%d,%d,%d,%d\n",
                i, sim->history_S[i], sim->history_E[i],
                sim->history_I[i], sim->history_R[i]);
    }

    fclose(file);
    printf("  ✓ Données exportées : %s\n", filename);
}

/**
 * Afficher les statistiques du pic
 */
void print_peak_info(Simulation *sim)
{
    int max_I = 0, peak_day = 0, max_E = 0;

    for (int i = 0; i < N_ITERATIONS; i++)
    {
        if (sim->history_I[i] > max_I)
        {
            max_I = sim->history_I[i];
            peak_day = i;
        }
        if (sim->history_E[i] > max_E)
        {
            max_E = sim->history_E[i];
        }
    }

    printf("  Statistiques :\n");
    printf("    - Pic infectés : %d au jour %d\n", max_I, peak_day);
    printf("    - Max exposés  : %d\n", max_E);
}

/* =============================================================================
 * PROGRAMME PRINCIPAL
 * ========================================================================== */

int main(void)
{
    printf("================================================================================\n");
    printf("PROJET M2 - PARTIE 2 : MODÈLE MULTI-AGENT SEIRS (C OPTIMISÉ)\n");
    printf("================================================================================\n\n");

    printf("Configuration :\n");
    printf("  - Nombre d'agents       : %d\n", N_AGENTS);
    printf("  - Taille grille         : %d×%d\n", GRID_SIZE, GRID_SIZE);
    printf("  - Itérations            : %d jours\n", N_ITERATIONS);
    printf("  - Réplications          : %d\n", N_REPLICATIONS);
    printf("  - Infectés initiaux     : %d\n\n", N_INITIAL_INFECTED);

// Créer le dossier de sortie
#ifdef _WIN32
    _mkdir(OUTPUT_DIR);
#else
    mkdir(OUTPUT_DIR, 0777);
#endif
    printf("✓ Dossier de sortie : %s/\n\n", OUTPUT_DIR);

    // Générer les seeds
    unsigned long seeds[N_REPLICATIONS];
    init_genrand(42); // Seed de base pour générer les seeds
    for (int i = 0; i < N_REPLICATIONS; i++)
    {
        seeds[i] = genrand_int32();
    }
    printf("✓ %d seeds générées\n\n", N_REPLICATIONS);

    clock_t total_start = clock();

    // Lancer les réplications
    for (int rep = 0; rep < N_REPLICATIONS; rep++)
    {
        printf("================================================================================\n");
        printf("RÉPLICATION %d/%d (seed=%lu)\n", rep + 1, N_REPLICATIONS, seeds[rep]);
        printf("================================================================================\n");

        // Initialiser le PRNG
        init_genrand(seeds[rep]);

        // Créer la simulation
        Simulation sim;
        sim.n_agents = N_AGENTS;
        sim.grid_size = GRID_SIZE;
        sim.seed = seeds[rep];

        // Allouer la mémoire
        sim.agents = (Agent *)malloc(N_AGENTS * sizeof(Agent));
        sim.infected_grid = (int *)calloc(GRID_SIZE * GRID_SIZE, sizeof(int)); // Initialisé à 0
        sim.history_S = (int *)malloc(N_ITERATIONS * sizeof(int));
        sim.history_E = (int *)malloc(N_ITERATIONS * sizeof(int));
        sim.history_I = (int *)malloc(N_ITERATIONS * sizeof(int));
        sim.history_R = (int *)malloc(N_ITERATIONS * sizeof(int));

        // Vérifier les allocations
        if (!sim.agents || !sim.infected_grid || !sim.history_S ||
            !sim.history_E || !sim.history_I || !sim.history_R)
        {
            fprintf(stderr, "Erreur : allocation mémoire échouée\n");
            return 1;
        }

        // Initialiser les agents
        init_agents(&sim);

        // Lancer la simulation
        run_simulation(&sim);

        // Exporter les résultats
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/results_C_rep_%02d.csv", OUTPUT_DIR, rep);
        export_csv(&sim, filename);

        // Afficher les statistiques
        print_peak_info(&sim);
        printf("\n");

        // Libérer la mémoire
        free(sim.agents);
        free(sim.infected_grid);
        free(sim.history_S);
        free(sim.history_E);
        free(sim.history_I);
        free(sim.history_R);
    }

    double total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("================================================================================\n");
    printf("✓ TOUTES LES RÉPLICATIONS TERMINÉES AVEC SUCCÈS\n");
    printf("================================================================================\n");
    printf("Temps total : %.2f secondes (%.1f minutes)\n", total_time, total_time / 60.0);
    printf("Fichiers générés : %d fichiers CSV dans %s/\n\n", N_REPLICATIONS, OUTPUT_DIR);
    printf("Prochaine étape : Implémenter le SMA en C++\n");

    return 0;
}