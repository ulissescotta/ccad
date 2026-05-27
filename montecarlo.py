"""
=============================================================================
INTEGRAÇÃO DE MONTE CARLO PARALELIZADA — ESTIMATIVA DE π
=============================================================================
Disciplina : Métodos Numéricos
Problema   : Estimar π via integração geométrica de Monte Carlo
Técnica    : Paralelismo com multiprocessing.Pool (workers independentes)

Fundamento matemático:
  - Círculo unitário inscrito no quadrado [-1,1] × [-1,1]
  - P(ponto dentro do círculo) = Área_círculo / Área_quadrado
                                = π·r² / (2r)² = π/4
  - Logo: π ≈ 4 · (pontos_dentro / total_pontos)
  - Convergência: erro ~ O(1/√N) → para ε ≈ 10⁻⁵ precisamos N ≈ 10¹⁰

Paralelismo:
  - multiprocessing.Pool divide N entre todos os núcleos lógicos
  - Cada worker usa semente independente (reprodutibilidade)
  - Processamento interno em mini-lotes de 10M (controle de RAM)

Execução:
  python monte_carlo_paralelo.py           # usa configuração automática
  python monte_carlo_paralelo.py --rapido  # versão curta para validação
=============================================================================
"""

import multiprocessing as mp
import numpy as np
import time
import sys
import os
from datetime import datetime

# ─── Configuração adaptativa ──────────────────────────────────────────────────

N_WORKERS   = mp.cpu_count()   # todos os núcleos lógicos disponíveis
SEED_BASE   = 42               # reprodutibilidade
MINI_LOTE   = 10_000_000       # amostras por iteração interna (~160 MB/worker)

# Meta: cada worker deve rodar ~70 s no mínimo.
# Calibrado para uma máquina moderna (~440M pts/s/núcleo via NumPy vetorizado).
# Ajuste TARGET_SEGUNDOS se quiser tempo diferente.
TARGET_SEGUNDOS   = 70                           # tempo-alvo por worker (s)
TAXA_ESTIMADA_PPS = 440_000_000                  # pontos/seg/núcleo (NumPy)
AMOSTRAS_WORKER   = int(TARGET_SEGUNDOS * TAXA_ESTIMADA_PPS)
TOTAL_AMOSTRAS    = AMOSTRAS_WORKER * N_WORKERS

MODO_RAPIDO = "--rapido" in sys.argv             # flag para testes rápidos
if MODO_RAPIDO:
    AMOSTRAS_WORKER = 50_000_000
    TOTAL_AMOSTRAS  = AMOSTRAS_WORKER * N_WORKERS

CHECKPOINT_SECS = 10   # intervalo de progresso (segundos)


# ─── Kernel de Monte Carlo (executado em cada processo) ───────────────────────

def worker_monte_carlo(args: tuple) -> dict:
    """
    Realiza N lançamentos de Monte Carlo para contar quantos pontos
    caem dentro do círculo unitário (x² + y² ≤ 1).

    Parâmetros
    ----------
    args : (worker_id: int, n_amostras: int, seed: int)

    Retorno
    -------
    dict : {"worker_id", "dentro", "total", "tempo_s"}
    """
    worker_id, n_amostras, seed = args
    rng      = np.random.default_rng(seed)
    dentro   = 0
    restante = n_amostras
    t0       = time.perf_counter()

    while restante > 0:
        lote  = min(MINI_LOTE, restante)
        x     = rng.uniform(-1.0, 1.0, lote)
        y     = rng.uniform(-1.0, 1.0, lote)
        dentro   += int(np.sum(x * x + y * y <= 1.0))
        restante -= lote

    return {
        "worker_id": worker_id,
        "dentro"   : dentro,
        "total"    : n_amostras,
        "tempo_s"  : time.perf_counter() - t0,
    }


# ─── Monitor de progresso (processo daemon) ───────────────────────────────────

def _monitor(t0_global: float, n_workers: int, stop_evt):
    while not stop_evt.is_set():
        elapsed     = time.perf_counter() - t0_global
        mins, secs  = divmod(int(elapsed), 60)
        barra       = "█" * min(int(elapsed / TARGET_SEGUNDOS * 20), 20)
        espaco      = "░" * (20 - len(barra))
        pct         = min(elapsed / TARGET_SEGUNDOS * 100, 100)
        print(f"\r  [{barra}{espaco}] {pct:5.1f}%  "
              f"{mins:02d}:{secs:02d}  |  {n_workers} worker(s) ativos",
              end="", flush=True)
        stop_evt.wait(CHECKPOINT_SECS)
    print()   # quebra de linha ao terminar


# ─── Função principal ─────────────────────────────────────────────────────────

def main() -> tuple:
    modo_str = "  [MODO RÁPIDO — validação]" if MODO_RAPIDO else ""
    print("=" * 65)
    print("  INTEGRAÇÃO DE MONTE CARLO PARALELIZADA — ESTIMATIVA DE π")
    print("=" * 65)
    print(f"  Data/hora      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python         : {sys.version.split()[0]}")
    print(f"  Workers        : {N_WORKERS}  (núcleos lógicos){modo_str}")
    print(f"  Total N        : {TOTAL_AMOSTRAS:,}  amostras")
    print(f"  N por worker   : {AMOSTRAS_WORKER:,}")
    print(f"  Erro teórico   : ≈ {4/np.sqrt(TOTAL_AMOSTRAS):.2e}  (O(1/√N))")
    print(f"  RAM por worker : ≈ {MINI_LOTE * 2 * 8 / 1024**2:.0f} MB  "
          f"(2 arrays float64 de {MINI_LOTE:,})")
    print("=" * 65)

    # Cada worker recebe uma seed única (seed_base + worker_id)
    args_lista = [
        (i, AMOSTRAS_WORKER, SEED_BASE + i)
        for i in range(N_WORKERS)
    ]

    # Inicia monitor de progresso em processo daemon
    stop_evt = mp.Event()
    monitor  = mp.Process(
        target  = _monitor,
        args    = (time.perf_counter(), N_WORKERS, stop_evt),
        daemon  = True,
    )
    print(f"\n  Iniciando execução paralela...\n")
    t_inicio = time.perf_counter()
    monitor.start()

    # ── EXECUÇÃO PARALELA ─────────────────────────────────────────────────
    with mp.Pool(processes=N_WORKERS) as pool:
        resultados = pool.map(worker_monte_carlo, args_lista)
    # ─────────────────────────────────────────────────────────────────────

    t_fim = time.perf_counter()
    stop_evt.set()
    monitor.join(timeout=3)

    # ─── Agregação ───────────────────────────────────────────────────────
    total_dentro  = sum(r["dentro"] for r in resultados)
    total_pts     = sum(r["total"]  for r in resultados)
    pi_estimado   = 4.0 * total_dentro / total_pts
    erro_abs      = abs(pi_estimado - np.pi)
    erro_rel      = erro_abs / np.pi * 100
    tempo_total   = t_fim - t_inicio
    throughput    = total_pts / tempo_total

    # ─── Relatório ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTADOS")
    print("=" * 65)
    print(f"  Tempo total          : {tempo_total:.2f} s  "
          f"({tempo_total / 60:.2f} min)")
    print(f"  Throughput           : {throughput:,.0f}  amostras/s")
    print(f"  Pontos totais        : {total_pts:,}")
    print(f"  Pontos dentro (acerto): {total_dentro:,}")
    print(f"  π estimado           : {pi_estimado:.10f}")
    print(f"  π real (numpy.pi)    : {np.pi:.10f}")
    print(f"  Erro absoluto        : {erro_abs:.4e}")
    print(f"  Erro relativo        : {erro_rel:.6f} %")
    print("=" * 65)
    print(f"\n  Detalhes por worker:")
    print(f"  {'ID':>3}  {'Amostras':>15}  {'Dentro':>15}  "
          f"{'π local':>12}  {'Tempo (s)':>10}")
    print(f"  {'─'*3}  {'─'*15}  {'─'*15}  {'─'*12}  {'─'*10}")
    for r in resultados:
        pi_loc = 4.0 * r["dentro"] / r["total"]
        print(f"  {r['worker_id']:>3}  {r['total']:>15,}  "
              f"{r['dentro']:>15,}  {pi_loc:>12.8f}  {r['tempo_s']:>10.2f}")

    print(f"\n  Análise de convergência (O(1/√N)):")
    print(f"    Esperado → {4/np.sqrt(total_pts):.4e}")
    print(f"    Obtido   → {erro_abs:.4e}")
    print("=" * 65)

    # ─── Exporta CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "resultado_monte_carlo.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("worker_id,amostras,dentro,pi_local,tempo_s\n")
        for r in resultados:
            pi_loc = 4.0 * r["dentro"] / r["total"]
            f.write(f"{r['worker_id']},{r['total']},{r['dentro']},"
                    f"{pi_loc:.10f},{r['tempo_s']:.4f}\n")
        f.write(f"TOTAL,{total_pts},{total_dentro},"
                f"{pi_estimado:.10f},{tempo_total:.4f}\n")
    print(f"\n  Resultado exportado → {csv_path}")

    return pi_estimado, tempo_total


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # if __name__ guard: OBRIGATÓRIO para multiprocessing no Windows/macOS
    main()
