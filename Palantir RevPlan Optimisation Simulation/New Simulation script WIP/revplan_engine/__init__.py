"""
revplan_engine — MLWB port of the Palantir RevPlan "Scenario-Based Simulation"
(UC2-2) capacity-allocation engine.

Single-scenario, in-memory port of the Foundry pipeline. The heavy engine modules
are byte-identical copies of the Palantir source (see each module's header); all
Foundry coupling is isolated in `_foundry_shim`. The orchestrator and entry point
live in `run_simulation`.

Typical use (inside an MLWB notebook):

    from revplan_engine.run_simulation import main, SimulationParams
    results = main(SimulationParams(simulation_name="my scenario"))
"""
