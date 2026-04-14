# ADR 008: Definition of the Cognition Layer (L6)

## Background
The Saccade project was initially designed with a 5-layer architecture (L1-L5). However, during the implementation phase, a new directory `cognition/` emerged, containing modules like `frame_selector.py` and `resource_manager.py`. These modules handle high-level resource monitoring and frame selection decisions, which do not neatly fit into the existing L1-L5 framework. The lack of a formalized architectural definition for these components has led to ambiguity in their role within the system.

## Decision
We formally define the `cognition/` directory as **Layer 6 (L6): Cognition & Resource Management**.

- **Responsibility:** High-level system monitoring, dynamic resource allocation, adaptive frame selection, and overall system health management.
- **Core Components:** 
  - `resource_manager.py`: Monitors GPU VRAM, CPU utilization, and other system metrics, applying limits or adjusting processing thresholds accordingly.
  - `frame_selector.py`: Adaptive frame dropping and selection logic based on system load and scene entropy.
- **Placement:** Sits alongside or above the pipeline orchestration, influencing how data flows through L1 and L2 by adjusting parameters dynamically based on hardware limits.

This layer explicitly formalizes the project's requirement for adaptive resource management, especially critical for edge devices with strict VRAM and compute constraints.

## Trade-offs
- **Pros:** Clarifies the architectural role of `cognition/` modules, ensuring they have a defined place in the system blueprint. Promotes better separation of concerns by keeping resource-aware throttling out of the core perception logic (L1/L2).
- **Cons:** Increases the formal layer count from 5 to 6, requiring updates to existing documentation to reflect the expanded scope.
