# Recursive-Entropic-Time-An-Internal-Subjective-Signal-Framework-for-Emergent-Consciousness
#  Recursive Entropic Time Agent: *The Planner*

This project implements a biologically inspired artificial agent based on the **Recursive Entropic Time (RET)** framework. It explores the emergence of goal-directed behavior, foresight, and internal timing by modeling cognition as an entropy-driven, self-paced process. The agent is embedded in a dynamic 2D environment and learns through its own subjective sense of stress and internal drives.

---

##  Core Concept: Recursive Entropic Time (RET)

Traditional reinforcement learning uses external reward signals and fixed time steps. In contrast, RET agents measure **internal cognitive stress** as a combination of:

* Hunger (drive entropy)
* Motor failure (external surprise)

Time is subjective: the agent advances internal learning only when enough internal tension (`Tint`) has accumulated. This aligns more closely with how animals and humans experience meaningful moments.

---

##  Overview of the System

### 1. **Environment**

A 2D world (`World`) containing:

* Empty space
* Walls (obstacles)
* Randomly placed food

The agent perceives a 5x5 grid around itself (`get_view`) and can perform five actions: up, down, left, right, or eat.

---

### 2. **Agent Architecture**

The agent, `GroundedMind_v15`, is composed of:

* `VisionNet`: A convolutional neural network encoding local spatial perception.
* `MotorNet`: A feedforward controller that chooses actions based on perception and current strategy (e.g., exploring vs. pursuing food).
* `RET Engine`: An internal stress meter (`S_total`) and subjective time accumulator (`Tint`) control when learning happens.

#### Key features:

* **Mental simulation**: The agent simulates multiple future action sequences and selects the plan with the lowest predicted stress.
* **Strategic switching**: A planning strategy vector (e.g., `[Exploring, GoToFood]`) modulates action choice.
* **Learning**: When a meaningful "tick" is reached, the agent updates its motor controller using its real-world experience.

---

## 🚀 How It Works

### Main Loop (Simplified):

1. **Perceive**: Agent sees a 5x5 window of the world.
2. **Plan**: It mentally simulates 16 possible futures over 4 steps each, evaluating predicted cognitive load.
3. **Act**: Chooses the first action of the best simulated future.
4. **Feel**: Computes `S_hunger`, `S_motor` (stress from bumping into a wall), and combines into `S_total`.
5. **Tick**: If `Tint` passes threshold, the agent learns by comparing outcome to expectation.
6. **Repeat**.

---

##  Example Outcome

```bash
SUCCESS! Food eaten at (10, 9) at time 1357.
--- TICK 71 @ 1360: Strategy='Exploring', S_Total=0.00 ---
...
****** AGENT SURVIVED THE ENTIRE SIMULATION! ******
Total Food Eaten: 18.
```

The RET agent learns to survive, plan ahead, and adapt strategies using its own internally generated time and stress feedback.

---

##  Visualization

The simulation ends with a plot showing cumulative food eaten over time, illustrating periods of planning, success, and survival.

---

##🔧 Dependencies

* Python 3.7+
* PyTorch
* NumPy
* Matplotlib

Install with:

```bash
pip install torch numpy matplotlib
```

---

##  Why This Matters

This project demonstrates that agents can develop **foresight, strategic behavior, and autonomous learning** using internal drives and subjective time, without relying on externally defined reward functions or fixed schedules.

It is an early prototype of a **recursive, embodied AI** architecture—closer to natural cognition than conventional reinforcement learning.

