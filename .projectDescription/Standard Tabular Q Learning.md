```table-of-contents
```

**Date:** 2025-11-20
**Tags:** #AI #ludo #reinforcementlearning 

---
# Method Name: Context-Aware Potential-Based Q-Learning


### **1. The Core Philosophy**

- **The Problem:** Standard Q-Learning fails in Ludo because the state space (positions of 16 pieces) is too vast ($\approx 10^{9}$ states). Agents take millions of games to learn that "Killing at Tile 10" is the same concept as "Killing at Tile 50."
    
- **The Solution:** We abstract **Capabilities** instead of **Positions**. The agent does not see _where_ it is; it sees _what it can do_ (e.g., "I have a Kill opportunity," "I have a Safety opportunity").
    
- **The Brain:** A standard Tabular Q-Table (no neural networks) that maps these strategic opportunities + the game score to a specific action value.
    

---

### **2. State Representation ($S$)**

The state is strictly defined as a tuple of 5 Integers:

$$S = (F_{P1}, F_{P2}, F_{P3}, F_{P4}, C)$$

#### **Part A: Piece Potentials ($F_{Pi}$)**

For each piece $P_1..P_4$, we simulate its move with the current dice roll and classify the **outcome** into one of **7 Tactical Categories**. This collapses the entire board into simple concepts.

| **ID** | **Name**    | **Trigger Condition (What happens if I move?)**                      | **Strategic Intent**                       |
| ------ | ----------- | -------------------------------------------------------------------- | ------------------------------------------ |
| **0**  | **Null**    | Piece cannot move (Stuck at home, Finished, or Blocked).             | N/A (Action Masked)                        |
| **1**  | **Neutral** | Moves to an empty field tile. No special interaction.                | Default progression.                       |
| **2**  | **Risk**    | Lands on a tile where an enemy is **1-6 steps behind**.              | **Danger Alert.** "Move to escape."        |
| **3**  | **Boost**   | Lands on a Star (Jump) OR moves significantly $> \text{Dice Value}$. | **Speed.** "Take the shortcut."            |
| **4**  | **Safety**  | Lands on a Globe, Safe Star, or enters the **Home Corridor**.        | **Defense.** "Bank the progress."          |
| **5**  | **Kill**    | Lands on an opponent's tile.                                         | **Aggression.** "Teleport enemy to start." |
| **6**  | **Goal**    | Enters the final Winning Triangle.                                   | **Victory.** The ultimate goal.            |

#### **Part B: Team Context ($C$)**

A global variable that tells the agent "Are we winning or losing?" This allows the agent to switch strategies (Aggressive vs. Defensive) based on the game phase.

The Weighted Equity Score:

Instead of just counting "safe pieces" (which is myopic), we score the actual board state:

$$\text{Score} = (\text{Goal} \times 100) + (\text{Home\_Corridor} \times 50) + (\text{Safe\_Globe} \times 10) + (\text{Raw\_Distance})$$

The Gap Calculation:

$$\text{Gap} = \text{My\_Score} - \text{Max}(\text{Opponent\_Scores})$$

The 3 Context States:

| ID  | Name     | Condition                   | Behavior to Learn                                                |
| --- | -------- | --------------------------- | ---------------------------------------------------------------- |
| 0   | Trailing | $\text{Gap} < -20$          | Panic Mode. High risk tolerance. Value Kills/Boosts over Safety. |
| 1   | Neutral  | $-20 \le \text{Gap} \le 20$ | Balanced Race. Standard play. (Start of game is here).           |
| 2   | Leading  | $\text{Gap} > 20$           | Lockdown Mode. Zero risk. Value Safety/Defense over Kills.       |


Total State Space: $7^4 \times 3 = \mathbf{7,203 \text{ States}}$.

(This fits easily in L1 Cache for extreme speed).

---

### **3. Action Space & Masking**

- **Actions:** `[0, 1, 2, 3]` corresponding to `Move P1`, `Move P2`, `Move P3`, `Move P4`.
    
- **Validity Masking:** Before querying the Q-Table, we check the Potentials. If a piece has **Potential 0 (Null)**, its action is **masked (forbidden)**. The agent effectively chooses from the _valid_ pieces only.
    

---

### **4. Reward Shaping (The "Teacher")**

To force the agent to learn distinct strategies for "Winning" vs "Losing," we use **Dynamic Reward Scaling**.

|**Event**|**Base Reward**|**Trailing Multiplier (Panic)**|**Leading Multiplier (Secure)**|**Logic**|
|---|---|---|---|---|
|**Goal**|`+100`|x 1.0 (+100)|x 1.0 (+100)|Always good.|
|**Kill**|`+50`|**x 1.5 (+75)**|x 0.8 (+40)|"If losing, Kills are critical."|
|**Safety**|`+15`|x 0.5 (+7.5)|**x 2.0 (+30)**|"If winning, Safety is priority."|
|**Boost**|`+10`|x 1.2 (+12)|x 1.0 (+10)|Speed matters in a race.|
|**Neutral**|`+1`|x 1.0 (+1)|x 1.0 (+1)|Better than nothing.|
|**Risk**|`-10`|x 0.5 (-5)|**x 2.0 (-20)**|"Risks are fatal when leading."|
|**Death**|`-20`|x 1.0 (-20)|x 1.0 (-20)|Getting killed is always bad.|

---

### **5. The Execution Loop (Step-by-Step)**

1. **Observation:**
    - Dice Roll: `6`.
    - Board State: We are ahead by 40 points.
    - 
2. **Context Check:** `Gap = +40`. Context is **Leading (2)**.
    
3. **Abstraction (Feature Extraction):**
    - $P1$: Can Kill enemy (`5`).
    - $P2$: Can enter Home Corridor (`4`).
    - $P3$: Can Move normally (`1`).
    - $P4$: Stuck at home (`0`).
    - **State Tuple:** `(5, 4, 1, 0, 2)`.
        
4. **Q-Table Lookup:**
    
    - The agent checks row `(5, 4, 1, 0, 2)`.
    - It sees that in Context 2 (Leading), the value for **Action 1 (Safety)** is higher than **Action 0 (Kill)** because of previous training with the "Leading Multiplier."

5. **Decision:** Choose **Action 1 (Move P2)** to secure the win safely.
    
6. **Update:** The agent receives reward `+30` (Base 15 * 2.0). It updates the Q-value for that state-action pair.

---

### **6. Why This Design is Robust**

1. **Handles "Teleporting":** The **Kill (5)** potential ensures the agent instantly recognizes the value of sending an enemy home, regardless of where it happens.
    
2. **Handles "Call to Defense":** The **Leading Context (2)** combined with the **Safety (4)** potential explicitly teaches the agent to "chill" and protect pieces when it has the lead.
    
3. **Handles "Local Threats":** Even if the context is "Leading," if a specific piece has **Risk (2)** potential (enemy behind), the agent sees a unique state tuple `(Risk, ...)` and learns to prioritize saving that specific piece.