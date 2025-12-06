# Combined State Space Analysis: Zone-Based + Potential-Based

## Current State Spaces

### Zone-Based (12 dimensions)
```
(HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)
```
- **HOME/PATH/SAFE/GOAL**: Token counts per zone (0-4 each)
- **EV1-EV4**: Enemy vulnerable flags per token (0-1 each)
- **TV1-TV4**: Token vulnerable flags per token (0-1 each)

### Potential-Based (5 dimensions)
```
(P1, P2, P3, P4, Context)
```
- **P1-P4**: Potential classifications per token (0-6: NULL, NEUTRAL, RISK, BOOST, SAFETY, KILL, GOAL)
- **Context**: Game state (0-2: TRAILING, NEUTRAL, LEADING)

---

## Redundancy Analysis

### 🔴 **High Redundancy** (Can be eliminated)

#### 1. **EV flags vs POT_KILL**
- **EV flags**: Check if token CAN capture enemy (requires movability + next_pos can capture)
- **POT_KILL**: Check if next_pos can capture enemy (part of potential classification)
- **Overlap**: Both indicate capture opportunities
- **Difference**: EV requires movability check, POT_KILL is more general
- **Decision**: Keep POT_KILL (more informative), remove EV flags

#### 2. **TV flags vs POT_RISK** (Partial)
- **TV flags**: Token is under threat NOW (enemy within 6 moves)
- **POT_RISK**: Probability of risk at NEXT position
- **Overlap**: Both indicate danger
- **Difference**: TV is current threat, POT_RISK is future risk
- **Decision**: Keep both (complementary: current vs future risk)

### 🟡 **Medium Redundancy** (Can be optimized)

#### 3. **Zone counts vs Individual piece locations**
- **Zone counts**: Aggregate (HOME=2, PATH=1, SAFE=1, GOAL=0)
- **Potential per piece**: Individual piece states (P1=KILL, P2=HOME, etc.)
- **Overlap**: Zone counts can be inferred from individual piece states
- **Difference**: Zone gives distribution, potential gives per-piece detail
- **Decision**: Keep zone counts (strategic overview) + keep potential (tactical detail)
- **Optimization**: Zone counts are already aggregated, so no redundancy here

#### 4. **Context vs Zone distribution** (Indirect)
- **Context**: Derived from weighted equity score (leading/trailing/neutral)
- **Zone counts**: Token distribution across zones
- **Overlap**: Both indicate game state
- **Difference**: Context is relative (vs opponent), zone is absolute
- **Decision**: Keep both (complementary: relative vs absolute)

### 🟢 **No Redundancy** (Keep both)

#### 5. **POT_BOOST, POT_SAFETY, POT_GOAL**
- These are unique to potential classification
- No equivalent in zone-based
- **Decision**: Keep all

#### 6. **Zone distribution (HOME/PATH/SAFE/GOAL)**
- Strategic overview of token positions
- Not captured by individual potentials
- **Decision**: Keep

---

## Optimized Combined State Space Design

### **Option 1: Minimal Redundancy (Recommended)** ⭐
```
Dimensions: 15 total

Zone Distribution (4):
  - HOME, PATH, SAFE, GOAL (counts 0-4)

Per-Token Tactical (8):
  - P1, P2, P3, P4 (potential classifications 0-6)
  - TV1, TV2, TV3, TV4 (current threat flags 0-1)

Game Context (1):
  - Context (trailing/neutral/leading 0-2)

Aggregate Tactical (2):
  - Num pieces with POT_KILL (0-4)
  - Num pieces with POT_RISK (0-4)
```

**Rationale:**
- Removed EV flags (redundant with POT_KILL)
- Kept TV flags (current threat, complementary to POT_RISK)
- Added aggregate counts for quick strategic decisions
- Total: 15 dimensions (vs 17 if just concatenated)

### **Option 2: Maximum Information (No Aggregation)**
```
Dimensions: 13 total

Zone Distribution (4):
  - HOME, PATH, SAFE, GOAL

Per-Token Detail (8):
  - P1, P2, P3, P4 (potentials)
  - TV1, TV2, TV3, TV4 (current threats)

Game Context (1):
  - Context
```

**Rationale:**
- Removed EV flags (redundant)
- Kept everything else
- Total: 13 dimensions (most compact)

### **Option 3: Enhanced with Aggregates**
```
Dimensions: 17 total

Zone Distribution (4):
  - HOME, PATH, SAFE, GOAL

Per-Token Tactical (8):
  - P1, P2, P3, P4
  - TV1, TV2, TV3, TV4

Game Context (1):
  - Context

Aggregate Counts (4):
  - Num pieces with POT_KILL (0-4)
  - Num pieces with POT_RISK (0-4)
  - Num pieces with POT_SAFETY (0-4)
  - Num pieces with POT_BOOST (0-4)
```

**Rationale:**
- Maximum information
- Aggregate counts for quick strategic decisions
- Total: 17 dimensions (same as concatenation, but better organized)

---

## Recommended Implementation: **Option 1** ⭐

### State Tuple Structure
```python
(
    # Zone distribution (strategic overview)
    home_count,      # 0-4
    path_count,      # 0-4
    safe_count,      # 0-4
    goal_count,      # 0-4
    
    # Per-token tactical (individual piece states)
    p1, p2, p3, p4,  # 0-6 (potential classifications)
    tv1, tv2, tv3, tv4,  # 0-1 (current threat flags)
    
    # Game context (relative state)
    context,         # 0-2 (trailing/neutral/leading)
    
    # Aggregate tactical (quick strategic decisions)
    num_kill_potential,   # 0-4 (pieces with POT_KILL)
    num_risk_potential,   # 0-4 (pieces with POT_RISK)
)
```

### Benefits
1. **Eliminates redundancy**: Removed EV flags (covered by POT_KILL)
2. **Maintains information**: All unique features from both abstractions
3. **Adds strategic aggregates**: Quick counts for decision-making
4. **Compact**: 15 dimensions vs 17 if just concatenated
5. **Well-organized**: Clear separation of strategic vs tactical

### Implementation Notes
- **TV flags**: Keep because they indicate CURRENT threat (complementary to POT_RISK which is FUTURE risk)
- **Aggregate counts**: Help agent quickly assess "how many pieces can kill" or "how many are at risk"
- **Context**: Keep because it's relative (vs opponent), zone counts are absolute

---

## Comparison Table

| Feature | Zone-Only | Potential-Only | Concatenated | Optimized (Option 1) |
|---------|-----------|---------------|--------------|---------------------|
| **Dimensions** | 12 | 5 | 17 | 15 |
| **Zone Distribution** | ✅ | ❌ | ✅ | ✅ |
| **Per-Token Potentials** | ❌ | ✅ | ✅ | ✅ |
| **EV Flags** | ✅ | ❌ | ✅ | ❌ (redundant) |
| **TV Flags** | ✅ | ❌ | ✅ | ✅ (complementary) |
| **Context** | ❌ | ✅ | ✅ | ✅ |
| **Aggregate Counts** | ❌ | ❌ | ❌ | ✅ (new) |
| **Redundancy** | None | None | High | Low |

---

## Expected Performance

### Theoretical Advantages
1. **More information** than either alone
2. **Less redundancy** than simple concatenation
3. **Better organization** for learning
4. **Strategic + tactical** coverage

### Potential Challenges
1. **Larger state space** (15 vs 12 dimensions) = more training needed
2. **More complex** = slower convergence
3. **Need to verify** aggregates help vs hurt

### Recommendation
- **Test Option 1 first** (15 dimensions, minimal redundancy)
- **Compare with zone-based** (12 dimensions) to see if extra info helps
- **If too slow**, try Option 2 (13 dimensions, no aggregates)

---

## Next Steps

1. ✅ **Review this analysis** - Confirm redundancies identified correctly
2. ✅ **Choose option** - Option 1 (recommended) or Option 2/3
3. ✅ **Implement** - Add `'combined'` or `'hybrid'` mode to TabularQAgent
4. ✅ **Test** - Run same training config and compare with zone-based
5. ✅ **Evaluate** - Does combined perform better than zone-based alone?

