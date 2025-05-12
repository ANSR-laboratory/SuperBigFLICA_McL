## Column Index and Descriptions

| Index | Column Name      | Description |
|-------|------------------|-------------|
| 0     | `CDRSB`          | Clinical Dementia Rating – Sum of Boxes |
| 1     | `SITEID`         | Site identifier |
| 2     | `PTGENDER`       | Participant gender (1 = male, 2 = female) |
| 3     | `PTAGE`          | Participant age in years |
| 4     | `GDS`            | Geriatric Depression Scale total score |
| 5     | `NPI`            | Total Neuropsychiatric Inventory (NPI) score |

### NPI Sub-item Scores (Columns 6–17)

| Index | Label | NPI Domain |
|-------|-------|-------------|
| 6     | A     | Delusions |
| 7     | B     | Hallucinations |
| 8     | C     | Agitation/Aggression |
| 9     | D     | Depression/Dysphoria |
| 10    | E     | Anxiety |
| 11    | F     | Elation/Euphoria |
| 12    | G     | Apathy/Indifference |
| 13    | H     | Disinhibition |
| 14    | I     | Irritability/Lability |
| 15    | J     | Aberrant Motor Behavior |
| 16    | K     | Sleep/Nighttime Behavior |
| 17    | L     | Appetite and Eating Disorders |

### NPI Composite Subsyndromes (Columns 18–21)

| Index | Variable          | Composition |
|-------|-------------------|-------------|
| 18    | `NPI_Psychosis`   | Delusions + Hallucinations + Sleep |
| 19    | `NPI_Affective`   | Depression + Anxiety |
| 20    | `NPI_Hyperactivity` | Agitation + Irritability + Euphoria + Motor Behavior + Disinhibition |
| 21    | `NPI_Apathy`      | Apathy + Appetite |

### Clinical Labels (Columns 22–23)

| Index | Variable    | Description |
|-------|-------------|-------------|
| 22    | `A_Status`  | Amyloid PET status (1 = positive, 0 = negative, or missing) |
| 23    | `C_Status`  | Cognitive status (e.g., 1 = cognitively impaired, 0 = healthy control) |

---

> ⚠️ **Note**: This dataset is not for public distribution. Please ensure that any use complies with ADNI's data sharing policies.
