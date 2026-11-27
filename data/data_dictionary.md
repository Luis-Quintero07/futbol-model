# Data Dictionary

Total rows: 12,528
Columns: 35

| Column | Type | Description |
|--------|------|-------------|
| Div | str | Código de liga (E0, SP1, D1, I1, F1) (100.0% completado) |
| Season | str | Temporada (ej. 2023-2024) (100.0% completado) |
| Date | datetime64[us] | Fecha del partido (100.0% completado) |
| HomeTeam | str | Equipo local (nombre estandarizado) (100.0% completado) |
| AwayTeam | str | Equipo visitante (100.0% completado) |
| FTHG | Int64 | Goles local tiempo completo (100.0% completado) |
| FTAG | Int64 | Goles visitante tiempo completo (100.0% completado) |
| FTR | str | Resultado original (H/D/A) (100.0% completado) |
| HTHG | Int64 | Goles local medio tiempo (100.0% completado) |
| HTAG | Int64 | Goles visitante medio tiempo (100.0% completado) |
| HTR | str |  (100.0% completado) |
| HS | float64 | Tiros local (100.0% completado) |
| AS | float64 | Tiros visitante (100.0% completado) |
| HST | float64 | Tiros a puerta local (100.0% completado) |
| AST | float64 | Tiros a puerta visitante (100.0% completado) |
| HF | float64 | Faltas local (100.0% completado) |
| AF | float64 | Faltas visitante (100.0% completado) |
| HC | float64 | Corners local (100.0% completado) |
| AC | float64 | Corners visitante (100.0% completado) |
| HY | float64 | Amarillas local (100.0% completado) |
| AY | float64 | Amarillas visitante (100.0% completado) |
| HR | float64 | Rojas local (100.0% completado) |
| AR | float64 | Rojas visitante (100.0% completado) |
| B365H | float64 | Odds Bet365 victoria local (100.0% completado) |
| B365D | float64 | Odds Bet365 empate (100.0% completado) |
| B365A | float64 | Odds Bet365 victoria visitante (100.0% completado) |
| B365>2.5 | float64 | Odds Bet365 Over 2.5 goles (85.4% completado) |
| B365<2.5 | float64 | Odds Bet365 Under 2.5 goles (85.4% completado) |
| B365AHH | float64 |  (85.3% completado) |
| B365AHA | float64 |  (85.3% completado) |
| Result | int64 | Resultado numérico (1/0/-1) (100.0% completado) |
| imp_prob_H | float64 | Prob. implícita local (sin vigorish) (100.0% completado) |
| imp_prob_D | float64 | Prob. implícita empate (sin vigorish) (100.0% completado) |
| imp_prob_A | float64 | Prob. implícita visitante (sin vigorish) (100.0% completado) |
| overround | float64 | Margen casa (vigorish) (100.0% completado) |