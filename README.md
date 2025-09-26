# Proyecto Sprint 9 — Megaline: Recomendador de Plan

## Contexto del sprint
Megaline, un operador de telefonía, necesita un sistema que recomiende a sus clientes cuándo migrar de **Smart (0)** a **Ultra (1)**.  

El objetivo es entrenar, validar y documentar un **clasificador supervisado** que logre un **accuracy ≥ 0.75 en un conjunto de test bloqueado**, asegurando además interpretabilidad y pruebas de robustez.

---

## Estructura del repositorio
```
Proyecto_Sprint_9/
│
├── notebooks/
│   └── Proyecto_Sprint_9:_Megaline_Recomendador_de_Plan.ipynb
│
├── data/
│   └── users_behavior.csv
│
├── README.md
└── requirements.txt
```

---

## Notebook principal

### `Proyecto_Sprint_9:_Megaline_Recomendador_de_Plan.ipynb`
Incluye todo el flujo del proyecto:
1. Introducción y objetivo  
2. Protocolo e imports  
3. Carga y validación de datos  
4. Partición en train/valid/test  
5. Baselines (Dummy y heurística por MB)  
6. Modelos supervisados (Logistic Regression, Decision Tree, Random Forest)  
7. Optimización de hiperparámetros (Random Forest con RandomizedSearchCV)  
8. Evaluación final en test  
9. Pruebas de cordura (bootstrap, shuffle, baseline aleatorio, permutation importance, curvas ROC y PR)  
10. Conclusiones  

---

## Pipeline del proyecto

1. **Datos**  
   - Columnas: `calls, minutes, messages, mb_used, is_ultra`.  
   - Sin nulos ni duplicados.  
   - Balance de clases: ~70% Smart, ~30% Ultra.  

2. **Partición**  
   - 60% train, 20% valid, 20% test.  
   - Estratificación para preservar proporciones.  

3. **Baselines**  
   - DummyClassifier (predice la mayoría).  
   - Heurística simple (`mb_used >= mediana`).  

4. **Modelos supervisados**  
   - Regresión Logística (con escalado).  
   - Árbol de Decisión.  
   - Random Forest.  

5. **Optimización de hiperparámetros**  
   - Random Forest ajustado con `RandomizedSearchCV` (scoring = balanced accuracy, 5 folds).  
   - Mejores parámetros: profundidad moderada, `n_estimators` altos, `max_features` = sqrt/log2, con o sin `class_weight='balanced'`.  

6. **Evaluación final en test**  
   - Reentrenamiento en train+valid.  
   - Métricas: accuracy, balanced accuracy, matriz de confusión.  
   - Intervalo de confianza 95% (bootstrap con reentrenamiento).  

7. **Pruebas de cordura**  
   - Predicciones de una sola clase.  
   - Baseline aleatorio vs dummy.  
   - Etiquetas barajadas (shuffle).  
   - Importancia de features por permutación.  
   - Curvas ROC y Precision-Recall.  

8. **Conclusiones**  
   - Random Forest optimizado fue el mejor modelo.  
   - Cumplió el umbral de accuracy ≥ 0.75 en test.  
   - `mb_used` resultó la variable más relevante.  
   - El modelo tiende a subpredecir Ultra; si se prioriza recall, conviene ajustar umbral.

---

## Resultados principales

- **Random Forest optimizado**  
  - Accuracy en test: **≈0.80**  
  - Balanced accuracy: **≈0.78**  
  - IC 95% accuracy (bootstrap): **[0.767, 0.851]**

- **Matriz de confusión (test):**

|               | Pred Smart | Pred Ultra |
|---------------|------------|------------|
| Real Smart    | 248        |  34        |
| Real Ultra    |  40        |  74        |

---

## Requisitos y ejecución

### Dependencias
- Python 3.10+  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `scikit-learn`  

Instalación rápida:
```bash
pip install -r requirements.txt
```

### Ejecución
1. Clonar el repo.  
2. Colocar `users_behavior.csv` en la carpeta `data/`.  
3. Abrir el notebook y ejecutar las celdas secuencialmente.  

---

## Reproducibilidad

- **Semilla global** (`RANDOM_STATE = 42`).  
- **CV estratificada (5 folds)** para tuning.  
- **Bootstrap (B=300)** para IC 95% en test.  

---

## Autoría
Proyecto desarrollado por **Luis Liévano**.  
