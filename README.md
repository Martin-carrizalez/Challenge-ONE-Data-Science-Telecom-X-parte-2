# 🤖 TelecomX — Parte 2: Predicción de Cancelación (Churn)
**Oracle Next Education · Alura LATAM — Challenge 2 Data Science**  
**Autor:** Angel

---

## 📊 Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Dataset de entrada | `telecomx_datos_limpios.csv` (Parte 1) |
| Variable objetivo | `Evasion` (0 = Permanece · 1 = Evade) |
| Modelos entrenados | LR · Decision Tree · Random Forest · HistGradientBoosting · Voting |
| Mejor modelo | Voting Classifier (LR + RF + HGB, soft) |
| Técnica de balance | SMOTE solo en train (post-split) |
| Validación | GridSearchCV + 5-Fold Cross-Validation |

---

## 🔹 Descripción del Proyecto

En la **Parte 1** se realizó el análisis exploratorio (EDA) de los datos de clientes de TelecomX.  
En esta **Parte 2** se construye un pipeline de Machine Learning profesional para **predecir qué clientes
tienen mayor probabilidad de cancelar**, permitiendo intervenciones proactivas de retención.

### Pipeline (orden estricto — sin Data Leakage)
```
CSV limpio
  → Drop columnas irrelevantes
  → One-Hot Encoding
  → train_test_split 80/20    ← PRIMERO el split
  → SMOTE solo en train       ← DESPUÉS el balance
  → StandardScaler en train   ← fit solo en train
  → GridSearchCV (cv=5)       ← búsqueda en train
  → Evaluar en test set       ← solo al final
```

---

## 📁 Estructura del Proyecto

```
📦 TelecomX-Churn/
 ┣ 📓 TelecomX_LATAM.ipynb           ← Parte 1: EDA
 ┣ 📓 TelecomX_Parte2_ML.ipynb       ← Parte 2: Machine Learning
 ┣ 📊 telecomx_datos_limpios.csv     ← Dataset procesado (salida Parte 1)
 ┣ 📄 README.md                      ← Parte 1
 ┗ 📄 README_Parte2.md               ← Este archivo
```

---

## 🔄 Flujo del Notebook

| Paso | Contenido |
|------|-----------|
| 1 | Carga del CSV limpio + verificación de calidad |
| 2 | Drop de `ID_Cliente` y `Cargo_Diario` (multicolinealidad) |
| 3 | One-Hot Encoding con `drop_first=True` |
| 4 | Análisis de balance de clases |
| 5 | Split 80/20 → SMOTE → StandardScaler |
| 6 | Correlación y análisis dirigido |
| 7 | Variantes manuales — 3 LR + 3 RF comparadas |
| 8 | GridSearchCV — optimización de hiperparámetros |
| 9 | K-Fold Cross-Validation (K=5) con ImbPipeline |
| 10 | Curvas de aprendizaje — diagnóstico overfitting |
| 11 | Curva ROC + matrices de confusión + tabla final |
| 12 | Ajuste de umbral (threshold tuning) |
| 13 | Importancia de variables |
| 14 | Análisis de errores (FN y FP) |
| 15 | Modelo Ensamble — Voting Classifier (LR + RF + HGB) |
| 16 | Extra — Justificación técnica del ensamble (4 indicios) |
| 17 | Informe final y estrategia de retención |

---

## 🧠 Modelos Entrenados

| Modelo | Normalización | Optimización | Notas |
|--------|--------------|-------------|-------|
| Regresión Logística | ✅ StandardScaler | GridSearchCV (L1/L2, C) | Interpretable por coeficientes |
| Árbol de Decisión | ❌ | Manual (max_depth, min_samples_leaf) | Baseline de árbol simple |
| Random Forest | ❌ | GridSearchCV (n_estimators, max_depth) | Mejor modelo individual |
| HistGradientBoosting | ❌ | Manual | Equivalente a XGBoost, nativo sklearn |
| **Voting Classifier** | ❌ | Soft voting pesos 1:2:2 | **Modelo final recomendado** |

---

## 📈 Resultados

### ¿Por qué Recall es la métrica principal?

| Error | Consecuencia | Costo |
|-------|-------------|-------|
| Falso Negativo (cliente se va y no lo detectamos) | Pérdida definitiva del ingreso recurrente | 🔴 Alto |
| Falso Positivo (contactamos a quien no se iba) | Campaña de retención innecesaria | 🟡 Bajo |

### Principales factores predictores

| Rango | Variable | Dirección | Interpretación |
|-------|----------|-----------|----------------|
| 1 | `Meses_Contrato` | ↓ Negativo | Más antigüedad → menos churn |
| 2 | `Cargo_Total` | ↓ Negativo | Mayor gasto acumulado = más comprometido |
| 3 | `Cargo_Mensual` | ↑ Positivo | Precio alto sin valor percibido → riesgo |
| 4 | Contrato mensual | ↑ Positivo | Sin compromiso de largo plazo |
| 5 | Cheque electrónico | ↑ Positivo | Pago no automático = intención de salida |
| 6 | Fibra óptica | ↑ Positivo | Precio/valor no percibido como justo |
| 7 | Soporte técnico | ↓ Negativo | Servicios adicionales fidelizan |

---

## 🧩 Modelo Ensamble — Voting Classifier

Combina tres modelos con **soft voting** (promedio de probabilidades):

```
LR (peso 1) ──┐
RF (peso 2) ──┼──→ Promedio ponderado → Predicción final
HGB (peso 2) ─┘
```

**4 indicios técnicos que justifican el ensamble:**

1. **Relación no lineal** — La curva LOWESS muestra que la evasión no sigue una recta.
2. **Interacciones entre variables** — Fibra+Mensual+Cheque tiene 2.5× más riesgo que el promedio.
3. **Desbalance de clases** — Los ensambles son más robustos que LR ante distribuciones sesgadas.
4. **Outliers en cargos** — El Bagging reduce la sensibilidad a valores extremos.

---

## 💡 Estrategias de Retención

| Prioridad | Factor de riesgo | Acción | Impacto estimado |
|-----------|-----------------|--------|-----------------|
| 🔴 ALTA | Antigüedad < 12 meses | Onboarding activo + contacto en mes 6 | -15% churn temprano |
| 🔴 ALTA | Contrato mensual | Descuento 10–15% al migrar a anual | -10% churn global |
| 🔴 ALTA | Cheque electrónico | Incentivo para pago automático | -5% churn |
| 🟡 MEDIA | Sin soporte/seguridad | Cross-selling paquete servicios | +ARPU y -churn |
| 🟡 MEDIA | Fibra óptica + cargo alto | Revisión precio/plan | -8% churn segmento |

### Impacto financiero estimado

| Acción | Clientes retenidos | Ingreso mensual recuperado |
|--------|-------------------|---------------------------|
| Migración a contrato anual | ~280 | ~$20,720 USD/mes |
| Onboarding activo | ~187 | ~$13,838 USD/mes |
| Cross-selling servicios | ~140 | ~$10,360 USD/mes |
| **Total** | **~607** | **~$44,918 USD/mes** |

---

## ▶️ Cómo ejecutar

### Google Colab (recomendado)
1. Subir `TelecomX_Parte2_ML.ipynb` y `telecomx_datos_limpios.csv` a Colab
2. Ejecutar: `!pip install imbalanced-learn`
3. **Runtime → Run all**

### Local
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook TelecomX_Parte2_ML.ipynb
```

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-SMOTE-blueviolet)

---

*Desarrollado como parte del **Challenge 2 — Data Science** del programa Oracle Next Education con Alura LATAM.*
