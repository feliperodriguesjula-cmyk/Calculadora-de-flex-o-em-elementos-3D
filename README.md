# Calculadora de Flexão em Elementos Estruturais 3D (FEM)

Aplicação desenvolvida em Python + Streamlit para análise de flexão de vigas utilizando o Método dos Elementos Finitos (Euler-Bernoulli).

## 🔧 Funcionalidades

- Aplicação de cargas estilo CAD (plano + ponto no plano)
- Análise em dois planos (XY / XZ)
- Cálculo de:
  - Flecha máxima (mm)
  - Momento fletor
  - Força cortante
  - Tensão máxima
  - Verificação por Von Mises
- Comparação com:
  - Tensão admissível (fy / FS)
  - Limite de escoamento
- Seções:
  - Retangular
  - Barra redonda
  - Tubo redondo
  - Tubo retangular
  - Seção composta
- Catálogo de materiais editável

---

## 📐 Modelo Teórico

Modelo de viga de Euler-Bernoulli com discretização 1D:

- 2 DOFs por nó (w, θ)
- Matriz de rigidez clássica:
  
  EI/L³ * matriz 4x4

- Pós-processamento para:
  - Momento interno
  - Cortante
  - Deformada

---

## ▶ Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
