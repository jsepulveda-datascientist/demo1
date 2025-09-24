# Session 4 – API de ML con Flask, Entrenamiento y Docker

Este proyecto expone un servicio de Machine Learning en un único contenedor Docker, con endpoints para **salud**, **predicción** y **reentrenamiento** del modelo.

## 📦 Requisitos

Instalar dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

## ▶️ Ejecutar localmente

```bash
python -m src.main
```

## 🌐 Endpoints disponibles

### 1. Health check

Verifica que el servicio está corriendo.

```bash
curl http://127.0.0.1:8080/api/health
```

Respuesta esperada:

```json
{"status": "ok"}
```

---

### 2. Predicción

Envía instancias con atributos `Age` y `EstimatedSalary`:

```bash
curl -X POST http://127.0.0.1:8080/api/predict   -H "Content-Type: application/json"   -d '{"instances":[{"Age":35,"EstimatedSalary":45000},{"Age":52,"EstimatedSalary":120000}]}'
```

Ejemplo de respuesta:

```json
{
  "results": [
    {"prediction": 0, "proba": 0.23},
    {"prediction": 1, "proba": 0.81}
  ]
}
```

---

### 3. Entrenamiento

Permite reentrenar el modelo en caliente. Sobrescribe el archivo `model.pkl`.

#### a) Sintético (fallback)

```bash
curl -X POST http://127.0.0.1:8080/api/train   -H "Content-Type: application/json"   -d '{}'
```

#### b) Desde CSV

> Se espera que el CSV tenga las columnas `Age`, `EstimatedSalary` y una columna objetivo (`Purchased`, `label`, `target` o `y`).

```bash
curl -X POST http://127.0.0.1:8080/api/train   -H "Content-Type: application/json"   -d '{"csv_path":"data/SocialNetworkAds.csv"}'
```

#### c) Desde JSON inline

```bash
curl -X POST http://127.0.0.1:8080/api/train   -H "Content-Type: application/json"   -d '{"instances":[
        {"Age":35,"EstimatedSalary":45000,"label":0},
        {"Age":52,"EstimatedSalary":120000,"label":1}
      ]}'
```

---

## 🐳 Uso con Docker

### Construir la imagen

```bash
docker build -t ml-session4:latest .
```

### Ejecutar el contenedor

```bash
docker run --rm -p 8080:8080 -e MODEL_PATH=/app/model.pkl ml-session4:latest
```

### Reentrenar dentro del contenedor

```bash
curl -X POST http://127.0.0.1:8080/api/train -H "Content-Type: application/json" -d '{}'
```

### Persistir el modelo entrenado

```bash
mkdir -p artifacts
docker run --rm -p 8080:8080   -v $(pwd)/artifacts:/app   -e MODEL_PATH=/app/model.pkl   ml-session4:latest
```

El modelo entrenado quedará guardado en `./artifacts/model.pkl`.
