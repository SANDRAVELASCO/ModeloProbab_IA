# =============================
# MODELO NAIVE BAYES CON SKLEARN
# PREDICCIÓN DE ACTIVIDAD FÍSICA
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
# Paso 1: Creación de un conjunto de datos simulado
data = {
    'frecuencia_cardiaca': [80, 150, 100, 130, 85, 170, 120, 160, 90, 140],
    'calorias': [150, 500, 250, 400, 180, 600, 300, 550, 170, 450],
    'duracion_min': [30, 20, 25, 25, 40, 15, 35, 18, 45, 22],
    'actividad': ['caminar', 'correr', 'caminar', 'correr', 'caminar', 
                  'correr', 'bicicleta', 'correr', 'caminar', 'bicicleta']
}
# Convertimos los datos en un DataFrame
df = pd.DataFrame(data)
# Paso 2: Separar características (X) y etiquetas (y)
X = df[['frecuencia_cardiaca', 'calorias', 'duracion_min']]
y = df['actividad']
# Paso 3: Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Paso 4: Crear y entrenar el modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)
# Paso 5: Hacer predicciones
y_pred = modelo.predict(X_test)
# Paso 6: Evaluar el modelo
print("🔎 Exactitud del modelo:", accuracy_score(y_test, y_pred))
print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred))
# Paso 7 (opcional): Predecir una nueva actividad
nueva_entrada = [[125, 350, 30]]  # frecuencia cardiaca, calorías, duración
prediccion = modelo.predict(nueva_entrada)
print("\n🔮 Predicción para nueva entrada:", prediccion[0])
