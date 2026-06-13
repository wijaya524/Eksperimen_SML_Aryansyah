import os
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. Tentukan URI Model Anda (Arahkan ke folder mlruns lokal)
# Pastikan path mlruns Anda terdeteksi dengan benar
model_uri = "runs:/60c27036f7e647aeb8a19f8f084c6a71/model"

print("[-] Memuat model dari MLflow Local Store...")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("[+] Model berhasil dimuat!")
except Exception as e:
    print(f"[!] Gagal memuat model. Error: {e}")
    exit(1)

# 2. Membuat Endpoint Scoring / Predict
@app.route('/invocations', methods=['POST'])
def predict():
    try:
        # Menerima data JSON dari client
        data_json = request.get_json()
        
        # Mengubah data input menjadi Pandas DataFrame
        # Format JSON standard MLflow biasanya: {"dataframe_records": [{...}, {...}]} 
        # atau {"columns": [...], "data": [[...]]}
        if "dataframe_records" in data_json:
            df = pd.DataFrame(data_json["dataframe_records"])
        elif "columns" in data_json and "data" in data_json:
            df = pd.DataFrame(data=data_json["data"], columns=data_json["columns"])
        else:
            # Fallback jika json berupa list dict biasa [{...}]
            df = pd.DataFrame(data_json)
            
        # Melakukan prediksi menggunakan model MLflow
        predictions = model.predict(df)
        
        # Mengembalikan hasil prediksi ke client
        return jsonify({"predictions": predictions.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 3. Menjalankan Server Menggunakan Waitress (Aman untuk Windows)
if __name__ == '__main__':
    try:
        from waitress import serve
        print("[+] Memulai server model menggunakan Waitress pada http://127.0.0.1:5002")
        serve(app, host='127.0.0.1', port=5002)
    except ImportError:
        print("[!] Pustaka 'waitress' tidak ditemukan. Menjalankan via Flask Development Server...")
        print("[+] Memulai server model pada http://127.0.0.1:5002")
        app.run(host='127.0.0.1', port=5002, debug=False)