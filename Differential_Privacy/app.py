from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('dp_model')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'input' not in data:
            return jsonify({"error": "Invalid input data"}), 400
        
        input_data = np.array(data['input']).reshape(1, 28, 28) / 255.0
        
        predictions = model.predict(input_data)
        predicted_label = np.argmax(predictions, axis=1)[0]
        
        return jsonify({"predicted_label": int(predicted_label)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
