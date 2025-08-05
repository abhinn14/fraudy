from flask import Flask, request, jsonify
from ml import load_models, assess
import os
from flask_cors import CORS

# --- Basic App Setup ---
app = Flask(__name__)
CORS(app)
models = load_models()
# -------------------------

@app.route("/assess", methods=["POST"])
def assess_transaction():
    """
    Receives transaction data and assesses its fraud risk.
    It expects the 'time' field to be an integer hour (0-23),
    as the conversion is now handled by the upstream Express server.
    """
    data = request.json
    print(f"Received data for assessment: {data}")

    # The time conversion block has been REMOVED from here.
    # The 'assess' function will now receive data with 'time' already as an hour.

    try:
        result = assess(data, models)
        risk = result["risk"].replace(" Risk", "")  # Clean up the risk string

        response = {
            "risk": risk,
            "probabilities": result["probs"],
            "flags": result["flags"],
            "is_fraud": False  # Default to False
        }

        # Set is_fraud to True only for High risk transactions
        if risk == "High":
            response["is_fraud"] = True

        print(f"Assessment result: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"💥 Error during assessment: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
