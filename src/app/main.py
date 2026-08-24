from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
)


@app.get("/")
def health_check():
    """
    Return the API health status.

    This endpoint can be used by monitoring services and load balancers
    to verify that the application is running.
    """
    return {"status": "ok"}


class CustomerData(BaseModel):
    """
    Input schema for customer churn prediction.

    Defines the customer features required by the inference pipeline.
    """

    # Demographic features
    gender: str
    Partner: str
    Dependents: str

    # Phone service features
    PhoneService: str
    MultipleLines: str

    # Internet service features
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    # Contract and billing features
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    # Numerical features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Predict whether a customer is likely to churn.

    The input is validated using the CustomerData schema and then
    passed to the inference pipeline.

    Returns:
        A JSON response containing the churn prediction or an error.
    """
    try:
        customer_data = customer.dict()
        prediction = predict(customer_data)

        return {"prediction": prediction}

    except Exception as error:
        return {"error": str(error)}


def gradio_interface(
    gender,
    partner,
    dependents,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    tenure,
    monthly_charges,
    total_charges,
):
    """
    Process Gradio form inputs and return a churn prediction.

    The function converts the UI inputs into the same data structure
    expected by the inference pipeline.
    """
    customer_data = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    prediction = predict(customer_data)

    return str(prediction)


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(
            ["Male", "Female"],
            label="Gender",
            value="Male",
        ),
        gr.Dropdown(
            ["Yes", "No"],
            label="Partner",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No"],
            label="Dependents",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No"],
            label="Phone Service",
            value="Yes",
        ),
        gr.Dropdown(
            ["Yes", "No", "No phone service"],
            label="Multiple Lines",
            value="No",
        ),
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service",
            value="Fiber optic",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Security",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Backup",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Device Protection",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Tech Support",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming TV",
            value="Yes",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming Movies",
            value="Yes",
        ),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month",
        ),
        gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing",
            value="Yes",
        ),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        ),
        gr.Number(
            label="Tenure (months)",
            value=1,
            minimum=0,
            maximum=100,
        ),
        gr.Number(
            label="Monthly Charges ($)",
            value=85.0,
            minimum=0,
            maximum=200,
        ),
        gr.Number(
            label="Total Charges ($)",
            value=85.0,
            minimum=0,
            maximum=10000,
        ),
    ],
    outputs=gr.Textbox(
        label="Churn Prediction",
        lines=2,
    ),
    title="Telco Customer Churn Predictor",
    description="""
Predict customer churn using a machine learning model.

Fill in the customer information below to receive a churn prediction.
The model uses XGBoost trained on historical telecom customer data.
""",
    examples=[
        [
            "Female",
            "No",
            "No",
            "Yes",
            "No",
            "Fiber optic",
            "No",
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Month-to-month",
            "Yes",
            "Electronic check",
            1,
            85.0,
            85.0,
        ],
        [
            "Male",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "DSL",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "Two year",
            "No",
            "Credit card (automatic)",
            60,
            45.0,
            2700.0,
        ],
    ],
    theme=gr.themes.Soft(),
)


app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui",
)