from typing import Annotated
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fasthtml.common import (
    Html,
    Script,
    Head,
    Title,
    Body,
    Div,
    Form,
    Input,
    Img,
    P,
    to_xml,
    Style,
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    Alert,
    AlertTitle,
    AlertDescription,
    Button,
    Badge,
    Separator,
    Lucide,
    Progress,
)
import base64

# Create main FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="FastAPI application serving an ONNX model for image classification",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
INPUT_SIZE = (160, 160)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LABELS = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']
BREED_FACTS = {
    'Beagle': "Beagles have approximately 220 million scent receptors, compared to a human's mere 5 million!",
    'Boxer': "Boxers were among the first dogs to be employed as police dogs and were used as messenger dogs during wartime.",
    'Bulldog': "Despite their tough appearance, Bulldogs were bred to be companion dogs and are known for being gentle and patient.",
    'Dachshund': "Dachshunds were originally bred to hunt badgers - their name literally means 'badger dog' in German!",
    'German_Shepherd': "German Shepherds can learn a new command in as little as 5 repetitions and obey it 95% of the time.",
    'Golden_Retriever': "Golden Retrievers were originally bred as hunting dogs to retrieve waterfowl without damaging them.",
    'Labrador_Retriever': "Labs have a special water-resistant coat and a unique otter-like tail that helps them swim efficiently.",
    'Poodle': "Despite their elegant appearance, Poodles were originally water retrievers, and their fancy haircut had a practical purpose!",
    'Rottweiler': "Rottweilers are descendants of Roman drover dogs and were used to herd livestock and pull carts for butchers.",
    'Yorkshire_Terrier': "Yorkies were originally bred to catch rats in clothing mills. Despite their small size, they're true working dogs!"
}

# Load the ONNX model
try:
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("models/onnx_model.onnx")
    ort_session.run(
        ["output"], {"input": np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)}
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: dict  # Change to dict for class probabilities
    success: bool
    message: str


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model inference

    Args:
        image (PIL.Image): Input image

    Returns:
        np.ndarray: Preprocessed image array
    """
    # Convert to RGB if not already
    image = image.convert("RGB")

    # Resize
    image = image.resize(INPUT_SIZE)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply mean and std normalization
    img_array = (img_array - MEAN) / STD

    # Transpose to channel-first format (NCHW)
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)

    return img_array


# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def ui_home():
    content = Html(
        Head(
            Title("Dog Breed Classifier"),
            ShadHead(tw_cdn=True, theme_handle=True),
            Script(
                src="https://unpkg.com/htmx.org@2.0.3",
                integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                crossorigin="anonymous",
            ),
            Style(
                """
                body {
                    background: url('https://media.giphy.com/media/3o7aD0QrhhK5TYwpuU/giphy.gif'); /* Giphy GIF as background */
                    background-size: cover; /* Cover the entire background */
                    background-repeat: no-repeat; /* Do not repeat the image */
                    background-attachment: fixed; /* Keep the background fixed during scrolling */
                }
                """
            ),
        ),
        Body(
            Div(
                Card(
                    CardHeader(
                        Div(
                            CardTitle("Dog Breed Classifier 🐶"),
                            Badge("AI Powered", variant="secondary", cls="w-fit bg-gradient-to-r from-blue-500 to-green-500 text-white font-bold hover:from-blue-600 hover:to-green-600 transition duration-300"),
                            cls="flex items-center justify-between",
                        ),
                        CardDescription(
                            "Upload an image to classify the breed of the dog. Our AI model will analyze it instantly!"
                        ),
                    ),
                    CardContent(
                        Form(
                            Div(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        required=True,
                                        cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 file:cursor-pointer",
                                    ),
                                    P(
                                        "Drag and drop an image or click to browse",
                                        cls="text-sm text-muted-foreground text-center mt-2",
                                    ),
                                    cls="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors",
                                ),
                                Img(
                                    src="",
                                    id="file-preview",
                                    alt="File Preview",
                                    cls="mt-4 w-full h-auto rounded-lg shadow-lg hidden",
                                ),
                                Button(
                                    Lucide("sparkles", cls="mr-2 h-4 w-4"),
                                    "Classify Image",
                                    type="submit",
                                    cls="w-full bg-gradient-to-r from-blue-500 to-green-500 text-white hover:from-blue-600 hover:to-green-600 transition duration-300",
                                ),
                                cls="space-y-4",
                            ),
                            enctype="multipart/form-data",
                            hx_post="/classify",
                            hx_target="#result",
                        ),
                        Div(id="result", cls="mt-6"),
                    ),
                    cls="w-full max-w-3xl shadow-lg transition-transform transform hover:scale-105",
                    standard=True,
                ),
                cls="container flex items-center justify-center min-h-screen p-4",
            ),
            cls="bg-background text-foreground",
        ),
    )
    return to_xml(content)


@app.post("/classify", response_class=HTMLResponse)
async def ui_handle_classify(file: Annotated[bytes, File()]):
    try:
        response = await predict(file)
        image_b64 = base64.b64encode(file).decode("utf-8")

        predicted_class = max(response.predictions.items(), key=lambda x: x[1])[0]
        confidence = max(response.predictions.values())

        # Get the fact about the predicted class
        breed_fact = BREED_FACTS.get(predicted_class, "No facts available for this breed.")

        # Create the results display with grid layout
        results = Div(
            Div(
                # Left column - Image
                Div(
                    Img(
                        src=f"data:image/jpeg;base64,{image_b64}",
                        alt="Uploaded image",
                        cls="w-full rounded-lg shadow-lg aspect-square object-cover",
                    ),
                    CardHeader(
                        Div(
                            CardTitle("Did You Know?", cls="text-black"),
                            Lucide("info"),
                            cls="flex items-center justify-between",
                        ),
                        cls="text-lg font-bold",
                    ),
                    CardContent(
                        P(f"{breed_fact}", cls="font-medium text-black"),
                    ),
                    cls="mt-4 bg-blue-100 border border-blue-300 rounded-lg p-4",
                ),
                # Right column - Results
                Div(
                    Badge(
                        f"It's a {predicted_class.lower()}!",
                        variant="outline",
                        cls=f"{'bg-green-500/20 hover:bg-green-500/20 border-green-500/50' if confidence > 0.8 else 'bg-yellow-500/20 hover:bg-yellow-500/20 border-yellow-500/50'} text-lg",
                    ),
                    # Confidence Progress Section
                    Div(
                        Div(
                            P("Confidence Score", cls="font-medium"),
                            P(
                                f"{confidence:.1%}",
                                cls="text-xl font-bold",
                            ),
                            cls="flex justify-between items-baseline",
                        ),
                        Progress(
                            value=int(confidence * 100),
                            cls="h-2",
                        ),
                        cls="mt-4 space-y-2",
                    ),
                    Separator(cls="my-6"),
                    Separator(cls="my-6"),

                    # Detailed Analysis Section
                    P("Top 5 Predictions", cls="font-semibold mb-2"),
                    Div(
                        *[
                            Div(
                                Div(
                                    P(f"{label}", cls="font-medium"),
                                    P(
                                        f"{prob:.1%}",
                                        cls=f"font-medium { "" if label == predicted_class else 'text-muted-foreground'}",
                                    ),
                                    cls="flex justify-between items-center",
                                ),
                                Progress(
                                    value=int(prob * 100),
                                    cls="h-2",
                                ),
                                cls="space-y-2",
                            )
                            for label, prob in response.predictions.items()
                        ],
                        cls="space-y-4",
                    ),
                ),
                cls="grid grid-cols-1 md:grid-cols-2 gap-6",
            ),
            cls="animate-in fade-in-50 duration-500",
        )

        return to_xml(results)

    except Exception as e:
        error_alert = Alert(
            AlertTitle("Error ❌"),
            AlertDescription(str(e)),
            variant="destructive",
            cls="mt-4",
        )
        return to_xml(error_alert)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Annotated[bytes, File(description="Image file to classify")]):
    try:
        image = Image.open(io.BytesIO(file))
        processed_image = preprocess_image(image)

        outputs = ort_session.run(
            ["output"], {"input": processed_image.astype(np.float32)}
        )

        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))

        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        predictions = {LABELS[i]: float(probabilities[i]) for i in top_indices}

        return PredictionResponse(
            predictions=predictions, success=True, message="Classification successful"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True}, status_code=200
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
