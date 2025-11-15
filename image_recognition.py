import os
from PIL import Image
from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig, HttpOptions
from google import genai
import asyncio

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file!")


client = genai.Client(
    api_key=GEMINI_API_KEY, http_options=HttpOptions(api_version="v1")
)
model_id = "gemini-2.5-flash"


# Image path is the file path of the image being used
async def ReturnStringy(image_path, client, model_id):
    img = Image.open(image_path)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            (
                "Convert **only the math** in the image to clean string. "
                "No extra text, no explanations. "
                "Define position [0,0] as the bottom left corner of the image. "
                "First index of position is x and second one is y. "
                "For each character, assign it with an array [topleft, topright, leftbottom, rightbottom]. "
                "These are the corners in which the character lays over. "
                'Something like {"label": "2", "box": [[0,20],[20,20],[0,0],[20,0]]}.'
            ),
            img,
        ],
        config=GenerateContentConfig(temperature=0),
    )
    content = response.text
    return content


async def async_handwriting_to_latex(
    image_path: str, client: genai.Client, model_id: str
):
    img = Image.open(image_path)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            (
                "You are a perfect handwriting-to-LaTeX OCR. "
                "Convert **only the math** in the image to clean LaTeX. "
                "Return **only** the LaTeX code inside $$ delimiters. "
                "No extra text, no explanations."
            ),
            img,
        ],
        config=GenerateContentConfig(
            temperature=0,
        ),
    )
    if response.text is None:
        raise ValueError("gemini did not cook")
    text = response.text.strip()

    if "$$" in text:
        start = text.find("$$") + 2
        end = text.rfind("$$")
        latex_body = text[start:end].strip()
        return "$" + latex_body + "$"
    else:
        return "There was an error"


async def async_ForEachReturnError(
    image_path: str, client: genai.Client, model_id: str
):
    """
    Asynchronously transcribes a math problem and checks for errors,
    returning a structured string.
    """
    img = Image.open(image_path)

    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            (
                "In this json response, a math problem is inscribed. This includes fractions, arthmetic."
                "transcribe the following problem into an equation"
                "if error is present for the label that is incorrect, put error=True, expectedresult={answer}"
                'For example,your input would look somehting like this: { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] }'
            ),
            (
                'Your output would look something like { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] , "error":True, "expectedresult=4}'
            ),
            img,
        ],
        config=GenerateContentConfig(
            temperature=0,
        ),
    )
    return response.text


async def main():
    image_path = "/home/dilyxs/Downloads/sample_let.png"
    print(f"Processing image: {image_path}")
    basicpositionlabelling = await ReturnStringy(image_path, client, model_id)
    print(basicpositionlabelling)
    latext = await async_handwriting_to_latex(image_path, client, model_id)
    print(latext)


asyncio.run(main())
