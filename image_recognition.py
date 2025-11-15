import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# Image path is the file path of the image being used
def ReturnStringy(str) -> str:
    img = Image.open(image_path)
    response = model.generate_content(
        [
            "Convert **only the math** in the image to clean string"
            "No extra text, no explanations."
            "define position [0,0] as the botom left corner of the image"
            "first index of position is x and second one is y"
            "For each character, assign it with an array [topleft, topright,leftbottom,rightbottom]"
            "these are the corners in which the character lays over"
            'something like { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] },',
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0, response_mime_type="text/plain"
        ),
    )
    return response.text


def ForEachReturnError(str) -> str:
    img = Image.open(image_path)
    response = model.generate_content(
        [
            "In this json response, a math problem is inscribed. This includes fractions, arthmetic."
            "transcribe the following problem into an equation"
            "if error is present for the label that is incorrect, put error=True, expectedresult={answer}"
            'For example,your input would look somehting like this: { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] }',
            'Your output would look something like { "label": "2", "box": [[0,20],[20,20] ,[0,0] [20, 0] , "error":True, "expectedresult=4}',
            img,
        ],
        generation_config=genai.GenerationConfig(
            temperature=0, response_mime_type="text/plain"
        ),
    )
    return response.text


if __name__ == "__main__":
    image_path = "/home/dilyxs/Downloads/sample_let.png"
    print(f"Processing image: {image_path}")
    basicpositionlabelling = ReturnStringy(image_path)
    print(basicpositionlabelling)
    perfectjson = ForEachReturnError(basicpositionlabelling)
    print(perfectjson)
