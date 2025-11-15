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
def handwriting_to_latex(image_path: str) -> str:
    img = Image.open(image_path)
    response = model.generate_content(
        [
            "You are a perfect handwriting-to-LaTeX OCR. "
            "Convert **only the math** in the image to clean LaTeX. "
            "Return **only** the LaTeX code inside $$ delimiters. "
            "No extra text, no explanations.",
            img
        ],
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="text/plain"
        )
    )
    text = response.text.strip()
    if "$$" in text:
        start = text.find("$$") + 2
        end = text.rfind("$$")
        latex_body = text[start:end].strip()
        return "$" + latex_body + "$" #Forces inline math latex, as gemini returns without dollar signs
    else:
        return "There was an error"

if __name__ == "__main__":
    #Using a manual path for testing purposes
    image_path = '/Users/hasan/Downloads/WhatsApp Image 2025-11-14 at 22.06.06.jpeg'       
    print(f"Processing image: {image_path}")
    latex = handwriting_to_latex(image_path)
    print(latex)


    
    
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python handwriting_to_latex.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")
    latex = handwriting_to_latex(image_path)
    print("\nLaTeX Output:")
    print("=" * 40)
    print(latex)
    print("=" * 40)
    
    """