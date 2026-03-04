import io

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from .pipeline import process_image
from .postprocess import adjust_zoom, finalize_output
from .validators import ProcessingError, ensure_file_size

MAX_INPUT_BYTES = 10_000_000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process-image")
async def process_image_endpoint(
    file: UploadFile = File(...), preset: str = Form(...)
) -> Response:
    content = await file.read()
    try:
        ensure_file_size(len(content), MAX_INPUT_BYTES)
        output = process_image(content, preset)
        zoomed_img, _ = adjust_zoom(Image.open(io.BytesIO(output)).convert("RGB"))
        final_bytes = finalize_output(zoomed_img, preset)
        return Response(content=final_bytes, media_type="image/jpeg")
    except ProcessingError as error:
        return JSONResponse(
            status_code=400,
            content={"error_code": error.code, "message": error.message},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "SERVER_ERROR",
                "message": "Wystąpił błąd serwera. Spróbuj ponownie.",
            },
        )


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}
