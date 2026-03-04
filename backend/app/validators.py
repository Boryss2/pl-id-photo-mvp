from dataclasses import dataclass


@dataclass
class ProcessingError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def ensure_file_size(file_size: int, max_bytes: int) -> None:
    if file_size > max_bytes:
        raise ProcessingError(
            code="INPUT_FILE_TOO_LARGE",
            message="Plik wejściowy jest za duży.",
        )


def ensure_face_detected(detected: bool) -> None:
    if not detected:
        raise ProcessingError(
            code="FACE_NOT_DETECTED",
            message="Nie wykryto twarzy na zdjęciu.",
        )


def ensure_output_ratio(width: int, height: int) -> None:
    target_ratio = 35 / 45
    ratio = width / height
    if abs(ratio - target_ratio) > 0.01:
        raise ProcessingError(
            code="OUTPUT_RATIO_INVALID",
            message="Nieprawidłowe proporcje zdjęcia wyjściowego.",
        )


def ensure_output_resolution(width: int, height: int, preset: str) -> None:
    if preset == "id" and (width < 492 or height < 633):
        raise ProcessingError(
            code="OUTPUT_RESOLUTION_INVALID",
            message="Zbyt niska rozdzielczość dla wariantu dowodowego.",
        )
    if preset == "passport" and (width < 768 or height < 1004):
        raise ProcessingError(
            code="OUTPUT_RESOLUTION_INVALID",
            message="Zbyt niska rozdzielczość dla wariantu paszportowego.",
        )


def ensure_face_ratio(face_ratio: float) -> None:
    if face_ratio < 0.5 or face_ratio > 0.8:
        raise ProcessingError(
            code="FACE_SIZE_OUT_OF_RANGE",
            message="Twarz nie mieści się w zalecanym zakresie 50–80% kadru.",
        )


def ensure_chin_to_bottom(chin_to_bottom_ratio: float) -> None:
    if chin_to_bottom_ratio < 0.08 or chin_to_bottom_ratio > 0.22:
        raise ProcessingError(
            code="CHIN_POSITION_INVALID",
            message="Nieprawidłowa odległość brody od dolnej krawędzi zdjęcia.",
        )


def ensure_output_file_size(file_size: int, max_bytes: int) -> None:
    if file_size > max_bytes:
        raise ProcessingError(
            code="OUTPUT_FILE_TOO_LARGE",
            message="Plik wyjściowy przekracza limit 2,5 MB.",
        )
