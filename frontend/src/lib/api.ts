export type ProcessPreset = "id" | "passport";

export type ProcessResult =
  | { ok: true; blob: Blob }
  | { ok: false; error: string; errorCode?: string };

const apiBase =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

export async function processImage(
  file: File,
  preset: ProcessPreset
): Promise<ProcessResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("preset", preset);

  try {
    const response = await fetch(`${apiBase}/process-image`, {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      const data = (await response.json().catch(() => null)) as
        | { message?: string; error_code?: string }
        | null;
      return {
        ok: false,
        error: data?.message || "Nie udało się przetworzyć zdjęcia.",
        errorCode: data?.error_code,
      };
    }

    const blob = await response.blob();
    return { ok: true, blob };
  } catch (error) {
    return {
      ok: false,
      error: "Błąd połączenia z serwerem. Spróbuj ponownie.",
    };
  }
}
