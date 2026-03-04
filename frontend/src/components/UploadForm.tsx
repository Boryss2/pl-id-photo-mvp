"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { processImage, ProcessPreset } from "../lib/api";

type Status = "idle" | "processing" | "ready" | "paid";

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preset, setPreset] = useState<ProcessPreset>("id");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [agreed, setAgreed] = useState(false);
  const [processedBlob, setProcessedBlob] = useState<Blob | null>(null);
  const [paid, setPaid] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const downloadUrl = useMemo(() => {
    if (!processedBlob || !paid) {
      return null;
    }
    return URL.createObjectURL(processedBlob);
  }, [processedBlob, paid]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setStatusMessage(null);
    setPaid(false);
    setProcessedBlob(null);
    setStatus("idle");
    const selected = event.target.files?.[0] || null;
    if (!selected) {
      setFile(null);
      return;
    }
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Akceptujemy tylko pliki JPG lub PNG.");
      setFile(null);
      return;
    }
    setFile(selected);
  };

  const renderPreview = async (blob: Blob) => {
    const image = new Image();
    const url = URL.createObjectURL(blob);
    image.src = url;
    await image.decode();

    const canvas = canvasRef.current;
    if (!canvas) {
      URL.revokeObjectURL(url);
      return;
    }
    const maxWidth = 360;
    const scale = maxWidth / image.width;
    const width = maxWidth;
    const height = image.height * scale;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      URL.revokeObjectURL(url);
      return;
    }

    ctx.filter = "blur(2px)";
    ctx.drawImage(image, 0, 0, width, height);
    ctx.filter = "none";

    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "bold 20px Arial";
    ctx.rotate(-0.2);
    for (let y = -height; y < height * 2; y += 50) {
      for (let x = -width; x < width * 2; x += 180) {
        ctx.fillText("PODGLĄD", x, y);
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    URL.revokeObjectURL(url);
  };

  const handleProcess = async () => {
    if (!file) {
      setError("Dodaj zdjęcie przed przetwarzaniem.");
      return;
    }
    if (!agreed) {
      setError("Zaznacz zgodę RODO przed kontynuacją.");
      return;
    }
    setError(null);
    setStatusMessage(null);
    setStatus("processing");
    const result = await processImage(file, preset);
    if (!result.ok) {
      setStatus("idle");
      if (result.errorCode) {
        setStatusMessage(`Błąd ${result.errorCode}: ${result.error}`);
      } else {
        setStatusMessage(result.error);
      }
      return;
    }
    setProcessedBlob(result.blob);
    setStatusMessage("Sukces: otrzymaliśmy finalny plik.");
    setStatus("ready");
    await renderPreview(result.blob);
  };

  const handlePaymentStub = () => {
    setPaid(true);
    setStatus("paid");
  };

  return (
    <section className="card">
      <h2>Wgraj zdjęcie</h2>
      <div className="grid">
        <div>
          <label>
            Plik JPG/PNG
            <input className="input" type="file" onChange={handleFileChange} />
          </label>
          <label style={{ display: "block", marginTop: 12 }}>
            Wariant
            <select
              className="input"
              value={preset}
              onChange={(event) =>
                setPreset(event.target.value as ProcessPreset)
              }
            >
              <option value="id">Dowód (492×633)</option>
              <option value="passport">Paszport (min 768×1004)</option>
            </select>
          </label>
          <label style={{ display: "block", marginTop: 12 }}>
            <input
              type="checkbox"
              checked={agreed}
              onChange={(event) => setAgreed(event.target.checked)}
            />{" "}
            Akceptuję RODO i politykę prywatności (
            <Link href="/privacy">link</Link>)
          </label>

          <button
            className="cta"
            style={{ marginTop: 16 }}
            onClick={handleProcess}
            disabled={!file || status === "processing"}
          >
            {status === "processing" ? "Przetwarzamy..." : "Przetwórz zdjęcie"}
          </button>

          {error && (
            <p style={{ color: "#b91c1c", marginTop: 12 }}>{error}</p>
          )}
          {statusMessage && (
            <p style={{ color: "#065f46", marginTop: 12 }}>{statusMessage}</p>
          )}
        </div>

        <div>
          <h3>Podgląd finalnego pliku (zabezpieczony)</h3>
          <canvas ref={canvasRef} className="watermark" />
          {!processedBlob && (
            <p className="muted" style={{ marginTop: 8 }}>
              Podgląd pojawi się po przetworzeniu.
            </p>
          )}

          {status === "ready" && (
            <div style={{ marginTop: 16 }}>
              <p className="status">
                Gotowe. To jest finalny plik po pełnym przetworzeniu. Aby pobrać,
                przejdź do płatności.
              </p>
              <button className="cta" onClick={handlePaymentStub}>
                Zapłać 19 zł
              </button>
              <p className="muted" style={{ marginTop: 8 }}>
                Płatność testowa (stub). Po wdrożeniu zostanie podłączony
                operator płatności.
              </p>
            </div>
          )}

          {status === "paid" && downloadUrl && (
            <div style={{ marginTop: 16 }}>
              <p className="status">Płatność potwierdzona. Plik jest finalny.</p>
              <a className="cta" href={downloadUrl} download="zdjecie.jpg">
                Pobierz zdjęcie
              </a>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
