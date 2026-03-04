import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Zdjęcie do dokumentów (PL)",
  description: "Szybki plik JPG zgodny z wymaganiami technicznymi.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="pl">
      <body>{children}</body>
    </html>
  );
}
