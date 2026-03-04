import UploadForm from "../components/UploadForm";

export default function HomePage() {
  return (
    <main className="container">
      <section className="card" style={{ marginBottom: 24 }}>
        <h1>Zdjęcie do dokumentów (PL) bez stresu</h1>
        <p className="muted">
          Wgrywasz zdjęcie, my przygotowujemy plik JPG o poprawnych parametrach
          technicznych. Bez AI i bez upiększania.
        </p>
        <div className="grid" style={{ marginTop: 16 }}>
          <div className="card">
            <h3>Jak to działa</h3>
            <ol>
              <li>Prześlij zdjęcie JPG/PNG</li>
              <li>Zobacz zabezpieczony podgląd</li>
              <li>Zapłać 19 zł i pobierz plik</li>
            </ol>
          </div>
          <div className="card">
            <h3>Co dostajesz</h3>
            <p>
              Otrzymujesz plik zgodny z wymaganiami technicznymi. Decyzja o
              akceptacji należy do urzędu.
            </p>
            <p className="muted">
              Nie gwarantujemy akceptacji. Nie poprawiamy urody ani rysów twarzy.
            </p>
          </div>
        </div>
      </section>

      <UploadForm />
    </main>
  );
}
