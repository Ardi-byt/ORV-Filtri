import cv2 as cv
import numpy as np

def konvolucija(slika, jedro):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    # Dimenzije slike in jedra
    visina, sirina = slika.shape
    v_jedra, s_jedra = jedro.shape

    # Odmiki za centriranje jedra
    v_odmik = v_jedra // 2
    s_odmik = s_jedra // 2

    # Inicializacija izhodne slike
    rezultat = np.zeros_like(slika, dtype=np.float32)

    # Izvedba konvolucije
    for i in range(visina):
        for j in range(sirina):
            vrednost = 0
            for k in range(v_jedra):
                for l in range(s_jedra):
                    v_index = i + (k - v_odmik)
                    s_index = j + (l - s_odmik)

                    # Preveri, če je indeks znotraj meja slike
                    if 0 <= v_index < visina and 0 <= s_index < sirina:
                        vrednost += slika[v_index, s_index] * jedro[k, l]

            rezultat[i, j] = vrednost

    return rezultat


def filtriraj_z_gaussovim_jedrom(slika, sigma):
    '''Filtrira sliko z Gaussovim jedrom.'''
    # Izračun velikosti jedra po formuli
    velikost_jedra = int((2 * sigma) * 2 + 1)
    
    # Ustvarimo prazno jedro
    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)
    
    # Izračunamo vrednost k po formuli
    k = velikost_jedra / 2 - 1/2
    
    # Izračunamo vrednosti znotraj jedra po formuli
    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x = i - k
            y = j - k
            jedro[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i-k-1)**2 + (j-k-1)**2) / (2 * sigma**2))

    
    # Uporabimo prej implementirano funkcijo konvolucija za filtriranje slike
    return konvolucija(slika, jedro)



def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    # Pretvori sliko v sivinsko, če je barvna
    if len(slika.shape) > 2:
        siva = np.mean(slika, axis=-1)
    else:
        siva = slika

    # Definiraj Sobelova jedra
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Izračunaj gradient v x in y smeri
    gradient_x = konvolucija(siva, sobel_x)
    gradient_y = konvolucija(siva, sobel_y)

    # Izračunaj magnitudo gradienta
    magnituda = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Spremeni barvo pikslov z magnitudo večjo od 100 na rdečo
    filtrirana_slika = slika.copy()
    rdeca_barva = [0, 0, 255]  # BGR za rdečo barvo (OpenCV uporablja BGR)
    visina, sirina = magnituda.shape

    for i in range(visina):
        for j in range(sirina):
            if magnituda[i, j] > 100:
                filtrirana_slika[i, j] = rdeca_barva

    return filtrirana_slika


if __name__ == '__main__':
    # Nalozim lenna sliko
    slika = cv.imread(".utils/lenna.png", cv.IMREAD_GRAYSCALE)
    if slika is None:
        print("Napaka pri nalaganju slike!")
        exit()
    
    # Pretvorimo sliko v float32 za natančnejše računanje
    slika = np.float32(slika)
    
    # Testirajmo funkcijo filtriranje z gaussovim jedrom z razlicnimi sigmami
    sigma_vrednosti = [0.5, 1.0, 2.0, 3.0]
    
    # Prvo pokaze originalno sliko
    cv.imshow("Originalna slika", np.uint8(slika))
    
    # Filtriramo sliko z različnimi vrednostmi sigma
    for sigma in sigma_vrednosti:
        filtrirana_slika = filtriraj_z_gaussovim_jedrom(slika, sigma)
        
        # Pretvorimo rezultat nazaj v uint8 za prikaz
        filtrirana_slika_uint8 = np.uint8(np.clip(filtrirana_slika, 0, 255))
        
        # Prikažemo filtrirano sliko
        cv.imshow(f"Gaussov filter (sigma={sigma})", filtrirana_slika_uint8)
        
        # Izpišemo velikost uporabljenega jedra
        velikost_jedra = int((2 * sigma) * 2 + 1)
        print(f"Sigma: {sigma}, Velikost jedra: {velikost_jedra}x{velikost_jedra}")
    
    # Naložimo barvno sliko za Sobelov filter
    barvna_slika = cv.imread(".utils/lenna.png")
    if barvna_slika is None:
        print("Napaka pri nalaganju barvne slike!")
        exit()
    
    # Uporabimo funkcijo filtriraj_sobel_smer
    filtrirana_sobel = filtriraj_sobel_smer(barvna_slika)
    
    # Prikažemo sliko s poudarjenimi gradienti
    cv.imshow("Original sobel slika", filtrirana_sobel)
    
    cv.waitKey(0)
    cv.destroyAllWindows()


