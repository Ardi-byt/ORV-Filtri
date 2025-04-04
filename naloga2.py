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
    pass


def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    pass

if __name__ == '__main__':
    pass

