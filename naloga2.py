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

    # Razširimo sliko z ničlami na robovih (padding)
    razsirjena_slika = np.pad(slika, ((v_odmik, v_odmik), (s_odmik, s_odmik)), mode='constant', constant_values=0)

    # Inicializacija izhodne slike
    rezultat = np.zeros_like(slika, dtype=np.float32)

    # Konvolucija z vektorskim racunanjem
    for i in range(visina):
        for j in range(sirina):
            # Za vsak piksel izrežemo okno iz razširjene slike
            okno_slike = razsirjena_slika[i:i+v_jedra, j:j+s_jedra]
            
            # Izračunamo vrednost konvolucije kot vsoto produkta okna in jedra
            rezultat[i, j] = np.sum(okno_slike * jedro)

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
            jedro[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i-k)**2 + (j-k)**2) / (2 * sigma**2))

    
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
    rdeca_barva = [0, 0, 255]
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
    
    # Vprašanje 1: Primerjava detekcije robov na temni in svetli sliki
    # Nalozim barvno sliko za primerjavo
    primerjava_slika = cv.imread(".utils/lenna.png")
    if primerjava_slika is None:
        print("Napaka pri nalaganju slike za primerjavo!")
        exit()
    
    # Ustvarimo temno in svetlo verzijo slike
    temna_slika = np.clip(primerjava_slika * 0.3, 0, 255).astype(np.uint8)
    svetla_slika = np.clip(primerjava_slika * 2.0, 0, 255).astype(np.uint8)
    
    # Prikazemo temno in svetlo sliko
    cv.imshow("Temna slika (30% svetlosti)", temna_slika)
    cv.imshow("Svetla slika (200% svetlosti)", svetla_slika)
    
    # Uporabimo Sobelov filter na temni in svetli sliki
    sobel_temna = filtriraj_sobel_smer(temna_slika)
    sobel_svetla = filtriraj_sobel_smer(svetla_slika)
    
    # Prikazemo rezultate detekcije robov
    cv.imshow("Sobel filter - temna slika", sobel_temna)
    cv.imshow("Sobel filter - svetla slika", sobel_svetla)

    # Vprašanje 2: Zakaj je pred uporabo detektorja robov smiselno uporabiti filter za glajenje?
    # Naložimo barvno sliko
    barvna_slika = cv.imread(".utils/lenna.png")
    if barvna_slika is None:
        print("Napaka pri nalaganju barvne slike!")
        exit()

    # Prikažemo originalno barvno sliko
    cv.imshow("Originalna barvna slika", barvna_slika)

    # Uporabimo Sobelov filter direktno na barvni sliki
    sobel_brez_glajenja = filtriraj_sobel_smer(barvna_slika)
    cv.imshow("Sobel filter brez glajenja", sobel_brez_glajenja)

    # Pretvorimo barvno sliko v sivinsko za Gaussov filter
    siva_slika = cv.cvtColor(barvna_slika, cv.COLOR_BGR2GRAY).astype(np.float32)

    # Zgladimo sivinsko sliko z Gaussovim filtrom
    sigma_za_glajenje = 3.0
    zglajena_siva_slika = filtriraj_z_gaussovim_jedrom(siva_slika, sigma_za_glajenje)

    # Pretvorimo zglajeno sivinsko sliko nazaj v barvno
    zglajena_barvna_slika = cv.cvtColor(np.uint8(np.clip(zglajena_siva_slika, 0, 255)), cv.COLOR_GRAY2BGR)

    # Prikažemo zglajeno sliko
    cv.imshow(f"Zglajena slika (sigma={sigma_za_glajenje})", zglajena_barvna_slika)

    # Uporabimo Sobelov filter na zglajeni sliki
    sobel_zglajena = filtriraj_sobel_smer(zglajena_barvna_slika)
    cv.imshow("Sobel filter na zglajeni sliki", sobel_zglajena)


    
    cv.waitKey(0)
    cv.destroyAllWindows()


