# Télé tactile (PC + téléphone)

- **Webcam du PC**, face à la télé : position du doigt.
- **Téléphone**, au ras du bas de la télé, caméra vers le haut : contact oui / non.
- Le PC injecte les clics (souris) à l'endroit où tu touches.

## Installation (PC)
    pip install -r requirements.txt
Python 3.10 à 3.12 recommandé (MediaPipe ne supporte pas toujours les versions les plus récentes).

## Page du téléphone sur GitHub
1. Crée un dépôt, ajoute `phone/index.html` à la racine sous le nom `index.html`.
2. Settings > Pages > branche `main`, dossier `/ (root)`.
3. Ouvre l'adresse `https://<ton-compte>.github.io/<depot>/` sur le téléphone.
(Secours : la même page est servie par le PC sur `https://<IP>:8443/`.)

## Utilisation
1. PC : `python server.py` → affiche l'adresse IP.
2. Téléphone : ouvre la page, saisis l'IP, Entrée.
   Si la connexion échoue : bouton « Autoriser le certificat du PC », accepte l'avertissement, reviens et réessaie (une seule fois).
3. L'aperçu s'ouvre sur le PC (téléphone à gauche, webcam à droite). Place le téléphone ;
   `R` pivote son image, `M` la retourne en miroir.
4. `Entrée` : calibration sur la télé.
   - 12 points : touche chaque cercle et garde le doigt immobile.
   - Contact : doigt sur l'écran (6 s), puis doigt à 3-5 cm (6 s).
5. Mode tactile. `F8` pause, `F9` quitter. Prochain lancement : `L` charge la calibration enregistrée.

Options : `--cam 1`, `--no-magnet`, `--no-hover`, `--no-debug`, `--port 8443`.

## Réglages utiles (en haut de server.py)
`DWELL`, `MAGNET_RADIUS`, `SPLINE_SMOOTH`, seuils de `WritingDetector`.
