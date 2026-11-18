"""Interface de démonstration pour le hackathon.  Lancer :  python src/app.py
puis ouvrir http://127.0.0.1:7860 dans le navigateur."""
import gradio as gr

import config as C
from predict import analyse_eye, risk_level

DISCLAIMER = ("⚠️ Outil d'aide au dépistage. Il ne pose pas de diagnostic : "
              "la décision reste celle de l'ophtalmologue.")


def run_single(eye, axial, anterior, posterior, pachy):
    if not all([axial, anterior, posterior, pachy]):
        raise gr.Error("Merci de fournir les 4 cartes.")
    prob, overlays = analyse_eye([axial, anterior, posterior, pachy], eye)
    txt = f"## Score de risque : {prob:.1%} — {risk_level(prob)}\n\n{DISCLAIMER}"
    return txt, [(o, m) for o, m in zip(overlays, C.MAPS)]


def run_compare(eye, a1, b1, c1, d1, a2, b2, c2, d2):
    if not all([a1, b1, c1, d1, a2, b2, c2, d2]):
        raise gr.Error("Merci de fournir les 4 cartes pour les DEUX examens.")
    p1, _ = analyse_eye([a1, b1, c1, d1], eye)
    p2, ov2 = analyse_eye([a2, b2, c2, d2], eye)
    delta = p2 - p1
    if delta > 0.15:
        verdict = "📈 Le score augmente nettement : possible progression, à contrôler."
    elif delta < -0.15:
        verdict = "📉 Le score diminue."
    else:
        verdict = "➖ Score stable entre les deux examens."
    txt = (f"**Examen 1 :** {p1:.1%}  |  **Examen 2 :** {p2:.1%}  |  **Δ = {delta:+.1%}**\n\n"
           f"### {verdict}\n\n{DISCLAIMER}")
    return txt, [(o, m) for o, m in zip(ov2, C.MAPS)]


def map_inputs(suffix=""):
    return [gr.Image(type="filepath", label=f"{m}{suffix}") for m in C.MAPS]


with gr.Blocks(title="Qarniya AI") as demo:
    gr.Markdown("# Qarniya AI — dépistage précoce du kératocône\nÉquipe NeuroLens · Hackathon Nexus HealthTech 2026")
    with gr.Tab("Analyse d'un œil"):
        eye = gr.Radio(["OD", "OS"], value="OD", label="Œil")
        with gr.Row():
            imgs = map_inputs()
        btn = gr.Button("Analyser", variant="primary")
        out_txt = gr.Markdown()
        out_gal = gr.Gallery(label="Zones suspectes (Grad-CAM)", columns=4, height=360)
        btn.click(run_single, [eye] + imgs, [out_txt, out_gal])

    with gr.Tab("Suivi de progression"):
        eye2 = gr.Radio(["OD", "OS"], value="OD", label="Œil")
        gr.Markdown("**Examen 1 (ancien)**")
        with gr.Row():
            e1 = map_inputs(" (1)")
        gr.Markdown("**Examen 2 (récent)**")
        with gr.Row():
            e2 = map_inputs(" (2)")
        btn2 = gr.Button("Comparer", variant="primary")
        out_txt2 = gr.Markdown()
        out_gal2 = gr.Gallery(label="Zones suspectes — examen récent", columns=4, height=360)
        btn2.click(run_compare, [eye2] + e1 + e2, [out_txt2, out_gal2])

if __name__ == "__main__":
    import sys
    # Sur Google Colab, on crée un lien public temporaire (*.gradio.live) pour ouvrir la démo
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in __import__("os").environ
    demo.launch(share=in_colab)
