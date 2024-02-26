import re
import os
import json
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import TableStyle

# from PPO.params import PARAMS_PPO

style = TableStyle([
    ('BACKGROUND', (0, 0), (3, 0), colors.blue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

    ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),

    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
])

ts = TableStyle(
    [
        ('BOX', (0, 0), (-1, -1), 2, colors.black),

        ('LINEBEFORE', (2, 1), (2, -1), 2, colors.red),
        ('LINEABOVE', (0, 2), (-1, 2), 2, colors.green),

        ('GRID', (0, 0), (-1, -1), 2, colors.black),
    ]
)


def drawMyRuler(pdf):
    pdf.drawString(100, 810, 'x100')
    pdf.drawString(200, 810, 'x200')
    pdf.drawString(300, 810, 'x300')
    pdf.drawString(400, 810, 'x400')
    pdf.drawString(500, 810, 'x500')
    pdf.drawString(10, 100, 'y100')
    pdf.drawString(10, 200, 'y200')
    pdf.drawString(10, 300, 'y300')
    pdf.drawString(10, 400, 'y400')
    pdf.drawString(10, 500, 'y500')
    pdf.drawString(10, 600, 'y600')
    pdf.drawString(10, 700, 'y700')
    pdf.drawString(10, 800, 'y800')


def dic_to_list(data):
    lista = [[v, str(k)] for v, k in list(data.items())]
    lista.insert(0, ['Parámetro', 'Valor'])
    return lista


def add_table(pdf, data, x, y):
    data = dic_to_list(data)
    table = Table(data)
    table.setStyle(style)
    table.setStyle(ts)
    table.wrapOn(pdf, 400, 100)
    table.drawOn(pdf, x, y)


def add_text(pdf, textLines, x, y):
    if isinstance(textLines, str):
        textLines = [textLines]
    text = pdf.beginText(x, y)
    text.setFont("Courier", 18)
    text.setFillColor(colors.black)
    for line in textLines:
        text.textLine(line)
    pdf.drawText(text)


def add_image(path, pdf, name, x, y, width=500, height=500):
    pdf.drawInlineImage(f'{path}/{name}', x, y, width=width,
                        height=height, preserveAspectRatio=True)


def create_report(path: str, method: str, state_params: dict, env_params: dict, 
                  title: str = None, subtitle: str ='', file_name: str = None):
    '''
    Genera un documento pdf con el reporte de entrenamiento
    de distintos algoritmos.

    path : str
        Dirección donde será guardado el documento ('Aqui-se-guardara/').
    states_params: dict
        ...
    env_params: dict
        ...
    title : str
        Título que estará en el encabezado de la primera hoja.
    subtitle : str
        Subtítulo que estará en el encabezado de la primera hoja.
    file_name : str
        Nombre del archivo que tendrá el documento ('reporte.pdf').
    '''
    if not isinstance(file_name, str):
        file_name = 'reporte.pdf'
    if not isinstance(title, str):
        title = f'Reporte de entrenamiento {method}'
    file_name = f'{path}/{file_name}'

    with open(f"{path}/config.json", 'r') as json_file:
        data = json.load(json_file)

    state_params = {re.sub(r'\$', '', k): u'\u00B1'+v for k,
              v in state_params.items()}

    # Parámetros de red neuronal
    ac_kwargs = data["ac_kwargs"]

    ac_kwargs.update({"gamma": data["gamma"], "rho": data["polyak"], "replay_size": data["replay_size"], "actor_lr": data["pi_lr"], "critic_lr": data["q_lr"]})

    # Parámetros de entrenamiento
    episodes = data["steps_per_epoch"] * data["epochs"] // data["max_ep_len"]
    train_params = {"batch_size": data["batch_size"], "episodes": episodes, "num_test_episodes": data["num_test_episodes"], "update_every": data["update_every"], "update_after": data["update_after"]}
    if method == 'td3':
        train_params["policy_delay"] = data["policy_delay"]

    # Parémetros de ruido
    noise_params = {"act_noise": data["act_noise"]}
    if method == 'td3':
        noise_params["target_noise"] = data["target_noise"]
        noise_params["noise_clip"] = data["noise_clip"]


    pdf = canvas.Canvas(file_name)
    pdf.drawCentredString(300, 800, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 26)
    pdf.drawCentredString(290, 760, subtitle)

    add_text(pdf, ['Espacio de', 'observación'], 100, 750)
    add_table(pdf, state_params, 100, 480)

    add_text(pdf, ['Parámetros del', 'ambiente'], 100, 450)
    add_table(pdf, env_params, 100, 250)

    add_text(pdf, ['Parámetros de', 'optimiazación de red'], 100, 220)
    add_table(pdf, data["ac_kwargs"], 100, 50)

    add_text(pdf, ['Parámetros de', 'entrenamiento DDPG'], 350, 750)
    add_table(pdf, train_params, 350, 590)

    add_text(pdf, ['Parámetros de', 'ruido'], 350, 550)
    add_table(pdf, noise_params, 350, 360)

    if os.path.exists(f'{path}/train_performance.png'):
        pdf.showPage()
        add_text(pdf, ['Rendimiento de entrenamiento'], 30, 750)
        add_image(path, pdf, 'train_performance.png', 100, 400, 350, 350)
    
    pdf.showPage()
    add_text(pdf, ['Simulaciones (estados)'], 30, 390)
    add_image(path, pdf, 'state_rollouts.png', 30, -10, 500, 500)

    pdf.showPage()
    add_text(pdf, ['Simulaciones (acciones)'], 30, 750)
    add_image(path, pdf, 'action_rollouts.png', 30, 350, 500, 500)
    add_text(pdf, ['Simulaciones (penalizaciones)'], 30, 400)
    add_image(path, pdf, 'score_rollouts.png', 30, 0, 500, 500)

    pdf.save()