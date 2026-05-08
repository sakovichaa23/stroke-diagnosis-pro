import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from config import FONT_PATH
from huggingface_hub import InferenceClient
import re

HF_TOKEN = os.environ.get("HF_TOKEN", None)

def get_ai_recommendations(verdict, hu_info, side_ru, area_percent):
    if not HF_TOKEN:
        if verdict == "НОРМА":
            return "Плановое наблюдение у невролога. Контроль артериального давления. МРТ при появлении симптомов."
        else:
            return f"Госпитализация в неврологическое отделение. Контроль АД каждый час. КТ-ангиография. Повторное КТ через 24 часа. Отмена антиагрегантов."
    
    try:
        client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
        
        if verdict == "НОРМА":
            prompt = f"""Напиши 4-5 коротких предложений (до 150 слов) рекомендаций для пациента с нормальной КТ. Без вступлений. Только конкретные действия. На русском."""
        else:
            prompt = f"""Напиши 4-5 коротких предложений (до 150 слов) конкретных медицинских действий. Данные: {verdict}, {hu_info}, {side_ru}, площадь {area_percent}. Пиши действия: 'Госпитализировать...', 'Назначить...', 'Провести...', 'Контролировать...'. Без вступлений. На русском."""

        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages=messages, max_tokens=250)
        text = response.choices[0].message.content.strip()
        
        text = re.sub(r'^[\s•\-*\d+\.]+', '', text)
        text = re.sub(r'Я рекомендую|Рекомендую|я рекомендую|Учитывая|Кроме того|Для этого|Необходимо|Следует', '', text)
        text = ' '.join(text.split())
        
        if not text.endswith('.'):
            text += '.'
        
        if len(text) > 500:
            text = text[:500]
            last_space = text.rfind(' ')
            if last_space > 450:
                text = text[:last_space]
            if not text.endswith('.'):
                text += '.'
        
        return text
    except Exception as e:
        if verdict == "НОРМА":
            return "Плановое наблюдение у невролога. Контроль артериального давления. МРТ при появлении симптомов."
        else:
            return f"Госпитализация в неврологическое отделение. Контроль АД каждый час. КТ-ангиография. Повторное КТ через 24 часа. Отмена антиагрегантов."

def generate_report_universal(results_list, output_name="Diagnosis_Report.pdf", is_batch=False):
    pdf = FPDF()
    has_font = os.path.exists(FONT_PATH)
    if has_font:
        try:
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_PATH)
        except:
            has_font = False
    disclaimer = "ВНИМАНИЕ: Данный отчет сформирован системой ИИ. Он носит справочный характер и не является диагнозом."

    for idx, item in enumerate(results_list):
        orig = item['orig_img']
        res = item['res_img']
        info = item['info']
        
        filename = info['filename'].replace('.dcm', '')
        
        pdf.add_page()

        if has_font:
            pdf.set_font("DejaVu", "B", 20)
            pdf.cell(0, 15, "МЕДИЦИНСКИЙ ОТЧЕТ АНАЛИЗА КТ", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 10)
            if is_batch:
                pdf.cell(0, 8, f"Файл: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 8, f"ID Пациента: {info['p_id']} | Файл: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Дата: {info['date']} | Время: {info['time']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            pdf.set_font("DejaVu", "B", 14)
            pdf.cell(0, 10, "1. РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 12)

            if info['verdict_ru'] == "ИНСУЛЬТ":
                pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, f"Заключение: {info['verdict_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, f"Плотность очага: {info.get('hu', 'Н/Д')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Локализация: {info['side_ru']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Площадь поражения: {info['area']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Скорость анализа: {info['speed']} мс", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Уверенность: {info['conf']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Модель: {info['model']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.ln(8)
            
            area_val = info['area'].replace('%', '')
            recommendations = get_ai_recommendations(info['verdict_ru'], info.get('hu', 'Н/Д'), info['side_ru'], area_val)
            
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(0, 0, 255)
            pdf.cell(0, 10, "2. КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("DejaVu", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6, recommendations, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.ln(5)

        temp_o = f"o_tmp_{idx}.jpg"
        temp_r = f"r_tmp_{idx}.jpg"
        cv2.imwrite(temp_o, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_r, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        
        pdf.image(temp_o, x=15, y=pdf.get_y(), w=85)
        pdf.image(temp_r, x=110, y=pdf.get_y(), w=85)
        os.remove(temp_o)
        os.remove(temp_r)

        pdf.set_y(pdf.get_y() + 5)
        if has_font:
            pdf.set_font("DejaVu", "", 7)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 4, disclaimer, align="C")
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(0, 4, "WARNING: AI Report. This is for informational purposes only.", align="C")

    pdf.output(output_name)
    return output_name

def create_analytics(df):
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    verdict_counts = df['Вердикт'].value_counts()
    colors_v = ['#4CAF50' if x == 'Норма' else '#D32F2F' for x in verdict_counts.index]
    axes[0].pie(verdict_counts, labels=verdict_counts.index, autopct='%1.1f%%', colors=colors_v)
    axes[0].set_title("Статус: Норма / Инсульт", fontsize=12, fontweight='bold')

    stroke_df = df[df['Вердикт'] == 'Инсульт']
    if not stroke_df.empty:
        areas = stroke_df['Площадь'].str.replace('%', '').astype(float)
        n, bins, patches = axes[1].hist(areas, bins=10, edgecolor='black')
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 0.5:
                patch.set_facecolor('#FFEB3B')
            elif bin_center < 2.0:
                patch.set_facecolor('#FB8C00')
            else:
                patch.set_facecolor('#D32F2F')
        axes[1].set_xlabel("Площадь поражения (%)")
    else:
        axes[1].text(0.5, 0.5, 'Данных нет', ha='center')
    axes[1].set_title("Тяжесть: Площадь поражения (%)", fontsize=12, fontweight='bold')

    if not stroke_df.empty:
        side_counts = stroke_df['Полушарие'].value_counts()
        short_labels = []
        for label in side_counts.index:
            side_char = "Л" if "Левое" in label else "П" if "Правое" in label else "?"
            if "ПМА" in label:
                short_labels.append(f"{side_char} ПМА")
            elif "СМА" in label:
                short_labels.append(f"{side_char} СМА")
            elif "ЗМА" in label:
                short_labels.append(f"{side_char} ЗМА")
            else:
                short_labels.append(f"{side_char}")
        axes[2].pie(side_counts, labels=short_labels, autopct='%1.1f%%',
                    colors=['#2196F3', '#00BCD4', '#4CAF50'])
    else:
        axes[2].text(0.5, 0.5, 'Инсультов нет', ha='center')
    axes[2].set_title("Локализация", fontsize=12, fontweight='bold')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    return Image.open(buf)
