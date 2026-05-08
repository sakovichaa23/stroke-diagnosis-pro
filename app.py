import gradio as gr
import pandas as pd
from main_logic import predict_stroke, process_batch
from config import model_paths, DB_DICOM_PATH
from database import get_history_dataframe

css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#header { text-align: center !important; }
footer { display: none !important; }
.compact-df { margin-top: 10px !important; }
img { object-fit: contain !important; }
"""

with gr.Blocks(fill_width=True) as demo:
    gr.Markdown("<div id='header'><h1 style='text-align:center;'>🧠 Диагностика инсульта по КТ</h1><h3 style='text-align:center;'>Интеллектуальная система анализа медицинских изображений</h3></div>")
    
    with gr.Tabs():
        with gr.Tab("🏥 Клинический режим"):
            with gr.Column():
                model_selector = gr.Dropdown(choices=list(model_paths.keys()), value=list(model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                input_f = gr.File(label="📸 ЗАГРУЗИТЕ DICOM ФАЙЛ", file_types=[".dcm"])
                
                with gr.Row():
                    btn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    clr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                
                status_out, details_out = gr.HTML(), gr.HTML()
                
                with gr.Row(equal_height=True):
                    o_orig = gr.Image(label="📷 Исходный снимок", height=400)
                    o_res = gr.Image(label="🎯 Результат сегментации", height=400)
                
                pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ (PDF)")
                
                gr.Markdown("<br>")
                gr.Markdown("### 🔍 ФИЛЬТРАЦИЯ")
                with gr.Row():
                    filter_verdict = gr.Radio(choices=["Все", "Инсульт", "Норма"], value="Все", label="По вердикту")
                    filter_side = gr.Dropdown(choices=["Все", "Левое", "Правое", "Не выявлено"], value="Все", label="По полушарию")
                    filter_order = gr.Dropdown(choices=[
                        "По дате (сначала новые)", 
                        "По дате (сначала старые)",
                        "По площади (от большей к меньшей)", 
                        "По площади (от меньшей к большей)",
                        "По достоверности (от большей к меньшей)", 
                        "По достоверности (от меньшей к большей)"
                    ], value="По дате (сначала новые)", label="Сортировка")
                
                with gr.Row():
                    filter_btn = gr.Button("🔍 ПРИМЕНИТЬ ФИЛЬТРЫ", variant="secondary")
                    reset_filter_btn = gr.Button("🔄 СБРОСИТЬ ФИЛЬТРЫ", variant="secondary")
                
                initial_df = get_history_dataframe()
                history_table = gr.Dataframe(value=initial_df, interactive=True, elem_classes="compact-df")
                
                with gr.Row():
                    download_csv_btn = gr.DownloadButton("📥 СКАЧАТЬ CSV")
            
            def apply_filters(verdict, side, order):
                df = get_history_dataframe()
                if df.empty:
                    return df
                if verdict != "Все":
                    df = df[df['Вердикт'] == verdict]
                if side != "Все":
                    df = df[df['Полушарие'].str.contains(side, na=False)]
                
                df = df.copy()
                
                if "Площадь" in order:
                    df['Площадь_число'] = df['Площадь'].str.replace('%', '').astype(float)
                    ascending = "меньшей" in order
                    df = df.sort_values(by='Площадь_число', ascending=ascending)
                    df = df.drop(columns=['Площадь_число'])
                elif "Достоверность" in order:
                    df['Достоверность_число'] = df['Достоверность'].str.replace('%', '').astype(float)
                    ascending = "меньшей" in order
                    df = df.sort_values(by='Достоверность_число', ascending=ascending)
                    df = df.drop(columns=['Достоверность_число'])
                else:
                    ascending = "старые" in order
                    df = df.sort_values(by='Дата', ascending=ascending)
                return df
            
            def reset_filters():
                return get_history_dataframe()
            
            filter_btn.click(apply_filters, [filter_verdict, filter_side, filter_order], history_table)
            reset_filter_btn.click(reset_filters, None, history_table)
            
            btn.click(predict_stroke, [input_f, model_selector], [o_res, o_orig, status_out, details_out, history_table, pdf_file])
            clr.add([input_f, o_res, o_orig, status_out, details_out, history_table, pdf_file])

        with gr.Tab("🚀 Массовый поток"):
            with gr.Column():
                bm_sel = gr.Dropdown(choices=list(model_paths.keys()), value=list(model_paths.keys())[0], label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
                binp = gr.File(label="📸 ЗАГРУЗИТЕ ПАКЕТ DICOM ФАЙЛОВ", file_count="multiple", file_types=[".dcm"])
                with gr.Row():
                    bbtn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
                    bclr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
                bst_out, bdet_out = gr.HTML(), gr.HTML()
                gr.Markdown("### 📊 СТАТИСТИКА ПОТОКА")
                bres = gr.Image(show_label=False)
                b_pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ (BATCH)")
                
                gr.Markdown("<br>")
                gr.Markdown("### 🔍 ФИЛЬТРАЦИЯ")
                with gr.Row():
                    b_filter_verdict = gr.Radio(choices=["Все", "Инсульт", "Норма"], value="Все", label="По вердикту")
                    b_filter_side = gr.Dropdown(choices=["Все", "Левое", "Правое", "Не выявлено"], value="Все", label="По полушарию")
                    b_filter_order = gr.Dropdown(choices=[
                        "По дате (сначала новые)", 
                        "По дате (сначала старые)",
                        "По площади (от большей к меньшей)", 
                        "По площади (от меньшей к большей)",
                        "По достоверности (от большей к меньшей)", 
                        "По достоверности (от меньшей к большей)"
                    ], value="По дате (сначала новые)", label="Сортировка")
                
                with gr.Row():
                    b_filter_btn = gr.Button("🔍 ПРИМЕНИТЬ ФИЛЬТРЫ", variant="secondary")
                    b_reset_filter_btn = gr.Button("🔄 СБРОСИТЬ ФИЛЬТРЫ", variant="secondary")
                    btn_priority = gr.Button("🎯 ПРИОРИТЕТ", variant="secondary")
                
                bhist = gr.Dataframe(interactive=True, elem_classes="compact-df")
                state_full_df = gr.State()
                
                with gr.Row():
                    bdl_b = gr.DownloadButton("📥 СКАЧАТЬ CSV")
            
            def apply_batch_filters(df, verdict, side, order):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame()
                if verdict != "Все":
                    df = df[df['Вердикт'] == verdict]
                if side != "Все":
                    df = df[df['Полушарие'].str.contains(side, na=False)]
                
                df = df.copy()
                
                if "Площадь" in order:
                    df['Площадь_число'] = df['Площадь'].str.replace('%', '').astype(float)
                    ascending = "меньшей" in order
                    df = df.sort_values(by='Площадь_число', ascending=ascending)
                    df = df.drop(columns=['Площадь_число'])
                elif "Достоверность" in order:
                    df['Достоверность_число'] = df['Достоверность'].str.replace('%', '').astype(float)
                    ascending = "меньшей" in order
                    df = df.sort_values(by='Достоверность_число', ascending=ascending)
                    df = df.drop(columns=['Достоверность_число'])
                else:
                    ascending = "старые" in order
                    df = df.sort_values(by='Дата', ascending=ascending)
                return df
            
            def reset_batch_filters(df):
                return df
            
            def priority_filter(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return pd.DataFrame()
                if 'Инсульт' not in df['Вердикт'].values:
                    return pd.DataFrame(columns=df.columns)
                filtered = df[df['Вердикт'] == 'Инсульт'].copy()
                if 'Площадь' in filtered.columns:
                    filtered = filtered.sort_values(by='Площадь', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
                return filtered
            
            bbtn.click(process_batch, [binp, bm_sel], [bres, bst_out, bdet_out, bhist, b_pdf_file, state_full_df])
            bclr.add([binp, bres, bst_out, bdet_out, bhist, b_pdf_file])
            
            b_filter_btn.click(apply_batch_filters, [state_full_df, b_filter_verdict, b_filter_side, b_filter_order], bhist)
            b_reset_filter_btn.click(reset_batch_filters, [state_full_df], bhist)
            btn_priority.click(priority_filter, [state_full_df], bhist)
            bdl_b.click(lambda: DB_DICOM_PATH, None, bdl_b)

    download_csv_btn.click(lambda: "stroke_history.db", None, download_csv_btn)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=gr.themes.Soft(), css=css)
