# 射出機產品品質預測系統（最終整合版）
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib, os, shap # type: ignore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import *
from PIL import Image, ImageTk


# === 初始化 ===
root = tk.Tk()
root.title("射出機產品品質預測系統")
img = ImageTk.PhotoImage(file='C:/Users/FCS6246/Desktop/圖片/ball/9.png')
root.iconphoto(False, img)
root.geometry("1200x800")
plt.rcParams["font.family"] = "Microsoft YaHei"

model = data_df = result_df = shap_values = None
model_feature_names, selected_columns, check_vars = [], [], {}
thresh_var = tk.DoubleVar(value=0.4)
filter_bad_only = tk.BooleanVar(value=False)
filter_by_prob = tk.BooleanVar(value=False)
prob_threshold = tk.DoubleVar(value=0.5)

# === 畫布與主畫面 ===
main_canvas = tk.Canvas(root)
main_scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
main_canvas.configure(yscrollcommand=main_scrollbar.set)
main_scrollbar.pack(side="right", fill="y")
main_canvas.pack(side="left", fill="both", expand=True)
main_frame = tk.Frame(main_canvas)
main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

# === 動態滾輪綁定 ===
def bind_scroll_to_canvas():
    root.bind_all("<MouseWheel>", lambda e: main_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    root.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))
    root.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))

def bind_scroll_to_tree():
    root.bind_all("<MouseWheel>", lambda e: tree.yview_scroll(int(-1*(e.delta/120)), "units"))
    root.bind_all("<Button-4>", lambda e: tree.yview_scroll(-1, "units"))
    root.bind_all("<Button-5>", lambda e: tree.yview_scroll(1, "units"))

bind_scroll_to_canvas()

# === 模型與資料 ===
def load_model():
    global model, model_feature_names
    path = filedialog.askopenfilename(filetypes=[("PKL", "*.pkl")])
    if path:
        try:
            model = joblib.load(path)
            model_label.config(text=f"模型已載入：{os.path.basename(path)}")
            model_feature_names = list(getattr(model, "feature_names_in_", []))
            if not model_feature_names:
                messagebox.showwarning("提醒", "模型中未儲存特徵欄位名稱")
        except Exception as e:
            messagebox.showerror("錯誤", f"載入模型失敗：{e}")

def load_excel():
    global data_df
    path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
    try:
        if path.endswith(".csv"):
            try:
                data_df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                data_df = pd.read_csv(path, encoding="big5")
        else:
            data_df = pd.read_excel(path)
        preview_data(data_df)
        show_column_selector()
    except Exception as e:
        messagebox.showerror("錯誤", f"讀取資料失敗：{e}")

# === 預覽表格 ===
def preview_data(df):
    tree.delete(*tree.get_children())
    tree["columns"], tree["show"] = list(df.columns), "headings"
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=max(100, len(col)*15))
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

# === 欄位選擇 ===
def show_column_selector():
    for widget in checkbox_frame.winfo_children(): widget.destroy()
    check_vars.clear()
    for i, col in enumerate(data_df.columns):
        var = tk.BooleanVar(value=(col in model_feature_names))
        tk.Checkbutton(checkbox_frame, text=col, variable=var, anchor='w').grid(row=i//4, column=i%4, sticky='w', padx=5, pady=2)
        check_vars[col] = var

def get_selected_columns():
    global selected_columns
    selected_columns = [col for col, var in check_vars.items() if var.get()]

# === 預測與 SHAP 分析 ===
def predict():
    global result_df
    if model is None or data_df is None:
        messagebox.showwarning("提醒", "請先載入模型與資料")
        return
    get_selected_columns()
    if not selected_columns:
        messagebox.showwarning("提醒", "請選擇要使用的特徵欄位")
        return
    if not model_feature_names:
        messagebox.showwarning("提醒", "模型無特徵欄位資訊")
        return
    if missing := [f for f in model_feature_names if f not in selected_columns]:
        messagebox.showerror("錯誤", f"缺少欄位：{missing}")
        return
    try:
        X = data_df[model_feature_names]
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba > thresh_var.get()).astype(int)
        result_df = data_df.copy()
        result_df["原始索引"] = result_df.index
        result_df["預測結果"], result_df["不良品機率"] = y_pred, y_proba
        apply_filter_and_preview()
        update_status_count(result_df)
        show_summary_chart_embedded()
    except Exception as e:
        messagebox.showerror("錯誤", f"預測失敗：{e}")

def show_shap():
    global shap_values
    if result_df is None:
        messagebox.showwarning("提醒", "請先執行預測")
        return
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(result_df[model_feature_names])
        shap.plots.beeswarm(shap_values, max_display=15)
        plt.show()
    except Exception as e:
        messagebox.showerror("錯誤", f"SHAP 分析失敗：{e}")

def show_single_shap(event):
    if result_df is None or shap_values is None:
        messagebox.showinfo("提示", "請先執行預測並產生 SHAP 分析")
        return
    selected = tree.selection()
    if not selected:
        return
    item = selected[0]
    row_values = tree.item(item)['values']
    row_dict = dict(zip(tree["columns"], row_values))
    if "原始索引" not in row_dict:
        messagebox.showerror("錯誤", "缺少原始索引欄位")
        return
    shap_idx = int(row_dict["原始索引"])
    shap_value = shap_values[shap_idx]
    window = tk.Toplevel(root)
    window.title(f"SHAP 解釋：第 {shap_idx} 筆資料")
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_value, max_display=10, show=False)
    canvas_plot = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)

def apply_filter_and_preview():
    df = result_df.copy()
    if filter_bad_only.get(): df = df[df["預測結果"] == 1]
    if filter_by_prob.get(): df = df[df["不良品機率"] >= prob_threshold.get()]
    preview_data(df)

def update_status_count(df):
    counts = df["預測結果"].value_counts()
    total, good, bad = len(df), counts.get(0, 0), counts.get(1, 0)
    status_label.config(text=f"總筆數：{total}，良品：{good}，不良品：{bad}")

def show_summary_chart_embedded():
    for widget in frame_chart.winfo_children(): widget.destroy()
    counts = result_df["預測結果"].value_counts()
    labels, values = ["良品", "不良品"], [counts.get(0, 0), counts.get(1, 0)]
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].pie(values, labels=labels, autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
    axes[1].bar(labels, values, color=["#4CAF50", "#F44336"])
    for i, v in enumerate(values):
        axes[1].text(i, v + 2, str(v), ha='center', va='bottom')
    chart = FigureCanvasTkAgg(fig, master=frame_chart)
    chart.draw()
    chart.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)

def export_result():
    if result_df is None:
        messagebox.showwarning("提醒", "尚未有預測結果")
        return
    path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
    if path:
        try:
            result_df.to_excel(path, index=False)
            messagebox.showinfo("成功", "已匯出預測結果")
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗：{e}")

def check_feature_alignment():
    if not model:
        messagebox.showwarning("提醒", "請先載入模型")
        return
    if not hasattr(model, "feature_names_in_"):
        messagebox.showwarning("提醒", "模型未儲存特徵欄位")
        return
    if data_df is None:
        messagebox.showwarning("提醒", "請先載入資料")
        return
    get_selected_columns()
    missing = [f for f in model_feature_names if f not in selected_columns]
    extra = [f for f in selected_columns if f not in model_feature_names]
    if missing:
        messagebox.showerror(" 欄位缺少", f"缺少欄位：\n{missing}")
    elif extra:
        messagebox.showwarning(" 多餘欄位", f"多出欄位：\n{extra}")
    elif selected_columns != model_feature_names:
        messagebox.showinfo(" 順序不同", "名稱一致但順序不同，將自動對齊")
    else:
        messagebox.showinfo(" 正確", "特徵欄位完全一致")

# === UI ===
frame_top = tk.Frame(main_frame)
frame_top.pack(pady=5)
for text, cmd in [
    ("載入模型", load_model), ("匯入資料", load_excel), ("開始預測", predict),
    ("匯出結果", export_result), ("SHAP 分析", show_shap), ("檢查欄位順序", check_feature_alignment)
]:
    tk.Button(frame_top, text=text, command=cmd).pack(side="left", padx=5)
model_label = tk.Label(frame_top, text="尚未載入模型")
model_label.pack(side="left", padx=10)

# === 表格區 ===
frame_middle = tk.Frame(main_frame)
frame_middle.pack(fill="both", expand=True)
frame_middle.grid_rowconfigure(0, weight=1)
frame_middle.grid_columnconfigure(0, weight=1)
canvas = tk.Canvas(frame_middle)
canvas.grid(row=0, column=0, sticky="nsew")
scroll_x = ttk.Scrollbar(frame_middle, orient="horizontal", command=canvas.xview)
scroll_x.grid(row=1, column=0, sticky="ew")
canvas.configure(xscrollcommand=scroll_x.set)
scroll_y = ttk.Scrollbar(frame_middle, orient="vertical", command=lambda *args: tree.yview(*args))
scroll_y.grid(row=0, column=1, sticky="ns")
inner_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=inner_frame, anchor="nw")
tree = ttk.Treeview(inner_frame)
tree.pack(fill="both", expand=True)
tree.configure(yscrollcommand=scroll_y.set)
tree.bind("<Enter>", lambda e: bind_scroll_to_tree())
tree.bind("<Leave>", lambda e: bind_scroll_to_canvas())
tree.bind("<<TreeviewSelect>>", show_single_shap)

# === 分類門檻設定 ===
frame_filter = tk.Frame(main_frame, bd=1, relief="groove", pady=5)
frame_filter.pack(fill="x", padx=10, pady=5)
tk.Label(frame_filter, text="分類門檻：").grid(row=0, column=0, padx=5)
tk.Scale(frame_filter, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", variable=thresh_var, length=150).grid(row=0, column=1)
tk.Entry(frame_filter, textvariable=thresh_var, width=5).grid(row=0, column=2, padx=5)
tk.Checkbutton(frame_filter, text="只顯示不良品", variable=filter_bad_only).grid(row=0, column=3, padx=5)
tk.Checkbutton(frame_filter, text="只顯示機率高於：", variable=filter_by_prob).grid(row=0, column=4)
tk.Entry(frame_filter, textvariable=prob_threshold, width=5).grid(row=0, column=5)
tk.Button(frame_filter, text="查詢", command=apply_filter_and_preview).grid(row=0, column=6, padx=10)
status_label = tk.Label(frame_filter, text="尚無預測結果")
status_label.grid(row=0, column=7, sticky="e", padx=10)

# === 圖表與欄位選擇 ===
frame_chart_and_check = tk.Frame(main_frame)
frame_chart_and_check.pack(fill="both", expand=True, padx=10)
frame_chart = tk.Frame(frame_chart_and_check, width=700, height=250)
frame_chart.pack(side="left", fill="both", expand=True)
frame_checkbox_outer = tk.LabelFrame(frame_chart_and_check, text="請選擇要作為特徵的欄位（可多選）", width=400)
frame_checkbox_outer.pack(side="right", fill="y", padx=10)
checkbox_frame = tk.Frame(frame_checkbox_outer)
checkbox_frame.pack(fill="both", expand=True, padx=5, pady=5)

# === 啟動應用 ===
root.mainloop()


# === 啟動應用 ===
# === 啟動應用 ===
# === 啟動應用 ===
# === 啟動應用 ===



# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===v

# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===v# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===# === 啟動應用 ===
# === 啟動應用 ===# === 啟動應用 ===v