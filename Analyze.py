import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import io
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
import urllib.request
import os

# 1.1 ระบบดาวน์โหลดและติดตั้งฟอนต์อัตโนมัติ
def setup_thai_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    font_name = "Sarabun-Regular.ttf"
    try:
        if not os.path.exists(font_name):
            urllib.request.urlretrieve(font_url, font_name)
        fm.fontManager.addfont(font_name)
        # ดึงชื่อฟอนต์ที่แท้จริงจากไฟล์
        prop = fm.FontProperties(fname=font_name)
        return prop
    except Exception as e:
        st.error(f"Font Load Error: {e}")
        return None

# เรียกใช้งานและเก็บตัวแปรไว้ใช้ทั่วโปรแกรม
thai_font_prop = setup_thai_font()

if thai_font_prop:
    # ตั้งค่า Default ให้ Matplotlib ใช้ฟอนต์ที่โหลดมา
    matplotlib.rcParams['font.family'] = thai_font_prop.get_name()
else:
    # กรณีโหลดไม่สำเร็จจริงๆ ให้ใช้ฟอนต์ที่มีในระบบ
    matplotlib.rcParams['font.family'] = 'sans-serif'

#=============================================================================

def apply_calc_logic(df):
    """รักษา Logic การคำนวณเดิม: สร้าง flag สำหรับการวิเคราะห์คุณภาพ"""
    pm_cols = [c for c in df.columns if 'PM' in c]
    pc_cols = [c for c in df.columns if 'PC' in c]
    piera_cols = pm_cols + pc_cols
    dht_cols = ['humidity', 'temperature']
    
    df['has_dht'] = df[[c for c in dht_cols if c in df.columns]].notnull().all(axis=1) if any(c in df.columns for c in dht_cols) else False
    df['has_piera'] = df[[c for c in piera_cols if c in df.columns]].notnull().any(axis=1) if piera_cols else False
    df['has_both'] = df['has_dht'] & df['has_piera']
    return df

def process_file(file):
    """สร้าง Dataset จากไฟล์เพียงอย่างเดียว โดยอ่านข้อมูลทุกวันที่มี"""
    try:
        df = pd.read_csv(file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y-%H-%M-%S')
        df = df.sort_values('datetime')
        
        points_data = {}
        if 'point_id' in df.columns:
            for pid in df['point_id'].unique():
                pdf = df[df['point_id'] == pid].copy()
                sensor_cols = [c for c in pdf.columns if c not in ['datetime', 'point_id']]
                pdf = pdf.dropna(subset=sensor_cols, how='all')
                points_data[str(pid)] = apply_calc_logic(pdf)
        else:
            suffixes = set([c.split('_')[-1] for c in df.columns if '_P' in c])
            for s in sorted(suffixes):
                p_cols = [c for c in df.columns if c.endswith(f'_{s}')]
                pdf = df[['datetime'] + p_cols].copy()
                pdf.columns = ['datetime'] + [c.replace(f'_{s}', '') for c in p_cols]
                sensor_cols = [c for c in pdf.columns if c != 'datetime']
                pdf = pdf.dropna(subset=sensor_cols, how='all')
                points_data[s] = apply_calc_logic(pdf)

        if not points_data:
            return None, "ไม่พบข้อมูล Point ที่มีข้อมูลในไฟล์"

        return {"raw_df": df, "points_data": points_data}, "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def calculate_stats(df):
    """คำนวณสถิติคุณภาพข้อมูล (Logic เดิม)"""
    SEC_PER_DAY = 86400
    MIN_PER_DAY = 1440
    def get_metrics(mask):
        count = mask.sum()
        avg_min = count / MIN_PER_DAY
        pct = (count / SEC_PER_DAY) * 100
        missing = SEC_PER_DAY - count
        return [count, avg_min, pct, max(0, missing)]
    
    overall = get_metrics(pd.Series([True] * len(df)))
    dht = get_metrics(df['has_dht'])
    piera = get_metrics(df['has_piera'])
    both = get_metrics(df['has_both'])
    
    return pd.DataFrame({
        'Metric': ['Total Seconds', 'Avg per Minute', '% of Day', 'Missing Seconds'],
        'Overall': overall, 'DHT22': dht, 'Piera': piera, 'Both Sensors': both
    }).set_index('Metric')

# --- 2. ฟังก์ชันสำหรับระบบใหม่ (Summary Report Only) ---

def calculate_continuity_v3(df, start_ts, end_ts, expected_sec):
    """คำนวณสถิติแยกประเภท พร้อม Overall และ Outlier"""
    mask = (df['datetime'] >= start_ts) & (df['datetime'] <= end_ts)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        return None, None

    # 1. Overall: จำนวนวินาทีที่มีแถวข้อมูล (เครื่องทำงาน)
    overall_count = len(df_filtered)
    
    # 2. แยกประเภทเซนเซอร์ (วินาทีที่มีข้อมูล)
    dht_count = df_filtered['has_dht'].sum()
    piera_count = df_filtered['has_piera'].sum()
    both_count = df_filtered['has_both'].sum()

    # 3. Outlier (> 6600)
    outlier_count = 0
    if 'PM2_5' in df_filtered.columns:
        outlier_count = len(df_filtered[df_filtered['PM2_5'] > 6600])

    stats = {
        'overall': {'sec': overall_count, 'pct': (overall_count / expected_sec) * 100},
        'dht': {'sec': dht_count, 'pct': (dht_count / expected_sec) * 100},
        'piera': {'sec': piera_count, 'pct': (piera_count / expected_sec) * 100},
        'both': {'sec': both_count, 'pct': (both_count / expected_sec) * 100},
        'outlier': outlier_count
    }
    
    # 4. Resample 5min สำหรับกราฟ
    resampled_graph = df_filtered.set_index("datetime").resample("5min").count()
    
    return stats, resampled_graph

def generate_summary_report(points_data, target_date_str, report_type, export_format, manual_text=None):
    # ดึงวันที่จาก Dataset
    first_df = list(points_data.values())[0]
    base_date = first_df['datetime'].dt.normalize().iloc[0]
    
    start_dt = base_date
    if report_type == "12 ชั่วโมง":
        end_dt = base_date + timedelta(hours=11, minutes=59, seconds=59)
        total_sec = 43200
    else:
        end_dt = base_date + timedelta(hours=23, minutes=59, seconds=59)
        total_sec = 86400

    # --- ส่วนตรรกะข้อความ ---
    # ถ้าไม่มีการส่ง manual_text มา (คือการกดคำนวณครั้งแรก) ให้สร้างข้อความอัตโนมัติ
    if manual_text is None:
        report_text = f"รายงานผลวันที่ {base_date.strftime('%d/%m/%Y')}\n"
        report_text += f"หลังจากเก็บข้อมูลมา {report_type}\n"
        report_text += f"ข้อมูลที่เข้าของแต่ละ Point จากทั้งหมด {total_sec} วินาที\n\n"

        point_details = ""
        for pid, df in points_data.items():
            stats, _ = calculate_continuity_v3(df, start_dt, end_dt, total_sec)
            if stats:
                point_details += f"P{pid}:\n"
                point_details += f"ข้อมูลเข้า (Overall): {stats['overall']['sec']} วินาที ({stats['overall']['pct']:.2f}%)\n"
                point_details += f"- DHT22: {stats['dht']['sec']} วินาที ({stats['dht']['pct']:.2f}%)\n"
                point_details += f"- Piera: {stats['piera']['sec']} วินาที ({stats['piera']['pct']:.2f}%)\n"
                point_details += f"- ทั้งคู่ (Both): {stats['both']['sec']} วินาที ({stats['both']['pct']:.2f}%)\n"
                point_details += f"Outlier ที่พบ: {stats['outlier']} ค่า\n\n"
        
        final_display_text = report_text + point_details
    else:
        # ถ้ามี manual_text (ส่งมาจาก text_area) ให้ใช้ข้อความนั้นวาดลงรูปเลย
        final_display_text = manual_text

    # --- ส่วนการสร้างรูปภาพ ---
    plt.close('all')
    fig = plt.figure(figsize=(10, 14))
    ax_text = fig.add_axes([0.1, 0.40, 0.8, 0.55]) 
    ax_text.axis('off')
    ax_graph = fig.add_axes([0.1, 0.1, 0.8, 0.25])

    # วาดข้อความ (ใช้ final_display_text)
    ax_text.text(0, 1, final_display_text, 
                 fontproperties=thai_font_prop, 
                 fontsize=11, 
                 verticalalignment='top', 
                 linespacing=1.4)

    # วาดกราฟ
    for pid, df in points_data.items():
        _, res_graph = calculate_continuity_v3(df, start_dt, end_dt, total_sec)
        if res_graph is not None:
            ax_graph.plot(res_graph.index, res_graph['has_both'], label=f'Point {pid}', linewidth=1.5)

    ax_graph.set_title("Data Continuity Trend (Resampled 5 min)")
    ax_graph.set_ylabel("Counts per 5 min")
    ax_graph.legend(loc='upper right')
    ax_graph.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)

    # Export Logic
    files = {}
    if export_format != "None": # เพิ่มเงื่อนไขกรณีอยากได้แค่ Text ไม่เอาไฟล์
        if export_format in ["PNG", "ทั้งสองแบบ"]:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            files['png'] = buf.getvalue()
        if export_format in ["PDF", "ทั้งสองแบบ"]:
            buf = io.BytesIO()
            with PdfPages(buf) as pdf: pdf.savefig(fig, bbox_inches='tight')
            files['pdf'] = buf.getvalue()
        
    return final_display_text, files
# --- 3. UI State Management ---
if 'analysis_sets' not in st.session_state: st.session_state.analysis_sets = {}
if 'selected_set_id' not in st.session_state: st.session_state.selected_set_id = None
if 'show_summary' not in st.session_state: st.session_state.show_summary = False

st.set_page_config(page_title="Sensor Quality Analysis Dashboard", layout="wide")




# --- 4. Sidebar UI (Logic เดิม) ---
with st.sidebar:
    st.title("🛠 Management")
    with st.expander("🆕 Create Analysis Set", expanded=not st.session_state.analysis_sets):
        up_file = st.file_uploader("Upload Combined CSV File", type=['csv'])
        
        if st.button("Confirm & Create", use_container_width=True):
            if up_file:
                result, msg = process_file(up_file)
                if result:
                    # ใช้ชื่อไฟล์เป็นหลัก หากซ้ำให้ต่อ Timestamp
                    base_name = up_file.name
                    set_id = base_name
                    if set_id in st.session_state.analysis_sets:
                        set_id = f"{base_name}_{datetime.now().strftime('%H%M%S')}"
                    
                    st.session_state.analysis_sets[set_id] = {
                        'date': base_name, # ใช้ชื่อไฟล์แสดงผลในตำแหน่ง date เดิม
                        'file_name': up_file.name,
                        **result 
                    }
                    st.session_state.selected_set_id = set_id
                    st.success(f"สร้าง Dataset: {set_id} สำเร็จ!")
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("กรุณาอัปโหลดไฟล์")

    st.markdown("---")
    st.subheader("📂 Saved Analysis Sets")
    for sid, sdata in list(st.session_state.analysis_sets.items()):
        col_select, col_del = st.columns([4, 1])
        # แสดงชื่อ dataset (ชื่อไฟล์) และจำนวน Point
        if col_select.button(f"📄 {sid} ({len(sdata['points_data'])} P)", key=f"sel_{sid}", use_container_width=True):
            st.session_state.selected_set_id = sid
        if col_del.button("🗑️", key=f"del_{sid}"):
            del st.session_state.analysis_sets[sid]
            if st.session_state.selected_set_id == sid: st.session_state.selected_set_id = None
            st.rerun()

# --- 5. Main Dashboard Area ---
if st.session_state.selected_set_id:
    curr_set = st.session_state.analysis_sets[st.session_state.selected_set_id]
    points_dict = curr_set['points_data']
    target_date = curr_set['date']
    
    # 🎯 Header พร้อมปุ่ม "สรุปผล"
    h1, h2 = st.columns([8, 2])
    with h1:
        st.title(f"📊 Analysis: {target_date}")
    with h2:
        st.write(" ")
        if st.button("📊 Summary Report", use_container_width=True, type="primary"):
            st.session_state.show_summary = not st.session_state.show_summary

    # 🎯 ระบบ Summary Report Overlay
    if st.session_state.show_summary:
        with st.container(border=True):
            st.subheader("📝 สร้างรายงานสรุปผล (Summary Report)")
            c1, c2 = st.columns(2)
            with c1:
                sel_type = st.radio("เลือกช่วงเวลา", ["12 ชั่วโมง", "24 ชั่วโมง"], horizontal=True)
            with c2:
                sel_format = st.selectbox("รูปแบบ Export", ["PNG", "PDF", "ทั้งสองแบบ"])
            
            # 1. สร้างข้อความเริ่มต้นใส่ session_state (ถ้ายังไม่มี)
            if st.button("✅ 1. คำนวณข้อมูลเริ่มต้น", use_container_width=True):
                # ดึงแค่ข้อความกับ Graph data (ยังไม่ทำไฟล์ PNG)
                report_txt, _ = generate_summary_report(points_dict, target_date, sel_type, "None")
                st.session_state.editable_report = report_txt

            # 2. ช่อง Text Area สำหรับแก้ไข (จะจำค่าที่พิมพ์ไว้)
            if 'editable_report' in st.session_state:
                final_text = st.text_area("แก้ไขสรุปรายงานก่อนดาวน์โหลด", 
                                          value=st.session_state.editable_report, 
                                          height=300, 
                                          key="report_editor")
                
                # 3. ปุ่มสำหรับ "บันทึกและสร้างไฟล์" จากข้อความที่แก้ไขแล้ว
                if st.button("🖼️ 2. ยืนยันข้อความนี้และสร้างไฟล์ดาวน์โหลด", use_container_width=True, type="primary"):
                    # ส่ง final_text กลับไปวาดลงบนรูปภาพใหม่
                    # เราจะเพิ่ม parameter 'manual_text' ในฟังก์ชันเดิม
                    updated_txt, report_files = generate_summary_report(
                        points_dict, target_date, sel_type, sel_format, manual_text=final_text
                    )
                    st.session_state.final_files = report_files
                    st.success("สร้างไฟล์พร้อมดาวน์โหลดแล้ว!")

                # 4. แสดงปุ่มดาวน์โหลด (ถ้าไฟล์ถูกสร้างแล้ว)
                if "final_files" in st.session_state:
                st.write("---")
                files = st.session_state.final_files
                
                # สร้าง 3 columns เพื่อดันปุ่มมาไว้ตรงกลาง
                col_l, col_mid, col_r = st.columns([1, 2, 1])
                
                with col_mid:
                    if 'png' in files:
                        st.download_button("📥 Download PNG", files['png'], "report.png", "image/png", use_container_width=True)
                    if 'pdf' in files:
                        st.download_button("📄 Download PDF", files['pdf'], "report.pdf", "application/pdf", use_container_width=True)

    # --- UI เดิม (ห้ามแก้) ---
    tabs = st.tabs(["📋 Executive Summary", "🔍 Gap Analysis", "📈 Trends & Charts"])
    
    with tabs[0]:
        all_stats = {}
        all_outliers = {}
        for pid, pdf in points_dict.items():
            all_stats[pid] = calculate_stats(pdf)
            all_outliers[pid] = pdf[pdf['PM2_5'] > 6600] if 'PM2_5' in pdf.columns else pd.DataFrame()

        m_cols = st.columns(len(points_dict) + 2)
        m_cols[0].metric("Date", str(target_date))
        for idx, pid in enumerate(points_dict.keys()):
            pct = all_stats[pid].loc['% of Day', 'Both Sensors']
            m_cols[idx+1].metric(f"Point {pid} (%)", f"{pct:.2f}%")
        
        with m_cols[-1]:
            st.write("**Outliers Found**")
            o_sub_cols = st.columns(len(points_dict))
            for idx, pid in enumerate(points_dict.keys()):
                count = len(all_outliers[pid])
                o_sub_cols[idx].markdown(f"### {count}\n<small>P{pid}</small>", unsafe_allow_html=True)

        st.divider()
        for pid, pdf in points_dict.items():
            st.subheader(f"📍 Point {pid}")
            res = all_stats[pid]
            st.dataframe(res.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1), use_container_width=True)
            csv_buf = io.StringIO()
            res.to_csv(csv_buf)
            st.download_button(f"📥 Download Report Point {pid}", csv_buf.getvalue(), f"Analyze_{pid}_{target_date}.csv", key=f"dl_{pid}")

    with tabs[1]:
        gap_threshold = st.number_input("Gap Threshold (วินาที)", min_value=1, value=5)
        g_cols = st.columns(len(points_dict))
        for idx, (pid, pdf) in enumerate(points_dict.items()):
            with g_cols[idx]:
                st.subheader(f"🔍 Point {pid}")
                df_gap = pdf.copy()
                df_gap['diff'] = df_gap['datetime'].diff().dt.total_seconds()
                gaps = df_gap[df_gap['diff'] > gap_threshold].sort_values('diff', ascending=False)
                st.metric(f"Total Gaps (P{pid})", len(gaps))
                st.table(gaps[['datetime', 'diff']].head(5).rename(columns={'diff': 'Duration (s)'}))
                with st.expander(f"ดู Gap ทั้งหมดของ {pid}"):
                    st.dataframe(gaps[['datetime', 'diff']].rename(columns={'diff': 'Duration (s)'}), use_container_width=True)
                st.divider()
                out_df = all_outliers[pid]
                st.write(f"🚨 **Outliers P{pid} (PM2.5 > 6600)**")
                if not out_df.empty:
                    st.dataframe(out_df[['datetime', 'PM2_5']].rename(columns={'PM2_5': 'PM2.5 (µg/m³)'}), use_container_width=True)
                else: st.success(f"Point {pid} ไม่พบ Outlier")

    with tabs[2]:
        st.subheader("📈 Data Continuity")
        interval = st.selectbox("เลือกช่วงเวลา Resample", ["1min", "3min", "5min"], index=0)
        fig_trend = go.Figure()
        for pid, pdf in points_dict.items():
            df_min = pdf.set_index('datetime').resample(interval).count()
            fig_trend.add_trace(go.Scatter(x=df_min.index, y=df_min['has_both'], name=f"Point {pid}", mode='lines'))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.divider()
        st.subheader("📊 Sensor Viewer")
        selected_points = st.multiselect("เลือก Point", list(points_dict.keys()), default=[list(points_dict.keys())[0]])
        metrics = st.multiselect("เลือกค่าที่ต้องการ", ["PM2_5", "temperature", "humidity"], default=["PM2_5"])
        show_outlier = st.checkbox("แสดง Outlier (PM2.5 > 6600)", value=True)
        fig_val = go.Figure()
        y_values_for_scale = []
        for pid in selected_points:
            sel_df = points_dict[pid]
            for m in metrics:
                if m in sel_df.columns:
                    fig_val.add_trace(go.Scatter(x=sel_df['datetime'], y=sel_df[m], name=f"{m} (P{pid})", mode='lines'))
                    if m == "PM2_5":
                        y_vals = sel_df[m].dropna() if show_outlier else sel_df[sel_df[m] <= 6600][m].dropna()
                        y_values_for_scale.extend(y_vals.values)
        if y_values_for_scale:
            ymin, ymax = min(y_values_for_scale), max(y_values_for_scale)
            padding = (ymax - ymin) * 0.1 if ymax != ymin else 10
            fig_val.update_yaxes(range=[max(0, ymin - padding), ymax + padding])
        st.plotly_chart(fig_val, use_container_width=True)
else:
    st.title("👈 โปรดอัปโหลดหรือเลือกชุดข้อมูล")

    st.info("ระบบจะแยก Overall และ Gap Analysis ของแต่ละ Point ให้โดยอัตโนมัติ")




