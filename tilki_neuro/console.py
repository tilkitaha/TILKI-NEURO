"""Local research console for actual study reports, separate from toy input demo."""
import json
from pathlib import Path


def build_console(root,report_path):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from .figures import figure
    report=json.loads(Path(report_path).read_text())
    root.title('TILKI NEURO | Evidence console')
    root.geometry('1250x880');root.minsize(850,650);root.configure(bg='#0b1220')
    style=ttk.Style(root);style.theme_use('clam')
    style.configure('TNotebook',background='#0b1220',borderwidth=0)
    style.configure('TNotebook.Tab',padding=(18,10),font=('Helvetica',12))
    style.configure('Treeview',rowheight=30,font=('Helvetica',11))
    style.configure('Treeview.Heading',font=('Helvetica',11,'bold'))
    title=tk.Label(root,text='TILKI / NEURO     RESEARCH CONSOLE',bg='#0b1220',fg='#77eadb',font=('Helvetica',22,'bold'))
    title.pack(anchor='w',padx=22,pady=(16,6))
    tk.Label(root,text='Recorded public EEG • Offline analysis • No headset or device control',bg='#0b1220',fg='#ffca82',font=('Helvetica',12)).pack(anchor='w',padx=22,pady=(0,12))
    tabs=ttk.Notebook(root);tabs.pack(fill='both',expand=True,padx=16,pady=10)
    plot=ttk.Frame(tabs);tabs.add(plot,text='Evidence')
    canvas=FigureCanvasTkAgg(figure(report),master=plot);canvas.draw();canvas.get_tk_widget().pack(fill='both',expand=True)
    comparison=ttk.Frame(tabs);tabs.add(comparison,text='All models')
    columns=('mode','task','subject','model','chosen','validation','test','windows')
    tree=ttk.Treeview(comparison,columns=columns,show='headings')
    for c in columns:
        tree.heading(c,text=c.replace('_',' ').title());tree.column(c,width=125,stretch=True)
    for r in report['experiments']:
        for name,m in r['models'].items():
            tree.insert('', 'end',values=(r['mode'],r['task'],r['subject'] or '7–12',name,
                        'Yes' if name==r['selected_on_validation'] else '',f"{m['validation']['balanced_accuracy']:.1%}",
                        f"{m['test']['balanced_accuracy']:.1%}",m['test']['n']))
    scroll=ttk.Scrollbar(comparison,orient='vertical',command=tree.yview);tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right',fill='y');tree.pack(fill='both',expand=True)
    details=ttk.Frame(tabs);tabs.add(details,text='Protocol & limitations')
    text=tk.Text(details,wrap='word',bg='#101f33',fg='#dde8f7',font=('Helvetica',13),padx=20,pady=20)
    text.pack(fill='both',expand=True)
    text.insert('end','PROTOCOL SHA-256\n'+report['protocol_sha256']+'\n\n')
    text.insert('end','WHAT THESE RESULTS DO NOT ESTABLISH\n\n'+'\n\n'.join(report['limits'])+'\n\n')
    text.insert('end','GATE STATUS\n\n')
    for r in report['experiments']:
        if r['reliability']:
            gate=r['reliability']['gate']
            text.insert('end',f"{r['mode']} / {r['subject'] or 'new subjects'}: {'ENABLED' if gate['enabled'] else 'DISABLED'} — {gate['reason']}\n\n")
    text.configure(state='disabled')
    return {'tabs':tabs,'models':tree,'canvas':canvas}


def main(path):
    import tkinter as tk
    root=tk.Tk();build_console(root,path);root.mainloop()
