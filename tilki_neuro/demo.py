"""Local desktop demo: actual model inference on clearly synthetic input."""
import numpy as np
from .data import synthetic
from .decoder import Decoder
from .evaluate import split, select_threshold
from .communication import Board


def main():
    import tkinter as tk
    data = synthetic()
    tr, va, te = split(data)
    model = Decoder(data.sfreq).fit(data.x[tr], data.y[tr])
    _, p, good = model.predict(data.x[va], 0)
    threshold, _ = select_threshold(data.y[va], p, good, 2)
    rng = np.random.default_rng(17)
    board = Board()
    root = tk.Tk()
    root.title('TILKI NEURO | Research console')
    root.geometry('940x700')
    root.configure(bg='#0b1220')
    def label(text, size=16, color='#e7efff'):
        w = tk.Label(root, text=text, bg='#0b1220', fg=color, font=('Helvetica', size), wraplength=870)
        w.pack(pady=8)
        return w
    label('TILKI / NEURO', 30)
    label('SYNTHETIC SIGNALS • No headset connected • Research prototype', 14, '#ffc36c')
    label('Choose a simulated input. Left moves; right selects. Return to idle between commands.', 15)
    status = label('Ready. Select twice, separated by idle, to confirm.', 16, '#77eadb')
    canvas = tk.Canvas(root, height=130, bg='#101e30', highlightthickness=0)
    canvas.pack(fill='x', padx=24, pady=8)
    selection = label('')
    pending = label('', 14, '#ffc36c')
    history = label('No confirmed phrases', 18)
    controls = tk.Frame(root, bg='#0b1220')
    controls.pack(pady=10)
    def refresh():
        selection.config(text='   |   '.join(('['+p+']') if i == board.index else p for i,p in enumerate(board.phrases)))
        pending.config(text='Pending confirmation: '+(board.pending or 'none'))
        history.config(text=' · '.join(board.history[-8:]) or 'No confirmed phrases')
    def feed(kind):
        label_id = kind if kind < 3 else 0
        idx = int(rng.choice(te[data.y[te] == label_id]))
        window = data.x[idx:idx+1].copy()
        if kind == 3:
            window[:, 0, :] = 0
        pred, prob, ok = model.predict(window, threshold)
        outcome = board.step(int(pred[0]))
        status.config(text=f'{outcome}\nModel score {prob.max():.1%} · threshold {threshold:.1%} · quality {"pass" if ok[0] else "rejected"}')
        canvas.delete('all')
        width = max(canvas.winfo_width(), 400)
        yy = window[0, 0, ::2]
        coords = [v for i,y in enumerate(yy) for v in (i*width/(len(yy)-1), 65-float(y)*1.5)]
        canvas.create_line(*coords, fill='#77eadb', width=2)
        refresh()
    for i, text in enumerate(('Idle', 'Left imagery', 'Right imagery', 'Disconnected channel')):
        tk.Button(controls, text=text, command=lambda k=i: feed(k), font=('Helvetica', 14), padx=10, pady=10).grid(row=0,column=i,padx=4)
    def cancel():
        board.cancel()
        refresh()
        status.config(text='Cancelled. Return to idle.')
    def undo():
        board.undo()
        refresh()
        status.config(text='Last phrase removed. Return to idle.')
    tk.Button(root, text='Cancel pending', command=cancel).pack(pady=4)
    tk.Button(root, text='Undo last phrase', command=undo).pack(pady=4)
    label('Scores are not calibrated certainty. Selected phrases stay in this window; nothing is sent.', 13, '#9fafc8')
    refresh()
    root.mainloop()
