#!/usr/bin/env python3
import os
import math
import tempfile
from datetime import timedelta
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Audio handling
from pydub import AudioSegment

# AssemblyAI
import assemblyai as aai

# -------------------------
# Config (edit if you want)
# -------------------------
CHUNK_SECONDS = 60 * 30       # chunk length (30 min). Tip: smaller chunks feel snappier.
OVERLAP_SECONDS = 30          # overlap between chunks
OUTPUT_ENCODING = "wav"       # wav is safe; mp3 also works
TARGET_SAMPLE_RATE = 16000    # mono 16 kHz keeps size small

LANG_CODES = [
    "ALD",  # Auto Language Detection
    "en","en_us","en_au","en_uk",
    "zh","nl","fi","fr","de","hi","it","ja","ko","pl",
    "pt","ru","es","tr","uk","vi",
]

# -------------------------
# Helpers
# -------------------------
def human_ts(ms: int) -> str:
    sec = ms // 1000
    return str(timedelta(seconds=sec))

def normalize_audio(in_path: str) -> AudioSegment:
    """Load with pydub/ffmpeg and convert to mono TARGET_SAMPLE_RATE."""
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    return audio

def split_with_overlap(audio: AudioSegment, chunk_ms: int, overlap_ms: int):
    """Yield (start_ms, end_ms, AudioSegment) with overlap."""
    n = len(audio)
    step = max(1, chunk_ms - overlap_ms)
    i = 0
    while i < n:
        start = i
        end = min(i + chunk_ms, n)
        yield (start, end, audio[start:end])
        if end == n:
            break
        i += step

def export_temp(seg: AudioSegment, ext: str) -> str:
    """Export segment to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    seg.export(path, format=ext)
    return path

def join_base_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path) or os.getcwd()
    return folder, base

def safe_get_api_key(ui_key_entry: tk.Entry) -> str:
    key = ui_key_entry.get().strip()
    if not key:
        key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    return key

# -------------------------
# Transcription core (worker)
# -------------------------
def transcribe_file(ui, file_path: str, lang_code: str, chunk_seconds=CHUNK_SECONDS, overlap_seconds=OVERLAP_SECONDS):
    # Everything in this function runs in a worker thread.
    def s(msg): ui.ui_set_status(msg)
    def log(msg): ui.ui_log_line(msg)
    def set_max(n): ui.ui_progress_set_max(n)
    def set_val(v): ui.ui_progress_set_value(v)

    try:
        s("Preparing audio…")
        ui.ui_set_busy(True)

        # API key
        api_key = safe_get_api_key(ui.api_key_entry)
        if not api_key:
            raise RuntimeError("Missing AssemblyAI API key. Enter it or set ASSEMBLYAI_API_KEY.")

        aai.settings.api_key = api_key

        # Load and normalize
        audio = normalize_audio(file_path)
        chunk_ms = chunk_seconds * 1000
        overlap_ms = overlap_seconds * 1000

        # Prepare output paths
        out_dir, base = join_base_name(file_path)
        part_dir = os.path.join(out_dir, f"{base}_parts")
        os.makedirs(part_dir, exist_ok=True)

        # Language config
        cfg_kwargs = dict(speech_model=aai.SpeechModel.best)
        if lang_code == "ALD":
            cfg_kwargs["language_detection"] = True
        else:
            cfg_kwargs["language_code"] = lang_code

        config = aai.TranscriptionConfig(**cfg_kwargs)

        # Split and transcribe
        s("Splitting audio…")
        chunks = list(split_with_overlap(audio, chunk_ms, overlap_ms))
        total = len(chunks)
        if total == 0:
            raise RuntimeError("Audio seems empty after loading. Please check the file.")

        # Switch to determinate now that we know total
        ui.ui_progress_switch_to_determinate()
        set_max(total)

        parts_text = []
        for idx, (start_ms, end_ms, seg) in enumerate(chunks, start=1):
            s(f"Chunk {idx}/{total}  [{human_ts(start_ms)}–{human_ts(end_ms)}]  exporting…")
            tmp_path = export_temp(seg, OUTPUT_ENCODING)

            try:
                s(f"Chunk {idx}/{total} uploading/transcribing… (this may take a while)")
                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(tmp_path)

                if transcript.status == "error":
                    raise RuntimeError(f"Chunk {idx} failed: {transcript.error}")

                # Save per-chunk text
                part_name = f"{base}_part{idx:03d}_{human_ts(start_ms).replace(':','-')}_to_{human_ts(end_ms).replace(':','-')}.txt"
                part_path = os.path.join(part_dir, part_name)
                with open(part_path, "w", encoding="utf-8") as f:
                    f.write(transcript.text or "")

                # Keep for full join with a header
                header = f"[Part {idx}  {human_ts(start_ms)}–{human_ts(end_ms)}]\n"
                parts_text.append(header + (transcript.text or "") + "\n\n")

                log(f"✓ Saved {part_path}")
            finally:
                try: os.remove(tmp_path)
                except Exception: pass

            set_val(idx)

        # Write joined file
        joined = "".join(parts_text).strip() + "\n"
        full_out = os.path.join(out_dir, f"{base}_FULL.txt")
        with open(full_out, "w", encoding="utf-8") as f:
            f.write(joined)

        log(f"\n✅ Done! Full transcript: {full_out}")
        log(f"🗂  Parts folder: {part_dir}")
        s("Finished.")
        ui.ui_message_info("Done", "Transcription completed.\nCheck the console area for file paths.")
    except Exception as e:
        s("Error.")
        ui.ui_message_error("Transcription error", str(e))
        log(f"ERROR: {e}")
    finally:
        ui.ui_set_busy(False)

# -------------------------
# GUI (main thread only)
# -------------------------
class AppUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AssemblyAI Transcriber")
        self._busy = False

        # File row
        frm_file = ttk.Frame(self.root, padding=8)
        frm_file.pack(fill="x")
        ttk.Label(frm_file, text="Audio file:").pack(side="left")
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(frm_file, textvariable=self.file_var, width=60)
        self.file_entry.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(frm_file, text="Browse…", command=self.pick_file).pack(side="left")

        # Language row
        frm_lang = ttk.Frame(self.root, padding=8)
        frm_lang.pack(fill="x")
        ttk.Label(frm_lang, text="Language:").pack(side="left")
        self.lang_var = tk.StringVar(value="pt")
        self.lang_menu = ttk.Combobox(frm_lang, textvariable=self.lang_var, values=LANG_CODES, state="readonly", width=12)
        self.lang_menu.pack(side="left", padx=6)

        # API key row
        frm_key = ttk.Frame(self.root, padding=8)
        frm_key.pack(fill="x")
        ttk.Label(frm_key, text="API Key (or set ASSEMBLYAI_API_KEY):").pack(side="left")
        self.api_key_entry = ttk.Entry(frm_key, show="*", width=50)
        self.api_key_entry.pack(side="left", padx=6)

        # Chunk/overlap row
        frm_opts = ttk.Frame(self.root, padding=8)
        frm_opts.pack(fill="x")
        ttk.Label(frm_opts, text="Chunk (sec):").pack(side="left")
        self.chunk_spin = tk.Spinbox(frm_opts, from_=15, to=3600, increment=5, width=6)
        self.chunk_spin.delete(0, "end"); self.chunk_spin.insert(0, str(CHUNK_SECONDS))
        self.chunk_spin.pack(side="left", padx=(4,12))

        ttk.Label(frm_opts, text="Overlap (sec):").pack(side="left")
        self.overlap_spin = tk.Spinbox(frm_opts, from_=0, to=120, increment=1, width=6)
        self.overlap_spin.delete(0, "end"); self.overlap_spin.insert(0, str(OVERLAP_SECONDS))
        self.overlap_spin.pack(side="left", padx=4)

        # Go button + status
        frm_go = ttk.Frame(self.root, padding=8)
        frm_go.pack(fill="x")
        self.go_btn = ttk.Button(frm_go, text="Transcribe", command=self.on_go)
        self.go_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(frm_go, textvariable=self.status_var).pack(side="left", padx=10)

        # Progress + console
        self.progress = ttk.Progressbar(self.root, length=400, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(0,6))

        self.console = tk.Text(self.root, height=12, wrap="word")
        self.console.pack(fill="both", expand=True, padx=8, pady=8)

    # -------- thread-safe UI helpers (use .after) --------
    def ui_set_status(self, msg: str):
        self.root.after(0, lambda: self._set_status(msg))

    def ui_log_line(self, s: str):
        self.root.after(0, lambda: self._log_line(s))

    def ui_progress_set_max(self, m: int):
        self.root.after(0, lambda: self._progress_set_max(m))

    def ui_progress_set_value(self, v: int):
        self.root.after(0, lambda: self._progress_set_value(v))

    def ui_progress_switch_to_determinate(self):
        self.root.after(0, self._progress_switch_to_determinate)

    def ui_message_info(self, title, text):
        self.root.after(0, lambda: messagebox.showinfo(title, text))

    def ui_message_error(self, title, text):
        self.root.after(0, lambda: messagebox.showerror(title, text))

    def ui_set_busy(self, busy: bool):
        self.root.after(0, lambda: self._set_busy(busy))

    # -------- internal (main thread only) --------
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _log_line(self, s: str):
        self.console.insert("end", s + "\n")
        self.console.see("end")

    def _progress_set_max(self, m: int):
        self.progress["maximum"] = max(1, m)

    def _progress_set_value(self, v: int):
        self.progress["value"] = v

    def _progress_switch_to_determinate(self):
        # stop marquee and switch to determinate
        self.progress.stop()
        self.progress.config(mode="determinate")
        self.progress["value"] = 0

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            # disable inputs, show watch cursor, start marquee
            self.go_btn.config(state="disabled")
            self.lang_menu.config(state="disabled")
            self.file_entry.config(state="disabled")
            self.api_key_entry.config(state="disabled")
            self.chunk_spin.config(state="disabled")
            self.overlap_spin.config(state="disabled")
            self.root.configure(cursor="watch")
            self.progress.config(mode="indeterminate")
            self.progress.start(50)  # animate
        else:
            # re-enable inputs, stop marquee
            self.progress.stop()
            self.root.configure(cursor="")
            self.go_btn.config(state="normal")
            self.lang_menu.config(state="readonly")
            self.file_entry.config(state="normal")
            self.api_key_entry.config(state="normal")
            self.chunk_spin.config(state="normal")
            self.overlap_spin.config(state="normal")

    # -------- regular UI methods --------
    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio", "*.m4a *.mp3 *.wav *.aac *.flac *.ogg *.wma *.webm"), ("All files","*.*")]
        )
        if path:
            self.file_var.set(path)

    def on_go(self):
        if self._busy:
            return
        path = self.file_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Select a file", "Please choose a valid audio file.")
            return

        try:
            chunk = int(self.chunk_spin.get())
            overlap = int(self.overlap_spin.get())
        except ValueError:
            messagebox.showwarning("Invalid numbers", "Chunk and overlap must be integers.")
            return
        if overlap >= chunk:
            messagebox.showwarning("Invalid overlap", "Overlap must be less than chunk size.")
            return

        lang = self.lang_var.get().strip()
        self.console.delete("1.0", "end")

        # Launch worker thread so UI stays responsive
        worker = threading.Thread(
            target=transcribe_file,
            args=(self, path, lang),
            kwargs=dict(chunk_seconds=chunk, overlap_seconds=overlap),
            daemon=True,
        )
        worker.start()

    def run(self):
        self.root.mainloop()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    AppUI().run()
