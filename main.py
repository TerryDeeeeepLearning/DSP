from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy.fft import fft, ifft
import sys
import librosa
import numpy as np
import soundfile as sf
import pyaudio
import seaborn as sns
import time
from gtts import gTTS
import librosa.display
import speech_recognition as sr
import os
from scipy.signal import chirp, resample, resample_poly
from scipy import signal
class FilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        # Alpha 值滑動條
        self.alpha_label = QLabel("Alpha (0-1):")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_value = QLabel("0.5")
        
        self.alpha_slider.valueChanged.connect(self.update_alpha_label)
        
        layout.addWidget(self.alpha_label)
        layout.addWidget(self.alpha_slider)
        layout.addWidget(self.alpha_value)

        # 截止頻率輸入
        self.cutoff1_label = QLabel("Cutoff Frequency 1 (Hz):")
        self.cutoff1_spin = QSpinBox()
        self.cutoff1_spin.setRange(20, 20000)
        layout.addWidget(self.cutoff1_label)
        layout.addWidget(self.cutoff1_spin)

        self.cutoff2_label = QLabel("Cutoff Frequency 2 (Hz):")
        self.cutoff2_spin = QSpinBox()
        self.cutoff2_spin.setRange(20, 20000)
        layout.addWidget(self.cutoff2_label)
        layout.addWidget(self.cutoff2_spin)

        # 按鈕
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.setWindowTitle("Filter Settings")
        
    def update_alpha_label(self):
        value = self.alpha_slider.value() / 100
        self.alpha_value.setText(f"{value:.2f}")
        
    def get_alpha(self):
        return self.alpha_slider.value() / 100

    def get_cutoff_frequencies(self):
        return self.cutoff1_spin.value(), self.cutoff2_spin.value()
    
class AudioParamDialog(QDialog):
    def __init__(self, dialog_type="generate", wave_type="chirp", parent=None):
        super().__init__(parent)
        self.dialog_type = dialog_type
        self.wave_type = wave_type
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # 採樣率選擇
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["11025", "22050", "24000", "44100", "48000", "96000", "192000"])
        self.sample_rate_combo.setCurrentText("44100")
        form_layout.addRow("Sampling Rate (Hz):", self.sample_rate_combo)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.1, 1.0)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setValue(0.5)
        form_layout.addRow("Amplitude:", self.amplitude_spin)

        # 持續時間
        self.duration_spin = QSpinBox()
        if self.dialog_type == "record":
            self.duration_spin.setRange(3, 30)
        else:
            self.duration_spin.setRange(1, 5)
        self.duration_spin.setValue(3)
        form_layout.addRow("Duration (seconds):", self.duration_spin)

        if self.dialog_type == "generate":
            if self.wave_type == "chirp":
                # chirp 波形需要最小和最大頻率
                self.min_freq_spin = QSpinBox()
                self.min_freq_spin.setRange(20, 20000)
                self.min_freq_spin.setValue(20)
                form_layout.addRow("Min Frequency (Hz):", self.min_freq_spin)

                self.max_freq_spin = QSpinBox()
                self.max_freq_spin.setRange(20, 20000)
                self.max_freq_spin.setValue(1000)
                form_layout.addRow("Max Frequency (Hz):", self.max_freq_spin)
            else:
                # 其他波形只需要單一頻率
                self.freq_spin = QSpinBox()
                self.freq_spin.setRange(20, 20000)
                self.freq_spin.setValue(1000)
                form_layout.addRow("Frequency (Hz):", self.freq_spin)

        layout.addLayout(form_layout)

        # 按鈕
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setWindowTitle("setting")
        self.setWindowIcon(QIcon('QtDSP.ico'))

    def get_values(self):
        values = {
            'sample_rate': int(self.sample_rate_combo.currentText()),
            'duration': self.duration_spin.value(),
            'amplitude': self.amplitude_spin.value()
        }
        
        if self.dialog_type == "generate":
            if self.wave_type == "chirp":
                values.update({
                    'min_freq': self.min_freq_spin.value(),
                    'max_freq': self.max_freq_spin.value()
                })
            else:
                values.update({
                    'frequency': self.freq_spin.value()
                })
        return values
        
class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self._setup_initial_state()
        
    def _setup_initial_state(self):
        """初始化所有狀態變數"""
        self.filename = None
        self.sampling_rate = 44100
        self.is_playing = False
        self.current_position = 0
        self.x = None  # 添加 x 的初始化
        self.recording_data = []
        self.is_recording = False
        self.stream = None
        self.start_timestamp = 0
        self.accumulated_time = 0
        self.playback_state = "stopped"
        self.label1 = None  # 添加 label1 的初始化
        self.rates = ["11025", "22050", "24000", "44100", "48000", "96000", "192000"]
        
        # Screen resolution
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width() - 10
        self.screen_height = screen.height() - 85

    def setupUi(self, MainWindow):
        """介面設置"""
        # 保存 MainWindow 引用
        self.MainWindow = MainWindow
        
        # Main window setup
        MainWindow.setWindowTitle("QtDSP")
        # 獲取螢幕解析度
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width() - 10
        self.screen_height = screen.height() - 85
        MainWindow.setGeometry(0, 0, self.screen_width, self.screen_height)
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowIcon(QIcon('ico/QtDSP.ico'))
        MainWindow.setStyleSheet("#MainWindow{border-image:url(img/pexels-hendrikbgr-744318.jpg)}")
        self.centralWidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralWidget)
        
        # 創建 label1
        self.label1 = QLabel(self.centralWidget)
        self.label1.setGeometry(0, 30, self.screen_width, 200)
        
        # Create Menubar
        self.menubar = QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.setup_menus()

        # Create Toolbar
        self.toolbar = QToolBar(MainWindow)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.setup_toolbar()

        # Create Statusbar
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: white;
                border-top: 1px solid #3d3d3d;
            }
            QLabel {
                padding: 3px;
                margin: 2px;
                border-radius: 3px;
                background-color: #3d3d3d;
            }
        """)
        
        # 創建狀態標籤
        self.statusLabel = QLabel("Ready", self.statusbar)
        self.statusLabel.setStyleSheet("color: white; padding: 3px 10px;")
        self.statusbar.addWidget(self.statusLabel)
        
        # 添加彈簧來控制間距
        spacer = QWidget(self.statusbar)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.statusbar.addWidget(spacer)
        
        self.sblabel1 = QLabel("Play Information", self.statusbar)
        self.sblabel1.setStyleSheet("color: white;")
        self.statusbar.addWidget(self.sblabel1)
        
        # 添加固定寬度的間隔
        spacer2 = QWidget(self.statusbar)
        spacer2.setFixedWidth(20)
        self.statusbar.addWidget(spacer2)
        
        self.sblabel2 = QLabel("File Information", self.statusbar)
        self.sblabel2.setStyleSheet("color: white;")
        self.statusbar.addPermanentWidget(self.sblabel2)
        
        MainWindow.setStatusBar(self.statusbar)

    def setup_menus(self):
        """選單"""    
        # 定義主選單結構
        menu_definitions = {
            "File": [("Open", self.FileOpen), ("Save", self.FileSave), ("Exit", QApplication.quit)],
            "View": [("Refresh", self.refresh_audio), ("Play", self.play_audio), 
                     ("Pause", self.pause_audio), ("Stop", self.Stop_audio)],
            "Generate": [("Generate Chirp", self.generate_chirp), ("Generate Sawtooth", self.generate_sawtooth),
                        ("Generate Triangle", self.generate_triangle), 
                        ("Generate Sinusoid", self.generate_sinusoid), ("Generate Square", self.generate_square)],
            "Audio": [("Record", self.record_audio), ("Downsampling", self.downsampling),
                     ("Upsampling", self.upsampling), ("Sampling Rate Conversion", self.sampling_rate_conversion)],
            "Filter": [],
            "Analysis": [],
            "Effect": [("Time Scaling", self.time_scaling), ("Time Streching", self.time_streching), ("Tremelo", self.tremelo),
                      ("Echo", self.Echo), ("Reverb", self.Reverb)],
            "AI": [("Chinese Recognition", self.recognize_chinese)],
            "Help": [("About", self.help)],
        }
    
        # 創建所有選單
        for menu_name, actions in menu_definitions.items():
            menu = self.menubar.addMenu(menu_name)
            
            if menu_name == "Filter":
                
                #FIR
                fir = menu.addMenu("FIR")
                fir_lowpass = fir.addAction("Lowpass")
                fir_lowpass.triggered.connect(lambda: self.apply_filter("fir", "lowpass"))
                fir_highpass = fir.addAction("Highpass")
                fir_highpass.triggered.connect(lambda: self.apply_filter("fir", "highpass"))
                
                #IIR
                iir = menu.addMenu("IIR")
                iir_lowpass = iir.addAction("Lowpass")
                iir_lowpass.triggered.connect(lambda: self.apply_filter("iir", "lowpass"))
                iir_highpass = iir.addAction("Highpass")
                iir_highpass.triggered.connect(lambda: self.apply_filter("iir", "highpass"))
                
                #FIR Window
                fir_window = menu.addMenu("FIR Window")
                fir_window_lowpass = fir_window.addAction("Lowpass")
                fir_window_lowpass.triggered.connect(lambda: self.apply_filter("fir_window", "lowpass"))
                fir_window_highpass = fir_window.addAction("Highpass")
                fir_window_highpass.triggered.connect(lambda: self.apply_filter("fir_window", "highpass"))
                fir_window_bandpass = fir_window.addAction("Bandpass")
                fir_window_bandpass.triggered.connect(lambda: self.apply_filter("fir_window", "bandpass"))
                fir_window_bandstop = fir_window.addAction("Bandstop")
                fir_window_bandstop.triggered.connect(lambda: self.apply_filter("fir_window", "bandstop"))
                
                #Butterworth
                butterworth = menu.addMenu("Butterworth")
                butterworth_lowpass = butterworth.addAction("Lowpass")
                butterworth_lowpass.triggered.connect(lambda: self.apply_filter("butterworth", "lowpass"))
                butterworth_highpass = butterworth.addAction("Highpass")
                butterworth_highpass.triggered.connect(lambda: self.apply_filter("butterworth", "highpass"))
                butterworth_bandpass = butterworth.addAction("Bandpass")
                butterworth_bandpass.triggered.connect(lambda: self.apply_filter("butterworth", "bandpass"))
                butterworth_bandstop = butterworth.addAction("Bandstop")
                butterworth_bandstop.triggered.connect(lambda: self.apply_filter("butterworth", "bandstop"))
                
                #Chebyshev Type I
                cheby1 = menu.addMenu("Chebyshev Type I")
                cheby1_lowpass = cheby1.addAction("Lowpass")
                cheby1_lowpass.triggered.connect(lambda: self.apply_filter("cheby1", "lowpass"))
                cheby1_highpass = cheby1.addAction("Highpass")
                cheby1_highpass.triggered.connect(lambda: self.apply_filter("cheby1", "highpass"))
                cheby1_bandpass = cheby1.addAction("Bandpass")
                cheby1_bandpass.triggered.connect(lambda: self.apply_filter("cheby1", "bandpass"))
                cheby1_bandstop = cheby1.addAction("Bandstop")
                cheby1_bandstop.triggered.connect(lambda: self.apply_filter("cheby1", "bandstop"))
                
                #Chebyshev Type II
                cheby2 = menu.addMenu("Chebyshev Type II")
                cheby2_lowpass = cheby2.addAction("Lowpass")
                cheby2_lowpass.triggered.connect(lambda: self.apply_filter("cheby2", "lowpass"))
                cheby2_highpass = cheby2.addAction("Highpass")
                cheby2_highpass.triggered.connect(lambda: self.apply_filter("cheby2", "highpass"))
                cheby2_bandpass = cheby2.addAction("Bandpass")
                cheby2_bandpass.triggered.connect(lambda: self.apply_filter("cheby2", "bandpass"))
                cheby2_bandstop = cheby2.addAction("Bandstop")
                cheby2_bandstop.triggered.connect(lambda: self.apply_filter("cheby2", "bandstop"))
                
                #Elliptic
                elliptic = menu.addMenu("Elliptic")
                elliptic_lowpass = elliptic.addAction("Lowpass")
                elliptic_lowpass.triggered.connect(lambda: self.apply_filter("elliptic", "lowpass"))
                elliptic_highpass = elliptic.addAction("Highpass")
                elliptic_highpass.triggered.connect(lambda: self.apply_filter("elliptic", "highpass"))
                elliptic_bandpass = elliptic.addAction("Bandpass")
                elliptic_bandpass.triggered.connect(lambda: self.apply_filter("elliptic", "bandpass"))
                elliptic_bandstop = elliptic.addAction("Bandstop")
                elliptic_bandstop.triggered.connect(lambda: self.apply_filter("elliptic", "bandstop"))

            elif menu_name == "Analysis":
                PSD = menu.addMenu("Power Spectral Density (PSD)")
                PSD_Peri = PSD.addAction("Periodogram")
                PSD_Peri.triggered.connect(self.Periodogram)  # 連接到 Periodogram 函式

                PSD_Wel = PSD.addAction("Welch's Method")
                PSD_Wel.triggered.connect(self.Welch)  # 連接到 Welch 函式

                Spectrogram = menu.addMenu("Spectrogram")
                Spectrogram_STFT = Spectrogram.addAction("STFT")
                Spectrogram_STFT.triggered.connect(self.STFT)  # 連接到 STFT 函式

                Spectrogram_SciPy = Spectrogram.addAction("SciPy")
                Spectrogram_SciPy.triggered.connect(self.SciPy)  # 連接到 SciPy 函式


            else:
                # 為其他選單添加動作
                for action_name, action_method in actions:
                    action = QAction(action_name, self.MainWindow)
                    action.triggered.connect(action_method)
                    menu.addAction(action)
                
    def update_status_bar(self):
        """狀態欄"""
        # Update play information
        if self.is_recording:
            play_status = "Recording"
        elif self.playback_state == "playing":
            play_status = "Play"
        elif self.playback_state == "paused":
            play_status = "Pause"
        else:
            play_status = "Stop"
        
        self.sblabel1.setText(f"Status: {play_status}")
        
        # Update file information
        if hasattr(self, 'x') and self.x is not None:
            channels = "Stereo" if self.x.ndim > 1 else "Mono"
            duration = len(self.x[0])/self.sampling_rate if self.x.ndim > 1 else len(self.x)/self.sampling_rate
            self.sblabel2.setText(f"File Info: [{channels}] Duration: {duration:.2f}s, Sample Rate: {self.sampling_rate}Hz")
        else:
            self.sblabel2.setText("File Info: No file loaded")

    def setup_toolbar(self):
        """圖標和動作"""
        toolbar_actions = [
            ("ico/fileOpen.ico", "Open file", self.FileOpen),
            ("ico/fileSave.ico", "Save file", self.FileSave),
            ("ico/viewRefresh.ico", "Refresh", self.refresh_audio),
            ("ico/viewPlay.ico", "Play", self.play_audio),
            ("ico/viewPause.ico", "Pause", self.pause_audio),
            ("ico/viewStop.ico", "Stop", self.Stop_audio),
            ("ico/viewRecord.ico", "Record", self.record_audio),
            ("ico/generateChirp.ico", "Generate Chirp", self.generate_chirp),
            ("ico/generateSawtooth.ico", "Generate Sawtooth", self.generate_sawtooth),
            ("ico/generateTriangle.ico", "Generate Triangle", self.generate_triangle),
            ("ico/generateSinusoid.ico", "Generate Sinusoid", self.generate_sinusoid),
            ("ico/generateSquare.ico", "Generate Square", self.generate_square),
            ("ico/aiSpeechRecognitionChinese.ico", "Reconize Chinese", self.recognize_chinese),
            ("ico/aiSpeechRecognitionEnglish.ico", "Reconize English", self.recognize_english),
            ("ico/aiSpeechRecognitionJapanese.ico", "Reconize Japanese", self.recognize_japanese),
            ("ico/aiSpeechRecognitionKorean.ico", "Reconize Korean", self.recognize_korean),
            ("ico/aiSpeechSynthesis.ico", "Synthesis", self.synthesis),
            ("ico/helpAbout.ico", "About", self.help),
        ]

        for icon, tooltip, action in toolbar_actions:
            tool_action = self.toolbar.addAction(QIcon(icon), tooltip)
            tool_action.triggered.connect(action)
            
    def display_signal(self):
        """秀出訊號"""
        if self.x is None:
            return
        try:
            if self.x.ndim not in {1, 2}:
                raise ValueError("Signal dimensionality not supported. Expected 1D or 2D signal.")
            
            if self.x.ndim == 1:
                nr, nc = 200, self.screen_width
            else:
                nr, nc = 400, self.screen_width
            
            interval = 100
            xx = self.subsample(self.x, nc)
            self.qImg = QImage(nc, nr, QImage.Format_RGB888)
            self.qImg.fill(QColor(83, 86, 92))
            painter = QPainter(self.qImg)
            
            pen = QPen(QColor(147, 147, 155))
            painter.setPen(pen)
            
            play_position = max(0, min(int(self.current_position / len(self.x[0]) * nc), nc - 1)) if self.x.ndim > 1 else max(0, min(int(self.current_position / len(self.x) * nc), nc - 1))
            painter.drawLine(play_position, 0, play_position, nr * 2)
            
            pen = QPen(QColor(221, 208, 200))
            pen.setWidth(2)
            painter.setPen(pen)
            
            if self.x.ndim == 1:
                for i in range(nc):
                    y2 = max(0, min(interval - int(xx[i] * interval), nr - 1))
                    painter.drawLine(i, interval, i, y2)
                painter.drawLine(0, interval, nc, interval)
            else:
                for i in range(nc):
                    y2_ch1 = max(0, min(interval - int(xx[0, i] * interval), nr - 1))
                    painter.drawLine(i, interval, i, y2_ch1)
                    y2_ch2 = max(0, min(interval * 3 - int(xx[1, i] * interval), nr - 1))
                    painter.drawLine(i, interval * 3, i, y2_ch2)
                painter.drawLine(0, interval, nc, interval)
                painter.drawLine(0, interval * 3, nc, interval * 3)
                painter.setPen(QColor(219, 210, 201))
                painter.drawLine(0, interval * 2, nc, interval * 2)
            
            painter.end()
            self.label1.setGeometry(0, 30, nc, nr)
            self.label1.setPixmap(QPixmap.fromImage(self.qImg))
            
            if self.is_playing:
                painter = QPainter(self.qImg)
                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                play_position = max(0, min(int((time.time() - self.start_timestamp) * self.sampling_rate / len(self.x) * nc), nc - 1))
                painter.drawLine(play_position, 0, play_position, nr)
                painter.end()
                
        except Exception as e:
            QMessageBox.critical(None, "Display Error", f"Error displaying signal: {str(e)}")
            
    def update_playback_position(self):
        """更新播放位置"""
        if not self.is_playing:
            return
            
        # 計算經過的時間
        adjustment = 0.4 #誤差
        elapsed_time = (time.time() - self.start_timestamp) - adjustment
        # 根據當前採樣率計算應該播放到的位置
        samples_per_second = self.sampling_rate
        self.current_position = int(elapsed_time * samples_per_second)
        
        # 檢查是否播放結束
        total_samples = len(self.x[0]) if self.x.ndim > 1 else len(self.x)
        
        if self.current_position >= total_samples:
            self.Stop_audio()
            self.current_position = 0
        else:
            self.display_signal()
            QTimer.singleShot(20, self.update_playback_position)

    def FileOpen(self):
        """從指定資料夾開檔"""
        try:
            dir = QFileDialog()
            dir.setDirectory("C:/DSP/Python")
            dir.setNameFilter("Audio Files (*.wav *.mp3)")
            
            if dir.exec_():
                filenames = dir.selectedFiles()
                if filenames:
                    # Stop currently playing audio
                    if self.is_playing:
                        self.Stop()
                        
                    self.filename = filenames[0]
                    filename_idx = self.filename.rfind("/") + 1
                    title = "QtDSP - " + self.filename[filename_idx:]
                    MainWindow.setWindowTitle(title)
    
                    self.x, self.sampling_rate = librosa.load(self.filename, sr=None, mono=False)
                    self.current_position = 0  # Reset playback position
                    self.display_signal()
                    self.update_status_bar()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading file: {str(e)}")

    def FileSave(self):
        """存檔"""
        save_filename, _ = QFileDialog.getSaveFileName(None, "Save File", "", "WAV Files (*.wav)")
        if save_filename:
            sf.write(save_filename, self.x.T, self.sampling_rate)
            self.statusbar.showMessage(f"File saved: {save_filename}")

    def play_audio(self):
        """播放"""
        if self.playback_state == 'stopped':
            self.current_position = 0
        if self.filename and self.x is not None:
            if not self.is_playing:
                # 這���設定start_timestamp為當前時間減去當前播放位置所代表的時間，這樣能夠從正確的地方繼續播放
                self.start_timestamp = time.time() - ((self.current_position) / self.sampling_rate)
            
            sd.play(self.x.T[self.current_position:], samplerate=self.sampling_rate)
            self.is_playing = True
            self.playback_state = "playing"
            self.update_playback_position()
            self.disable_audio_related_actions()
            self.update_status_bar()
        else:
            QMessageBox.information(MainWindow, "ERROR", "請先打開一個音檔")

    def pause_audio(self):
        """暫停"""
        if self.is_playing:
            sd.stop()
            self.current_position += 10500 #誤差
            self.is_playing = False
            self.playback_state = "paused"
            self.update_status_bar()
        elif not self.is_playing:
            QMessageBox.information(MainWindow, "ERROR", "你還沒按開始阿????")
        elif not self.filename: 
            QMessageBox.information(MainWindow, "ERROR", "沒檔案阿兄弟")
        elif self.playback_state == "pause":
            QMessageBox.information(MainWindow, "ERROR", "你已經暫停了ㄟ哥哥")

    def Stop_audio(self):
        """停止"""
        if not self.filename: 
            QMessageBox.information(MainWindow, "ERROR", "沒檔案阿兄弟")
            return
        sd.stop()
        self.current_position = 0
        self.is_playing = False
        self.playback_state = "stopped"
        self.enable_audio_related_actions()
        self.display_signal()
        self.update_status_bar()

    def refresh_audio(self):
        """重新整理音訊資料"""
        if not self.filename:
            QMessageBox.information(MainWindow, "ERROR", "沒有檔案可重新整理")
            return
    
        # 處理生成訊號的情況
        if self.filename in ['chirp', 'sawtooth', 'sinusoid', 'square', 'triangle']:
            QMessageBox.information(MainWindow, "INFO", "目前為生成訊號，無需重新整理")
            return
    
        # 保存當前播放狀態
        was_playing = self.is_playing
        if was_playing:
            self.Stop_audio()
    
        try:
            # 使用原始采样率重新加載
            self.x, self.sampling_rate = librosa.load(self.filename, sr=None, mono=False)
            self.current_position = 0
            self.display_signal()
            
            # 如果之前在播放，恢復播放
            if was_playing:
                self.play_audio()
        except Exception as e:
            QMessageBox.information(MainWindow, "ERROR", f"重新整理失敗: {str(e)}")
            
    def upsampling(self):
        """往上取樣"""
        if not hasattr(self, 'x') or self.x is None:
            QMessageBox.information(MainWindow, "Error", "No audio file loaded!")
            return
        
        factor, ok = QInputDialog.getInt(MainWindow, "Upsampling", 
                                         "Enter upsampling factor (2-8):", 
                                         value=2, min=2, max=8)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 執行重採樣
            if self.x.ndim > 1:
                self.x = np.array([resample_poly(channel, factor, 1) for channel in self.x])
            else:
                self.x = resample_poly(self.x, factor, 1)
            
            # 更新採樣率和重置播放位置
            self.sampling_rate *= factor
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    
    def downsampling(self):
        """往下取樣"""
        if not hasattr(self, 'x') or self.x is None:
            QMessageBox.information(MainWindow, "Error", "No audio file loaded!")
            return
        
        factor, ok = QInputDialog.getInt(MainWindow, "Downsampling", 
                                         "Enter downsampling factor (2-8):", 
                                         value=2, min=2, max=8)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 執行重採樣
            if self.x.ndim > 1:
                self.x = np.array([resample_poly(channel, 1, factor) for channel in self.x])
            else:
                self.x = resample_poly(self.x, 1, factor)
            
            # 更新採樣率和重置播放位置
            self.sampling_rate //= factor
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
            
    def sampling_rate_conversion(self):
        """選擇不同取樣率"""
        if not hasattr(self, 'x') or self.x is None:
            QMessageBox.information(MainWindow, "Error", "No audio file loaded!")
            return
        
        rate, ok = QInputDialog.getItem(MainWindow, "Sampling Rate Conversion",
                                      "Select new sampling rate (Hz):",
                                      self.rates, current=self.rates.index("44100"), editable=False)
        
        if ok:
            new_rate = int(rate)
            if new_rate != self.sampling_rate:
                # Calculate resampling ratio
                ratio = new_rate / self.sampling_rate
                new_length = int(len(self.x) * ratio)
                
                if self.x.ndim > 1:
                    self.x = np.array([resample(channel, int(len(channel) * ratio)) 
                                     for channel in self.x])
                else:
                    self.x = resample(self.x, new_length)
                
                self.sampling_rate = new_rate
                self.current_position = 0
                self.display_signal()
                self.update_status_bar()

    def record_audio(self):
        """錄音功能"""
        if not self.is_recording:
            response = QMessageBox.question(MainWindow, "Record", 
                                            "Do you want to start recording?", 
                                            QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.Yes:
                dialog = AudioParamDialog("record", MainWindow)
                if dialog.exec_():
                    values = dialog.get_values()
                    self.sampling_rate = values['sample_rate']
                    duration = values['duration']
                    self.amplitude = values['amplitude']  # 存儲 amplitude
    
                    self.is_recording = True
                    self.update_status_bar()
    
                    # 禁用工具列按鈕
                    for i in range(0, 18):
                        if i != 6:
                            self.toolbar.actions()[i].setEnabled(False)
    
                    # 禁用菜單項目
                    for menu in self.menubar.findChildren(QMenu):
                        for action in menu.actions():
                            action.setEnabled(False)
    
                    # 清除先前的錄音數據
                    self.recording_data = []
    
                    # 開始錄音
                    try:
                        self.stream = pyaudio.PyAudio().open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sampling_rate,
                            input=True,
                            frames_per_buffer=1024
                        )
                        self.record_audio_chunk()
    
                        # 設定自動停止的計時器
                        QTimer.singleShot(duration * 1000, self.stop_recording)
    
                    except Exception as e:
                        QMessageBox.critical(MainWindow, "Recording Error", 
                                            f"Failed to start recording: {str(e)}")
                        self.is_recording = False
        else:
            response = QMessageBox.question(MainWindow, "Record", 
                                            "Do you want to stop recording?", 
                                            QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.Yes:
                self.stop_recording()


    def stop_recording(self):
        """停止錄音"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.update_status_bar()
        # 停止錄音
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception as e:
            QMessageBox.critical(MainWindow, "Error", f"Error stopping recording: {str(e)}")
        
        # 保存錄音檔案
        if self.recording_data:
            recorded_audio = np.concatenate(self.recording_data).astype(np.int16)
            save_path = "recorded_audio.wav"
            sf.write(save_path, recorded_audio, self.sampling_rate)
            self.statusbar.showMessage(f"Recording saved: {save_path}")
        
            # 更新UI顯示
            self.filename = save_path
            MainWindow.setWindowTitle("QtDSP - recorded_audio")
            self.x, self.sampling_rate = librosa.load(self.filename, sr=self.sampling_rate, mono=False)
            self.display_signal()
        else:
            QMessageBox.information(MainWindow, "No Data", "No audio was recorded.")
        
        # 恢復工具列按鈕和菜單項目
        for i in range(0, 18):
            self.toolbar.actions()[i].setEnabled(True)
        
        for menu in self.menubar.findChildren(QMenu):
            for action in menu.actions():
                action.setEnabled(True)
    
    def record_audio_chunk(self):
        """錄音檢查"""
        if self.is_recording:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
    
                # 應用 amplitude 設定
                scaled_audio_data = (audio_data * self.amplitude).clip(-32768, 32767).astype(np.int16)
    
                # 檢查音量
                volume = np.abs(scaled_audio_data).mean()
                if volume > 30000:  # 接近最大值32767
                    self.statusbar.showMessage("Warning: Audio level too high!", 2000)
    
                self.recording_data.append(scaled_audio_data)
                QTimer.singleShot(10, self.record_audio_chunk)
            except Exception as e:
                self.is_recording = False
                QMessageBox.critical(None, "Recording Error", 
                                     f"Error during recording: {str(e)}")

    def disable_audio_related_actions(self):
        """停用工具"""
        self.toolbar.actions()[0].setEnabled(False)
        self.toolbar.actions()[1].setEnabled(False)
        self.toolbar.actions()[6].setEnabled(False)
        self.toolbar.actions()[7].setEnabled(False) 
        self.toolbar.actions()[8].setEnabled(False)  
        self.toolbar.actions()[9].setEnabled(False) 
        self.toolbar.actions()[10].setEnabled(False)
        self.toolbar.actions()[11].setEnabled(False) 
        self.toolbar.actions()[12].setEnabled(False) 
        self.toolbar.actions()[13].setEnabled(False)
        self.toolbar.actions()[14].setEnabled(False)
        self.toolbar.actions()[15].setEnabled(False)
        self.toolbar.actions()[16].setEnabled(False)

        for menu in self.menubar.findChildren(QMenu):
            for action in menu.actions():
                action.setEnabled(False)

    def enable_audio_related_actions(self):
        """恢復"""
        # Re-enable buttons/actions related to generating waves or recording
        self.toolbar.actions()[0].setEnabled(True)
        self.toolbar.actions()[1].setEnabled(True)
        self.toolbar.actions()[6].setEnabled(True) 
        self.toolbar.actions()[7].setEnabled(True) 
        self.toolbar.actions()[8].setEnabled(True) 
        self.toolbar.actions()[9].setEnabled(True)  
        self.toolbar.actions()[10].setEnabled(True) 
        self.toolbar.actions()[11].setEnabled(True) 
        self.toolbar.actions()[12].setEnabled(True) 
        self.toolbar.actions()[13].setEnabled(True) 
        self.toolbar.actions()[14].setEnabled(True) 
        self.toolbar.actions()[15].setEnabled(True) 
        self.toolbar.actions()[16].setEnabled(True) 
    
        for menu in self.menubar.findChildren(QMenu):
            for action in menu.actions():
                action.setEnabled(True)

    def subsample(self, x, n):
        """再抽樣"""
        if x.ndim == 1:
            return x[::len(x)//n]
        else:
            return np.array([row[::len(row)//n] for row in x])
    def time_scaling(self):
        """時間縮放"""
        if self.x is None:
            QMessageBox.warning(MainWindow, "警告", "請先載入音訊檔案！")
            return
        
        factor, ok = QInputDialog.getDouble(MainWindow, "時間縮放", 
                                             "請輸入縮放因子 (0.1-5.0):", 
                                             value=1.0, min=0.1, max=5.0)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 時間縮放
            self.x = librosa.effects.time_stretch(self.x, rate = factor)
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()

    def time_streching(self):
        """時間拉伸"""
        if self.x is None:
            QMessageBox.warning(MainWindow, "警告", "請先載入音訊檔案！")
            return
        
        factor, ok = QInputDialog.getDouble(MainWindow, "時間拉伸", 
                                             "請輸入拉伸因子 (0.1-5.0):", 
                                             value=1.0, min=0.1, max=5.0)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 時間拉伸
            self.x = librosa.effects.time_stretch(self.x, rate = factor)  # 只傳遞音訊信號和因子
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()

    def tremelo(self):
        """顫音效果"""
        if self.x is None:
            QMessageBox.warning(MainWindow, "警告", "請先載入音訊檔案！")
            return
        
        depth, ok = QInputDialog.getDouble(MainWindow, "顫音深度", 
                                            "請輸入顫音深度 (0.0-1.0):", 
                                            value=0.5, min=0.0, max=1.0)
        rate, ok = QInputDialog.getDouble(MainWindow, "顫音速率", 
                                           "請輸入顫音速率 (0.1-10.0):", 
                                           value=5.0, min=0.1, max=10.0)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 應用顫音效果
            t = np.linspace(0, len(self.x) / self.sampling_rate, len(self.x))
            modulation = 1 + depth * np.sin(2 * np.pi * rate * t)
            self.x *= modulation
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    def Echo(self):
        """回聲效果"""
        if self.x is None:
            QMessageBox.warning(MainWindow, "警告", "請先載入音訊檔案！")
            return
        
        delay, ok = QInputDialog.getInt(MainWindow, "回聲延遲", 
                                         "請輸入延遲時間 (毫秒):", 
                                         value=200, min=0)
        decay, ok = QInputDialog.getDouble(MainWindow, "回聲衰減", 
                                            "請輸入衰減因子 (0.0-1.0):", 
                                            value=0.5, min=0.0, max=1.0)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 應用回聲效果
            delay_samples = int(delay * self.sampling_rate / 1000)
            echo_signal = np.zeros(len(self.x) + delay_samples)
            echo_signal[delay_samples:] = self.x * decay
            self.x = self.x + echo_signal[:len(self.x)]
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    def Reverb(self):
        """混響效果"""
        if self.x is None:
            QMessageBox.warning(MainWindow, "警告", "請先載入音訊檔案！")
            return
        
        reverberation_time, ok = QInputDialog.getDouble(MainWindow, "混響時間", 
                                                        "請輸入混響時間 (秒):", 
                                                        value=1.0, min=0.1)
        if ok:
            # 停止當前播放
            if self.is_playing:
                self.Stop_audio()
                
            # 應用混響效果
            self.x = librosa.effects.preemphasis(self.x, coef=reverberation_time)
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()

    def apply_filter(self, filter_type, filter_mode):
        if self.x is None:
            QMessageBox.warning(self.MainWindow, "警告", "請先載入音訊檔案！")
            return

        # Initialize filter dialog
        dialog = FilterDialog(self.MainWindow)
        if dialog.exec_() != QDialog.Accepted:
            return

        alpha = dialog.get_alpha() if filter_type == 'iir' else None

        # Frequency response parameters
        N = 1024
        omega = np.linspace(0, np.pi, N)
        H = np.ones_like(omega, dtype=complex)

        # Design filter based on type and mode
        if filter_type == "fir":
            if filter_mode == "lowpass":
                H = 0.5 * (1 + np.exp(-1j * omega))
            elif filter_mode == "highpass":
                H = 0.5 * (1 - np.exp(-1j * omega))
        elif filter_type == "iir":
            if filter_mode == "lowpass":
                H = 1 / (1 + (omega / (alpha * np.pi / 2))**4)
            elif filter_mode == "highpass":
                H = 1 / (1 + (alpha * np.pi / 2 / np.maximum(1e-10, omega))**4)
        elif filter_type == "fir_window":
            cutoff1, cutoff2 = dialog.get_cutoff_frequencies()
            nyquist = self.sampling_rate / 2

            h = np.zeros(N)
            if filter_mode == "lowpass":
                h = np.hamming(N) * np.sinc(2 * cutoff1 / nyquist * (np.arange(N) - (N - 1) / 2))
            elif filter_mode == "highpass":
                h = np.hamming(N) * (
                    np.sinc(np.arange(N) - (N - 1) / 2) -
                    np.sinc(2 * cutoff1 / nyquist * (np.arange(N) - (N - 1) / 2))
                )
            elif filter_mode == "bandpass":
                h = np.hamming(N) * (
                    np.sinc(2 * cutoff2 / nyquist * (np.arange(N) - (N - 1) / 2)) -
                    np.sinc(2 * cutoff1 / nyquist * (np.arange(N) - (N - 1) / 2))
                )
            elif filter_mode == "bandstop":
                h = np.hamming(N) * (
                    np.sinc(np.arange(N) - (N - 1) / 2) - (
                        np.sinc(2 * cutoff2 / nyquist * (np.arange(N) - (N - 1) / 2)) -
                        np.sinc(2 * cutoff1 / nyquist * (np.arange(N) - (N - 1) / 2))
                    )
                )

            h /= np.sum(h)
            H = np.fft.fft(h, N)
        elif filter_type == "butterworth":
            cutoff1, cutoff2 = dialog.get_cutoff_frequencies()
            nyquist = self.sampling_rate / 2

            if filter_mode == "lowpass":
                H = 1 / (1 + (omega / (2 * np.pi * cutoff1 / nyquist))**4)
            elif filter_mode == "highpass":
                H = 1 / (1 + (2 * np.pi * cutoff1 / nyquist / np.maximum(1e-10, omega))**4)
            elif filter_mode == "bandpass":
                H = 1 / (1 + ((omega - 2 * np.pi * cutoff1 / nyquist) * (omega - 2 * np.pi * cutoff2 / nyquist))**4)
            elif filter_mode == "bandstop":
                H = 1 / (1 + (omega * (omega - 2 * np.pi * cutoff1 / nyquist) * (omega - 2 * np.pi * cutoff2 / nyquist))**4)
        elif filter_type in ["cheby1", "cheby2", "elliptic"]:
            cutoff1, cutoff2 = dialog.get_cutoff_frequencies()
            nyquist = self.sampling_rate / 2

            if filter_mode == "lowpass":
                if filter_type == "cheby1":
                    H = 1 / np.sqrt(1 + (omega / cutoff1)**2)
                elif filter_type == "cheby2":
                    H = 1 / np.sqrt(1 + (cutoff1 / omega)**2)
                elif filter_type == "elliptic":
                    H = 1 / (1 + (omega / cutoff1)**4)
            elif filter_mode == "highpass":
                if filter_type == "cheby1":
                    H = 1 / np.sqrt(1 + (cutoff1 / np.maximum(1e-10, omega))**2)
                elif filter_type == "cheby2":
                    H = 1 / np.sqrt(1 + (omega / cutoff1)**2)
                elif filter_type == "elliptic":
                    H = 1 / (1 + (cutoff1 / np.maximum(1e-10, omega))**4)
            elif filter_mode == "bandpass":
                if filter_type == "cheby1":
                    H = 1 / np.sqrt(1 + ((omega - cutoff1) * (omega - cutoff2))**2)
                elif filter_type == "cheby2":
                    H = 1 / np.sqrt(1 + ((cutoff1 - omega) * (cutoff2 - omega))**2)
                elif filter_type == "elliptic":
                    H = 1 / (1 + ((omega - cutoff1) * (omega - cutoff2))**4)
            elif filter_mode == "bandstop":
                if filter_type == "cheby1":
                    H = 1 / np.sqrt(1 + (omega * (omega - cutoff1) * (omega - cutoff2))**2)
                elif filter_type == "cheby2":
                    H = 1 / np.sqrt(1 + (omega * (cutoff1 - omega) * (cutoff2 - omega))**2)
                elif filter_type == "elliptic":
                    H = 1 / (1 + (omega * (omega - cutoff1) * (omega - cutoff2))**4)

        freq = omega * self.sampling_rate / (2 * np.pi)

        # Plot frequency response
        plt.figure(figsize=(8, 6))
        plt.plot(freq, np.abs(H))
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Frequency Response - {filter_type.upper()} {filter_mode}')
        plt.ylim(0, 1.1)

        # Show confirmation dialog
        confirm_dialog = QDialog(self.MainWindow)
        confirm_dialog.setWindowTitle("Frequency Response")
        layout = QVBoxLayout()

        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(confirm_dialog.accept)
        button_box.rejected.connect(confirm_dialog.reject)
        layout.addWidget(button_box)

        confirm_dialog.setLayout(layout)

        if confirm_dialog.exec_() == QDialog.Accepted:
            # Apply filter to the signal
            if self.x.ndim > 1:
                X = np.fft.fft(self.x[0])
            else:
                X = np.fft.fft(self.x)

            freq_signal = np.fft.fftfreq(len(X), 1 / self.sampling_rate)
            omega_signal = 2 * np.pi * freq_signal / self.sampling_rate
            H_signal = np.ones_like(freq_signal, dtype=complex)

            if filter_type == "fir":
                if filter_mode == "lowpass":
                    H_signal = 0.5 * (1 + np.exp(-1j * omega_signal))
                elif filter_mode == "highpass":
                    H_signal = 0.5 * (1 - np.exp(-1j * omega_signal))
            elif filter_type == "iir":
                positive_freq = freq_signal >= 0
                if filter_mode == "lowpass":
                    H_signal[positive_freq] = (1 - alpha) / 2 * (
                        (1 + np.exp(-1j * omega_signal[positive_freq])) /
                        (1 - alpha * np.exp(-1j * omega_signal[positive_freq]))
                    )
                elif filter_mode == "highpass":
                    H_signal[positive_freq] = (1 + alpha) / 2 * (
                        (1 - np.exp(-1j * omega_signal[positive_freq])) /
                        (1 - alpha * np.exp(-1j * omega_signal[positive_freq]))
                    )
                H_signal[~positive_freq] = H_signal[positive_freq][::-1]
            elif filter_type == "fir_window":
                h_signal = h / np.sum(h)
                H_signal = np.fft.fft(h_signal, len(X))
            elif filter_type == "butterworth":
                cutoff1, cutoff2 = dialog.get_cutoff_frequencies()
                nyquist = self.sampling_rate / 2

                if filter_mode == "lowpass":
                    H_signal = 1 / (1 + (freq_signal / cutoff1)**4)
                elif filter_mode == "highpass":
                    H_signal = 1 / (1 + (cutoff1 / np.maximum(1e-10, freq_signal))**4)
                elif filter_mode == "bandpass":
                    H_signal = 1 / (1 + ((freq_signal - cutoff1) * (freq_signal - cutoff2))**4)
                elif filter_mode == "bandstop":
                    H_signal = 1 / (1 + ((freq_signal * (freq_signal - cutoff1) * (freq_signal - cutoff2)))**4)
            elif filter_type in ["cheby1", "cheby2", "elliptic"]:
                cutoff1, cutoff2 = dialog.get_cutoff_frequencies()
                nyquist = self.sampling_rate / 2

                if filter_mode == "lowpass":
                    if filter_type == "cheby1":
                        H_signal = 1 / np.sqrt(1 + (freq_signal / cutoff1)**2)
                    elif filter_type == "cheby2":
                        H_signal = 1 / np.sqrt(1 + (cutoff1 / freq_signal)**2)
                    elif filter_type == "elliptic":
                        H_signal = 1 / (1 + (freq_signal / cutoff1)**4)
                elif filter_mode == "highpass":
                    if filter_type == "cheby1":
                        H_signal = 1 / np.sqrt(1 + (cutoff1 / np.maximum(1e-10, freq_signal))**2)
                    elif filter_type == "cheby2":
                        H_signal = 1 / np.sqrt(1 + (freq_signal / cutoff1)**2)
                    elif filter_type == "elliptic":
                        H_signal = 1 / (1 + (cutoff1 / np.maximum(1e-10, freq_signal))**4)
                elif filter_mode == "bandpass":
                    if filter_type == "cheby1":
                        H_signal = 1 / np.sqrt(1 + ((freq_signal - cutoff1) * (freq_signal - cutoff2))**2)
                    elif filter_type == "cheby2":
                        H_signal = 1 / np.sqrt(1 + ((cutoff1 -freq_signal) * (cutoff2 - freq_signal))**2)
                elif filter_type == "elliptic":
                    H_signal = 1 / (1 + ((freq_signal - cutoff1) * (freq_signal - cutoff2))**4)
            elif filter_mode == "bandstop":
                if filter_type == "cheby1":
                    H_signal = 1 / np.sqrt(1 + (freq_signal * (freq_signal - cutoff1) * (freq_signal - cutoff2))**2)
                elif filter_type == "cheby2":
                    H_signal = 1 / np.sqrt(1 + (freq_signal * (cutoff1 - freq_signal) * (cutoff2 - freq_signal))**2)
                elif filter_type == "elliptic":
                    H_signal = 1 / (1 + (freq_signal * (freq_signal - cutoff1) * (freq_signal - cutoff2))**4)

            filtered_X = X * H_signal
            if self.x.ndim > 1:
                self.x[0] = np.real(np.fft.ifft(filtered_X))
            else:
                self.x = np.real(np.fft.ifft(filtered_X))

            self.display_signal()

        plt.close()


    def generate_chirp(self):
        """稠啾訊號"""
        dialog = AudioParamDialog("generate", "chirp", MainWindow)
        if dialog.exec_():
            values = dialog.get_values()
            self.sampling_rate = values['sample_rate']
            duration = values['duration']
            frequency_start = values['min_freq']
            frequency_end = values['max_freq']
            amplitude = values['amplitude']
            
            self.filename = 'chirp'
            MainWindow.setWindowTitle("QtDSP-Chirp Signal")
            t = np.linspace(0, duration, int(self.sampling_rate * duration))
            self.x = amplitude * chirp(t, f0=frequency_start, f1=frequency_end, 
                                t1=duration, method='linear')
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()

    def generate_sawtooth(self):
        """鋸齒波"""
        dialog = AudioParamDialog("generate", MainWindow)
        if dialog.exec_():
            values = dialog.get_values()
            self.sampling_rate = values['sample_rate']
            duration = values['duration']
            frequency = values['frequency']
            amplitude = values['amplitude']
            
            self.filename = 'sawtooth'
            MainWindow.setWindowTitle("QtDSP-Sawtooth Signal")
            t = np.linspace(0, duration, int(self.sampling_rate * duration))
            self.x = amplitude * (1.0 - np.mod(t * frequency, 1.0))
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    
    def generate_sinusoid(self):
        """弦波"""
        dialog = AudioParamDialog("generate", MainWindow)
        if dialog.exec_():
            values = dialog.get_values()
            self.sampling_rate = values['sample_rate']
            duration = values['duration']
            frequency = values['frequency']
            amplitude = values['amplitude']
            
            self.filename = 'sinusoid'
            MainWindow.setWindowTitle("QtDSP-Sinusoid Signal")
            t = np.linspace(0, duration, int(self.sampling_rate * duration))
            self.x = amplitude * np.sin(2 * np.pi * frequency * t)
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    
    def generate_square(self):
        """方波"""
        dialog = AudioParamDialog("generate", MainWindow)
        if dialog.exec_():
            values = dialog.get_values()
            self.sampling_rate = values['sample_rate']
            duration = values['duration']
            frequency = values['frequency'] 
            amplitude = values['amplitude']
            
            self.filename = 'square'
            MainWindow.setWindowTitle("QtDSP-Square Wave Signal")
            t = np.linspace(0, duration, int(self.sampling_rate * duration))
            self.x = amplitude * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()
    
    def generate_triangle(self):
        """三角波"""
        dialog = AudioParamDialog("generate", MainWindow)
        if dialog.exec_():
            values = dialog.get_values()
            self.sampling_rate = values['sample_rate']
            duration = values['duration']
            frequency = values['frequency']
            amplitude = values['amplitude']
            
            self.filename = 'triangle'
            MainWindow.setWindowTitle("QtDSP-Triangle Wave Signal")
            t = np.linspace(0, duration, int(self.sampling_rate * duration))
            self.x = amplitude * (1 - np.cos(2 * np.pi * frequency * t))
            self.current_position = 0
            self.display_signal()
            self.update_status_bar()

    def Periodogram(self):
        """計算並顯示音訊的功率譜 (Periodogram)"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            if self.x.ndim == 1:
                f, Pxx = signal.periodogram(self.x, fs=self.sampling_rate)
                plt.figure(figsize=(10, 6))
                plt.semilogy(f, Pxx)
            else:
                plt.figure(figsize=(10, 6))
                for channel in range(self.x.shape[0]):
                    f, Pxx = signal.periodogram(self.x[channel, :], fs=self.sampling_rate)
                    plt.semilogy(f, Pxx, label=f"Channel {channel+1}")
                plt.legend()

            plt.title('Periodogram')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.grid()
            plt.show()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error calculating Periodogram: {str(e)}")

    def Welch(self):
        """計算並顯示音訊的 Welch 方法功率譜"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            if self.x.ndim == 1:
                f, Pxx = signal.welch(self.x, fs=self.sampling_rate, nperseg=1024)
                plt.figure(figsize=(10, 6))
                plt.semilogy(f, Pxx)
            else:
                plt.figure(figsize=(10, 6))
                for channel in range(self.x.shape[0]):
                    f, Pxx = signal.welch(self.x[channel, :], fs=self.sampling_rate, nperseg=1024)
                    plt.semilogy(f, Pxx, label=f"Channel {channel+1}")
                plt.legend()

            plt.title("Welch's Method")
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.grid()
            plt.show()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error calculating Welch's Method: {str(e)}")

    def STFT(self):
        """計算並顯示音訊的短時傅立葉變換 (STFT)"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            if self.x.ndim == 1:
                D = librosa.stft(self.x)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                plt.figure(figsize=(10, 6))
                librosa.display.specshow(S_db, sr=self.sampling_rate, x_axis='time', y_axis='hz', cmap='viridis')
            else:
                plt.figure(figsize=(10, 6))
                for channel in range(self.x.shape[0]):
                    D = librosa.stft(self.x[channel, :])
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    plt.subplot(self.x.shape[0], 1, channel + 1)
                    librosa.display.specshow(S_db, sr=self.sampling_rate, x_axis='time', y_axis='hz', cmap='viridis')
                    plt.title(f"STFT Spectrogram - Channel {channel+1}")

            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error calculating STFT: {str(e)}")

    def SciPy(self):
        """計算並顯示音訊的 SciPy Spectrogram"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            if self.x.ndim == 1:
                f, t, Sxx = signal.spectrogram(self.x, fs=self.sampling_rate)
                plt.figure(figsize=(10, 6))
                plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            else:
                plt.figure(figsize=(10, 6))
                for channel in range(self.x.shape[0]):
                    f, t, Sxx = signal.spectrogram(self.x[channel, :], fs=self.sampling_rate)
                    plt.subplot(self.x.shape[0], 1, channel + 1)
                    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
                    plt.title(f"Spectrogram - Channel {channel+1}")

            plt.colorbar(label='Intensity (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.show()
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error calculating SciPy Spectrogram: {str(e)}")

    def recognize_chinese(self):
        """使用 AI 語音辨識中文"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            recognizer = sr.Recognizer()
            with sr.AudioFile(self.filename) as source:
                audio_data = recognizer.record(source)
            result = recognizer.recognize_google(audio_data, language="zh-CN")
            QMessageBox.information(None, "Chinese Recognition", f"Recognized text: {result}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error recognizing Chinese: {str(e)}")


    def recognize_english(self):
        """使用 AI 語音辨識英文"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            recognizer = sr.Recognizer()
            with sr.AudioFile(self.filename) as source:
                audio_data = recognizer.record(source)
            result = recognizer.recognize_google(audio_data, language="en-US")
            QMessageBox.information(None, "English Recognition", f"Recognized text: {result}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error recognizing English: {str(e)}")

    def recognize_japanese(self):
        """使用 AI 語音辨識日文"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            recognizer = sr.Recognizer()
            with sr.AudioFile(self.filename) as source:
                audio_data = recognizer.listen(source)
            result = recognizer.recognize_google(audio_data, language="ja-JP")
            QMessageBox.information(None, "Japanese Recognition", f"Recognized text: {result}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error recognizing Japanese: {str(e)}")

    def recognize_korean(self):
        """使用 AI 語音辨識韓文"""
        try:
            if self.x is None:
                QMessageBox.warning(None, "Warning", "No audio file loaded. Please load an audio file first.")
                return

            recognizer = sr.Recognizer()
            with sr.AudioFile(self.filename) as source:
                audio_data = recognizer.listen(source)
            result = recognizer.recognize_google(audio_data, language="ko-KR")
            QMessageBox.information(None, "Korean Recognition", f"Recognized text: {result}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error recognizing Korean: {str(e)}")

    def synthesis(self):
        """使用 AI 語音合成"""
        try:
            text, ok = QInputDialog.getText(None, "Text to Speech", "Enter text to synthesize:")
            if not ok or not text.strip():
                return

            tts = gTTS(text=text, lang="en")  # 預設語言為英文，可改成其他語言
            output_path = QFileDialog.getSaveFileName(None, "Save Audio", "", "Audio Files (*.wav)")[0]
            if output_path:
                tts.save(output_path)
                QMessageBox.information(None, "Synthesis", "Audio file saved successfully.")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error during synthesis: {str(e)}")

    def help(self):
        """COPYRIGHT"""
        QMessageBox.information(MainWindow, "About","QtDSP version 2.0\nCopyright@ 賴泓達")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())