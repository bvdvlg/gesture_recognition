import tkinter
import PIL.Image, PIL.ImageTk
from image_container import Constants
import tkinter.ttk as ttk


class GUI:
    def __init__(self, app, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.photo = None
        self.app = app

        # open video source (by default this will try to open the computer webcam)
        self.vid = self.app.vid
        self.window.geometry('{}x{}'.format(int(self.vid.width+20), int(self.vid.height+110)))
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.vid.width, height=self.vid.height)
        self.canvas.place(x=0, y=50)

        label = tkinter.Label(self.window, text="upper_threshold")
        label.place(x=self.vid.width//2, y=10)
        self.upper_theshold_label = ttk.Label(self.window, text=str(Constants.Thresholds.UPPER_TRESHOLD))
        self.upper_theshold_label.place(x=self.vid.width-20, y=30)

        self.upper_scale = ttk.Scale(self.window, from_=1, to=30, command=self.upper_scaler, length=self.vid.width-30)
        self.upper_scale.place(x=10, y=30)
        self.upper_scale.set(Constants.Thresholds.UPPER_TRESHOLD)

        label = tkinter.Label(self.window, text="lower_threshold")
        label.place(x=self.vid.width // 2, y=50)
        self.lower_theshold_label = ttk.Label(self.window, text=str(Constants.Thresholds.LOWER_THRESHOLD))
        self.lower_theshold_label.place(x=self.vid.width-20, y=70)

        self.lower_scale = ttk.Scale(self.window, from_=0.1, to=10, command=self.lower_scaler, length=self.vid.width-30)
        self.lower_scale.place(x=10, y=70)
        self.lower_scale.set(Constants.Thresholds.LOWER_THRESHOLD)

        self.current_word_label = tkinter.Label(self.window, text="current word:")
        self.common_text_label = tkinter.Label(self.window, text="text:")
        self.current_word_label.place(x=10, y=90)
        self.common_text_label.place(x=10, y=110)

        self.current_word_label_info = tkinter.Label(self.window, text=self.app.text.current_word)
        self.common_text_label_info = tkinter.Label(self.window, text=self.app.text.text)
        self.current_word_label_info.place(x=100, y=90)
        self.common_text_label_info.place(x=45, y=110)

        self.button = tkinter.Checkbutton(self.window, text="Do you want to use t9?", command=self.Button_t9)
        self.button.place(x=50, y=550)
        self.t9label = tkinter.Label(self.window, text="Yes" if self.app.text.use_t9 else "No")
        self.button.select()
        self.t9label.place(x=20, y=550)

        self.clear_button = tkinter.Button(self.window, text="Clear", command=self.clear_button_allert)
        self.clear_button.place(x=400, y=550)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 5
        self.update()

        self.window.mainloop()
        app.text.text += app.text.current_word

    def clear_button_allert(self):
        self.app.text.text = ""
        self.app.text.current_word = ""
        self.common_text_label_info["text"] = self.app.text.text + self.app.text.current_word
        self.current_word_label_info["text"] = self.app.text.current_word

    def Button_t9(self):
        self.app.text.use_t9 = not self.app.text.use_t9
        self.t9label["text"] = "Yes" if self.app.text.use_t9 else "No"

    def upper_scaler(self, val):
        Constants.Thresholds.UPPER_TRESHOLD = float(val)
        self.upper_theshold_label["text"] = "%.2f" % Constants.Thresholds.UPPER_TRESHOLD

    def lower_scaler(self, val):
        Constants.Thresholds.LOWER_THRESHOLD = float(val)
        self.lower_theshold_label["text"] = "%.2f" % Constants.Thresholds.LOWER_THRESHOLD

    def update(self):
        # Get a frame from the video source
        ret, frame, marks = self.vid.get_frame()

        if marks is not None:
            marks = marks[0]
            self.app.tracker.update(marks.landmark)
            self.vid.draw_landmarks(frame, marks)
            if self.app.tracker.flag == 1 and self.app.tracker.delayer_flag == 0:
                out, probas = self.app.model.lands2sym(marks.landmark)
                sym = self.app.text.get_sym(out)

                if probas[0][int(out)] > Constants.Thresholds.PROBA_TRESHOLD:
                    self.app.text.append(sym)
                    self.app.tracker.flag = 0

                self.common_text_label_info["text"] = self.app.text.text + self.app.text.current_word
                self.current_word_label_info["text"] = self.app.text.current_word

            if self.app.tracker.delayer_flag > 0:
                self.app.tracker.delayer_flag -= 1
        else:
            self.app.tracker.update_delayed_flag()
            self.app.text.current_word = ""
            self.current_word_label_info["text"] = self.app.text.current_word

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(10, 100, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)