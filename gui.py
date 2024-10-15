from tkinter import *
from tkinter import font
from chatbot import get_response, bot_name

# Define colors
primary_color = "#FFC0CB"  # pink
secondary_color = "#800080"  # purple
bg_color = "#f8c8dc"  # bibypink (custom)

class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self.bold_font = font.Font(family="Helvetica", size=12, weight="bold")
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, background=bg_color)

        head_label = Label(self.window, bg=bg_color, fg=secondary_color,
                           text="Welcome to Chatbot!", font=self.bold_font, pady=10)
        head_label.place(relwidth=1)

        self.text_widget = Text(self.window, width=20, height=2, bg=bg_color, fg=secondary_color,
                                font=self.bold_font, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        self.msg_entry = Entry(self.window, bg=primary_color, fg=secondary_color, font=self.bold_font)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.825, relx=0.011)

        send_button = Button(self.window, text="Send", font=self.bold_font, width=20, bg=primary_color,
                             command=self._on_enter_pressed)
        send_button.place(relx=0.77, rely=0.825, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event=None):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        response = get_response(msg)
        self._insert_message(response, bot_name)

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        message = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, message)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

if __name__ == "__main__":
    gui = ChatApplication()
    gui.run()
