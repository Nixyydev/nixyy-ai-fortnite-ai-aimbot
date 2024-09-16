import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("500x350")

smoothness = 5

def setSmoothness(val):
    global smoothness
    smoothness = int(val)
    smoothness_label.configure(text=str(int(val)))
    

frame = customtkinter.CTkFrame(master=root)
frame.pack(padx=60, pady=20, fill="both", expand=True)

title_label = customtkinter.CTkLabel(master=frame, text="nixyy test", font=("Roboto", 24))
title_label.pack(padx=10, pady=12)

smoothness_label = customtkinter.CTkLabel(master=frame, text="5", font=("Roboto", 12))
smoothness_label.pack(padx=10, pady=12)

smoothness_slider_label = customtkinter.CTkLabel(master=frame, text="Smoothness:", font=("Roboto", 12))
smoothness_slider_label.pack(padx=10, pady=12)

smoothess_slider = customtkinter.CTkSlider(master=frame, from_=0, to=10, number_of_steps=10, command=setSmoothness)
smoothess_slider.set(5.0)
smoothess_slider.pack(padx=10, pady=12)


root.mainloop()
