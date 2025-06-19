import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("weatherAUS.csv")
df = df[['RainToday', 'Humidity9am', 'Temp9am', 'WindSpeed9am', 'RainTomorrow']].dropna()

le_rain_today = LabelEncoder()
le_rain_tomorrow = LabelEncoder()
df['RainToday'] = le_rain_today.fit_transform(df['RainToday'])
df['RainTomorrow'] = le_rain_tomorrow.fit_transform(df['RainTomorrow'])

X = df[['RainToday', 'Humidity9am', 'Temp9am', 'WindSpeed9am']]
y = df['RainTomorrow']
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X, y)

root = tk.Tk()
root.title("Prediksi Hujan Besok Berdasarkan Cuaca Hari Ini")

tk.Label(root, text="Apakah Hari Ini Hujan?").grid(row=0, column=0, padx=5, pady=5, sticky="e")
rain_today_cb = ttk.Combobox(root, values=['No', 'Yes'], state="readonly")
rain_today_cb.grid(row=0, column=1)
rain_today_cb.current(0)

tk.Label(root, text="Kelembaban 09.00 (0-100%)").grid(row=1, column=0, padx=5, pady=5, sticky="e")
humidity_entry = tk.Entry(root)
humidity_entry.grid(row=1, column=1)
humidity_entry.insert(0, "60")

tk.Label(root, text="Suhu 09.00 (Celsius)").grid(row=2, column=0, padx=5, pady=5, sticky="e")
temp_entry = tk.Entry(root)
temp_entry.grid(row=2, column=1)
temp_entry.insert(0, "20")

tk.Label(root, text="Kecepatan Angin 09.00 (km/jam)").grid(row=3, column=0, padx=5, pady=5, sticky="e")
wind_entry = tk.Entry(root)
wind_entry.grid(row=3, column=1)
wind_entry.insert(0, "10")

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.grid(row=5, column=0, columnspan=2, pady=10)

def prediksi():
    try:
        rain_today = le_rain_today.transform([rain_today_cb.get()])[0]
        humidity = float(humidity_entry.get())
        temp = float(temp_entry.get())
        wind = float(wind_entry.get())
        input_data = [[rain_today, humidity, temp, wind]]

        pred = clf.predict(input_data)[0]
        hasil = le_rain_tomorrow.inverse_transform([pred])[0]

        if hasil == "Yes":
            result_label.config(text="Prediksi: Besok AKAN HUJAN.", fg="blue")
        else:
            result_label.config(text="Prediksi: Besok TIDAK hujan.", fg="green")

    except Exception as e:
        result_label.config(text=f"Input tidak valid! {e}", fg="red")

tk.Button(root, text="Prediksi", command=prediksi, bg="orange").grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
