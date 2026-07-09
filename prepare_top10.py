import pandas as pd

print("Dosya okunuyor...")

# Sadece ihtiyacımız olan sütunları oku
df = pd.read_csv(
    "dashboard_data.csv",
    usecols=["line_name", "number_of_passenger"]
)

print("Toplanıyor...")

top_lines = (
    df.groupby("line_name", as_index=False)["number_of_passenger"]
      .sum()
)

top_lines.to_csv("top_lines.csv", index=False)

print("Bitti!")