import pypdf

path = r"C:\Users\ashwa\Downloads\CS3492-DBMS Unit-1.pdf"
reader = pypdf.PdfReader(path)

print(f"{len(reader.pages)} pages")

full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
print("total chars:", len(full_text))

print("\n--- First 1000 chars of page 1 ---")
print(reader.pages[0].extract_text()[:1000])

print("\n--- Searching for 'normal form' (case-insensitive) ---")
idx = full_text.lower().find("normal form")
if idx != -1:
    print(full_text[max(0, idx - 300):idx + 1000])
else:
    print("STILL NOT FOUND")