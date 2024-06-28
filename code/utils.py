import csv
import os
import fitz # PyMuPDF

def extract_text_and_save_to_csv(folder_paths, output_csv_file):
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['FilePath', 'PageNumber', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        writer.writeheader()

        for folder_path in folder_paths:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(root, file)
                        pdf = fitz.open(pdf_path)
                        for page_num in range(len(pdf)):
                            page = pdf.load_page(page_num)
                            text = page.get_text()
                            # Write row to CSV file
                            writer.writerow({'FilePath': pdf_path, 'PageNumber': page_num + 1, 'Text': text})
                        pdf.close()