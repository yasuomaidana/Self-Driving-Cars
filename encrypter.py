import PyPDF2

if __name__ == "__main__":
    file_name = input("PDF Name: ")
    reader = PyPDF2.PdfReader(file_name)
    writer = PyPDF2.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(input("File password:"))
    writer.write(file_name.replace(".pdf","-e.pdf"))