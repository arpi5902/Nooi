import os

BOOKS_FOLDER = "books/"

def load_books():
    books = {}
    for filename in os.listdir(BOOKS_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(BOOKS_FOLDER, filename), "r", encoding="utf-8") as file:
                books[filename] = file.read()
    return books

if __name__ == "__main__":
    books = load_books()
    print(f"Loaded {len(books)} books successfully!")
