import os
import time
import threading

played_files = set()

def schedule_deletion(filename, delay=600):
    def delete_later():
        time.sleep(delay)
        if filename not in played_files and os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted: {filename}")
    threading.Thread(target=delete_later, daemon=True).start()

def mark_as_played(filename):
    played_files.add(filename)
