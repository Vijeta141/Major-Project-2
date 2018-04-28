import csv, shutil
import random
import os

fieldnames = ["User-ID", "ISBN", "Book-Rating"]
with open('data/data_csv/book_crossings/BX-Book-Ratings.csv', 'r') as csvfile, open('data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv', 'w') as outputfile:
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    writer = csv.DictWriter(outputfile, fieldnames=fieldnames)
    for row in reader:
        if row['Book-Rating'] != '0':
            writer.writerow({'User-ID': row['User-ID'], 'ISBN': row['ISBN'], 'Book-Rating': row['Book-Rating']})

with open('data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)

    book_id = set()

    for row in reader:
        book_id.add(row['ISBN'])

book_id = list(book_id)

with open('data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv', 'r') as csvfile, open('data/data_csv/book_crossings/BX-Book-Explicit-Ratings-Updated.csv', 'w') as outputfile:
  
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    writer = csv.DictWriter(outputfile, fieldnames=fieldnames)

    for row in reader:
        
        if(row['User-ID'] != 'User-ID'):
                      
            row['ISBN'] = book_id.index(row['ISBN'])

            writer.writerow({'User-ID': row['User-ID'], 'ISBN': row['ISBN'], 'Book-Rating': row['Book-Rating']})

shutil.move('data/data_csv/book_crossings/BX-Book-Explicit-Ratings-Updated.csv', 'data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv')

f = open('data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv','r')
o = open('data/data_csv/book_crossings/BX-Book-Test-Ratings.csv', 'w')

file_size = os.path.getsize("data/data_csv/book_crossings/BX-Book-Explicit-Ratings.csv")

parsed = []

while (len(parsed) != 43367):

    offset = random.randrange(file_size)
    
    f.seek(offset)

    f.readline()

    random_line = f.readline()

    if random_line not in parsed:
        
        parsed.append(random_line)

        o.write(random_line)

f.seek(0)

o.close()

o = open('data/data_csv/book_crossings/BX-Book-Test-Ratings.csv', 'r')

explicit_ratings = f.readlines()
test_ratings = o.readlines()

new = open('data/data_csv/book_crossings/BX-Book-Training-Ratings.csv', 'w')

for e in explicit_ratings:
    if e not in test_ratings:
        new.write(e)

f.close()
o.close()
new.close()