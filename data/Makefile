#

FNAME=ldcc-20140209.tar.gz
URL="https://drive.google.com/uc?export=download&id=1b-llzNQdmKIp0FYMwzGOKmXdQUNpNXC8"

$(FNAME):
	wget -O $(FNAME) $(URL)

all: vocab.txt prep-title.txt

clean:
	rm -f vocab.txt prep-title.txt title.txt all.tsv

all.tsv: $(FNAME)
	python makecsv.py

title.txt: all.tsv
	cut -f 4- all.tsv | nkf -w -Lu -Z0 | mecab -Owakati > title.txt

vocab.txt prep-title.txt: title.txt
	python wmt_preprocess.py --vocab-file vocab.txt title.txt prep-title.txt
