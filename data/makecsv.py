import tarfile
import csv
import re

target_genre = ["it-life-hack", "kaden-channel", "dokujo-tsushin", "topic-news", \
    "livedoor-homme", "movie-enter", "peachy" "smax", "sports-watch"]

zero_fnames = []
one_fnames = []
tsv_fname = "all.tsv"
tgz_fname = "ldcc-20140209.tar.gz"

brackets_tail = re.compile('【[^】]*】$')
brackets_head = re.compile('^【[^】]*】')

def remove_brackets(inp):
    output = re.sub(brackets_head, '',
                   re.sub(brackets_tail, '', inp))
    return output

def read_title(f):
    # 2行スキップ
    next(f)
    next(f)
    title = next(f) # 3行目を返す
    title = remove_brackets(title.decode('utf-8'))
    return title[:-1]

with tarfile.open(tgz_fname) as tf:
    # 対象ファイルの選定
    for ti in tf:
        # ライセンスファイルはスキップ
        if "LICENSE.txt" in ti.name:
            continue
        for i in range(len(target_genre)):
            if target_genre[i] in ti.name and ti.name.endswith(".txt"):
                zero_fnames.append(ti.name)
                break
    with open(tsv_fname, "w") as wf:
        writer = csv.writer(wf, delimiter='\t')
        # ラベル 0
        for name in zero_fnames:
            f = tf.extractfile(name)
            title = read_title(f)
            row = [target_genre[0], 0, '', title]
            writer.writerow(row)
        # ラベル 1
        for name in one_fnames:
            f = tf.extractfile(name)
            title = read_title(f)
            row = [target_genre[1], 1, '', title]
            writer.writerow(row)