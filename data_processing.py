def create_doc_list(tokens, labels, docs):
    i = 0
    doc = []
    while i != len(tokens):
        token_label_pair = (tokens[i], labels[i])
        doc.append(token_label_pair)
        i += 1
    docs.append(doc)
    return docs


def doc_counter(filename):
    doc_count = 0
    file = open(filename, "r", encoding="utf8")
    for line in file:
        if line[0] == "#" and line[2] == "r":
            doc_count += 1
        else:
            continue
    print("File", filename, "contains", doc_count, "documents.")


def process_data(file_name):
    n_docs = 0
    tokens = []
    labels = []
    docs = []
    file = open(file_name, "r", encoding="utf8")
    for line in file:
        if line[0] == "#":
            continue
        elif line[0] == "\n":
            docs = create_doc_list(tokens, labels, docs)
            n_docs += 1
            tokens.clear()
            labels.clear()
            continue
        else:
            as_list = line.split("\t")
            tokens.append(as_list[1])
            labels.append(as_list[2].replace("\n", ""))
    file.close()

    docs = create_doc_list(tokens, labels, docs)
    n_docs += 1
    # print(docs)
    print("Number of documents processed from", file_name, ":", n_docs)
    return docs


# doc_counter("train.txt")
# doc_counter("dev.txt")
# doc_counter("test.txt")

# train_docs = process_data("train.txt")
# dev_docs = process_data("dev.txt")
# test_docs = process_data("test.txt")
