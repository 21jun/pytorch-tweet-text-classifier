
def normalise_text(text):
    text = text.str.lower()  # lowercase
    text = text.str.replace(r"\#", "")  # replaces hashtags
    text = text.str.replace(r"http\S+", "URL")  # remove URL addresses
    text = text.str.replace(r"@", "")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text
