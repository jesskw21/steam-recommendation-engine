message = input("> ")
words = message.split(' ')
emojis = {
    ":)": "😊",
    ":(": "😞",
    ":D": "😄",
    ":P": "😛",
    ";)": "😉",
    ":o": "😮",
    ":/": "😕",
    ":|": "😐",
    ":*": "😘",
    "B)": "😎",
    ":3": "😺",
    ":'(": "😢",
    ":$": "😳",
    ":@": "😡",
    ":&": "😬",
    ":!": "😲",
    ":^)": "😏",
    ":v": "😜",
    ":c": "😔",
    ":b": "😋",
    ":q": "😶",
    ":l": "😒",
    ":h": "😇",
}
output = ""
for word in words:
    emojis.get(word, word)
    output += emojis.get(word, word) + " "
print(output)
