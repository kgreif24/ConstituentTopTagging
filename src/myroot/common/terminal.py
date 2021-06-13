
def header(title="", description_short="", description_long="", width=70, char="#", sep="    ", flush=True):

  import textwrap
  print_str = ""
  # Top
  for i in range(width): print_str += char
  print_str += char + "\n" + char + "\n"
  # Set the title
  if title:
    print_str += char + " Title:\n"
    for txt in textwrap.wrap(title, width=width-len(sep)-2):
      print_str += char + sep + txt + "\n"
  # Give a short description
  if description_short:
    print_str += char + " Description (short):\n"
    for txt in textwrap.wrap(description_short, width=width-len(sep)-2):
      print_str += char + sep + txt + "\n"
  # Give a longish description
  if description_long:
    print_str += char + " Description (long):\n"
    for txt in textwrap.wrap(description_long, width=width-len(sep)-2):
      print_str += char + sep + txt + "\n"
  # Bottom
  print_str += char + "\n"
  for i in range(width): print_str += char

  if not flush:
    return print_str
  else:
    print(print_str)


if __name__ == "__main__":

  print(header(title="lolaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
