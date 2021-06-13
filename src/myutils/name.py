import warnings
import collections

# Globals
delimiter_tag="_"
delimiter_same_tag="-"
delimiter_groups="."

__f_extensions__ = [".pdf", ".jpg", ".png", ".root", ".h5", ".h5py"]
__no_tag_name__ = "no_tag"


class Name(object):

  def __inti__(self):

    # Name dict
    self._name = collections.OrderedDict()

  def add2tag(self, keys, tag=__no_tag_name__):

    # Some checks
    if tag not in self._name: self._name[tag] = []
    if not isinstance(keys, list): keys = [keys]
    # Fill name dict
    for key in keys:
      if key is not self._name[tag]:
        self._name[tag].append(key)
    return self

  def name(self):

    # Get the no tag part
    name = ""
    if __no_tag_name__ in self._name:
      no_tag = self._name.pop(__no_tag_name__)
      name = delimiter_groups.join(no_tag) + delimiter_groups
    # Convert to list
    tag_key_list = self._name.items()
    # Flatten this list
    tag_key_list = [tuple([key[0]] + key[1]) for key in tag_key_list]
    return name + encode(tag_key_list)


def encode(tag_name_list):

  """
  Encode names and associated tags to one string

  :param tag_name_list: list of tuples that connects tags and names
  :param delimiter_tag: string that is used to connect tag and name, e.g., tag+delimiter_tag+name
  :param delimiter_same_tag: string that is used to connect names taht are associated to the same tag, e.g., type+delimiter_tag+name1+delimiter_same_tag+...+delimiter_same_tag+nameN
  :param delimiter_groups: string that separates different group, e.g., tag1+delimiter_tag+name1+delimiter_groups+tag2+delimiter_tag+name2
  :type tag_name_list: list of tuples
  :type delimiter_tag: str
  :type delimiter_same_tag: str
  :type delimiter_groups: str
  :rtype str
  """

  # get a list of reserved characters
  reserved_str = [delimiter_tag, delimiter_same_tag, delimiter_groups]

  if not isinstance(tag_name_list, list):
    tag_name_list = [tag_name_list]

  # Initialize name that is about to be returned
  name_list = []
  # Loop over all elements in 'tag_name_list' to get tags and names
  for tag_name_tuple in tag_name_list:
    # First element is the tag
    tag = str(tag_name_tuple[0])
    names = [str(tag_name_tuple[i]) for i in range(1,len(tag_name_tuple))]
    # Do a check if an already reserved string has been used
    if any(True for string in names for key in reserved_str if key in string):
      warnings.warn("'Delimiter_tag', 'delimiter_same_tag' or 'delimiter_groups' should not be used in names. The string might not be decodable. If this is not important, ignore this message.", stacklevel=2)
    # update name_list
    name_list.append(tag+delimiter_tag+delimiter_same_tag.join(names))

  # Return final name
  return delimiter_groups.join(name_list)


def decode(name, tag=None):

  """
  Decode names and associated tags into a list of tuples

  :param name: a string that is supposed to be decoded
  :param delimiter_tag: string that is used to connect tag and name, e.g., tag+delimiter_tag+name
  :param delimiter_same_tag: string that is used to connect names taht are associated to the same tag, e.g., type+delimiter_tag+name1+delimiter_same_tag+...+delimiter_same_tag+nameN
  :param delimiter_groups: string that separates different group, e.g., tag1+delimiter_tag+name1+delimiter_groups+tag2+delimiter_tag+name2
  :type name: str
  :type delimiter_tag: str
  :type delimiter_same_tag: str
  :type delimiter_groups: str
  :rtype list of tuples
  """

  import re
  # split into groups
  group_list = name.split(delimiter_groups)
  # split based on delimiter_same_tag
  if not tag:
    return [tuple(re.split("%s|%s" % (delimiter_tag, delimiter_same_tag), group)) for group in group_list]
  else:
    return [tuple(re.split("%s|%s" % (delimiter_tag, delimiter_same_tag), group)) for group in group_list if tag in group][0][1]


def long_substr(data):

  substr = ""
  if len(data) > 1 and len(data[0]) > 0:
    for i in range(len(data[0])):
      for j in range(len(data[0])-i+1):
        if j > len(substr) and all(data[0][i:i+j] in x for x in data):
          substr = data[0][i:i+j]
  return substr


def is_substr(find, data):

  if len(data) < 1 and len(find) < 1:
    return False
  for i in range(len(data)):
    if find not in data[i]:
      return False
  return True


def common_name_from_tag(names):

  name_tag_list, common_tag_list = [], []
  # Split names (list of list of tuples)
  for name in names:
    name_tag_list.append(decode(name))
  # Get a list of all (unique) tags
  tags = list(set([name_tuple[0] for name_comp in name_tag_list for name_tuple in name_comp if name_tuple[1:]]))
  # If there are no tags, use other method
  if len(tags) == 0:
    return common_name_from_str(names)
  # Keep tags that are common to all names
  for tag in tags:
    for name_comp in name_tag_list:
      if not any(tag == name_tuple[0] for name_tuple in name_comp) and tag in tags:
        tags.remove(tag)
  # Now, to each tag get a list of names
  tag_dict = {tag : [] for tag in tags}
  for tag in tags:
    for name_comp in name_tag_list:
      for name_tuple in name_comp:
        if tag == name_tuple[0]:
          tag_dict[tag].append(delimiter_same_tag.join(list(name_tuple[1:])))
  # If there is more than one element in the list associated with a tag, get largest common substring
  for tag in dict(tag_dict):
    tag_dict[tag] = list(set(tag_dict[tag]))
    if len(tag_dict[tag]) == 1:
      tag_dict[tag] = tag_dict[tag].pop()
    else:
      common_str = long_substr(tag_dict[tag])
      if not common_str:
        del tag_dict[tag]
      else:
        tag_dict[tag] = common_str
  # Make it a new name
  return encode([(k, v) for k, v in tag_dict.iteritems()])


def common_name_from_str(names):

  return long_substr(names)


def common_name(names, tagged=True, default="common"):

  if len(names) == 1:
    return names[0]

  if tagged:
    return common_name_from_tag(names)
  else:
    c_name = common_name_from_str(names)
    if not c_name: return default
    else: return c_name


if __name__ == "__main__":

  a = ["a_first.b_second.c_third.e_rejBkgSig.pdf", "a_first.b_second.c_fourth.d_penis.e_rejBkgBkg.pdf", "a_first.g_second.c_third.e_rejBkgSig.pdf", "a_l.g_second.c_third.e_rejBkgSig.pdf"] 
  print(common_name(a))
