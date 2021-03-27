#!/usr/bin/env python

def _run ():

  try:
    import myutils
    print("[TEST] Congratulations, `myutils` is now installed on your system :).")
  except ImportError, e:
    print("[WARNING] Module `myutils` is not installed on your system. Something went wrong.")

  try:
    import myroot
    print("[TEST] Congratulations, `myroot` is now installed on your system :).")
  except ImportError, e:
    print("[WARNING] Module `myroot` is not installed on your system. Something went wrong.")

  try:
    import myhep
    print("[TEST] Congratulations, `myhep` is now installed on your system :).")
  except ImportError, e:
    print("[WARNING] Module `myhep` is not installed on your system. Something went wrong.")

  try:
    import myplt
    print("[TEST] Congratulations, `myplt` is now installed on your system :).")
  except ImportError, e:
    print("[WARNING] Module `myplt` is not installed on your system. Something went wrong.")

  try:
    import myml
    print("[TEST] Congratulations, `myml` is now installed on your system :).")
  except ImportError, e:
    print("[WARNING] Module `myml` is not installed on your system. Something went wrong.")


if __name__ == "__main__":

  _run()
