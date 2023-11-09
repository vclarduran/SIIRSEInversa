import struct

if struct.calcsize("P") * 8 == 64:

  print("Running in 64-bit mode.")

else:

  print("Not running in 64-bit mode.")