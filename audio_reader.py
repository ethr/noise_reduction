def chunk_size(max_freq):
  return 2 * max_freq

def chunk_time_milli(bytes_per_second, width, chunk_size):
  return chunk_size * sample_time_milli(bytes_per_second, width)

def sample_time_milli(bytes_per_second, width):
  return 1000 * width / bytes_per_second
