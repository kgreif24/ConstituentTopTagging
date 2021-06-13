import numpy as np

def run_batched (process, args, max_processes, queue=None):                                          

  """
  Generic method to run 'process' in parallel on batches of 'args'.                                  
  
  Arguments:
    process: The process to batch, parallelise. Assumed to inherit from                            
      'multiprocessing.Process'
    args: List of arguments to be batch, each item passed to 'process'.                            
    max_processes: Maximal number of concurrent processes to run.                                  
  """ 

  # Check(s)
  assert isinstance(args, (list, tuple))                                                             
  assert len(args)                                                                                   
  
  # Batch the function `args` as to never occupy more than `max_processes`.
  batches = map(list, np.array_split(args, np.ceil(len(args) / float(max_processes))))               

  # Loop batches of args                                                                             
  results = list()
  for ibatch, batch in enumerate(batches):
    print(" - Batch %s/%s | Contains %s arguments" % (ibatch + 1, len(batches), len(batch)))           
 
    # Convert files using multiprocessing                                                            
    processes = map(process, batch)                                                                  
    
    # Add queue (possibly `None`)                                                                    
    for p in processes: p.queue = queue                                                                                
    
    # Start processes
    for p in processes: p.start()                                                                    
    
    # (Opt.) Get results
    if queue is not None:
      results += [queue.get() for _ in processes]                                                    
    
    # Wait for conversion processes to finish                                                        
    for p in processes: p.join()                                                                     
  
  return results

