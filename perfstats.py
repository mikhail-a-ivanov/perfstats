import os
import numpy as np

class PerformanceStats:
    """
    Performance stats class.
    
    Instance attributes
    -------------------
    
    path: string
        path to the .log files
    gpuruns: bool
        tells whether the runs are GPU-accelerated
    verbose: bool
        tells whether extra printing should occur
    files: list of strings
        names of the .log files
    ncpus: list of ints
        total number of CPU cores (MPI procs X OpenMP threads)
    nprocs: list of ints
        number of MPI processes
    nthreads: list of ints
        number of OpenMP threads
    performance: list of floats
        MD performance, ns/day
    ngpus: list of ints
        total number of GPUs
    """
    def __init__(self, path=".", gpuruns=False, verbose=False):
        self.path = path
        self.gpuruns = gpuruns
        self.verbose = verbose
        
        # List of .log files:
        self.files = [file for file in os.listdir(self.path) if ".log" in file]
        
        # Performance stats
        self.nprocs = []
        self.nthreads = []
        self.performance = []
        self.ncpus = []
        if self.gpuruns:
            self.ngpus = []
            
        # Collect stats
        self.getStats()
        # Sort stats
        self.sortStats()
        
    def __str__(self):
        return(f"PerformanceStats object containing data from {len(self.files)} files.")
    
    def getStats(self):
        """
        Reads every log file and saves the performance data
        """
        for file in self.files:
            if self.verbose:
                print(f"Opening {file}...")
            with open(self.path + file) as f:
                lines = f.readlines()
                for line in lines:
                    if "Using" in line and "OpenMP" in line:
                        if self.verbose:
                            print(f"Number of OpenMP threads: {line.split()[1]}")
                        self.nthreads.append(line.split()[1])
                    elif "Using" in line and "MPI" in line:
                        if self.verbose:
                            print(f"Number of MPI processes: {line.split()[1]}")
                        self.nprocs.append(line.split()[1])
                    elif "Performance:" in line:
                        self.performance.append(line.split()[1])
                    elif self.gpuruns and "compatible GPU" in line:
                        if self.verbose:
                            print(f"Number of GPUs: {line.split()[-3]}")
                        self.ngpus.append(line.split()[-3])
                    elif "Fatal error" in line:
                        if self.verbose:
                            print(f"MD run stopped with a fatal error...")
        assert len(self.nprocs) == len(self.nthreads) == len(self.performance), \
        "Number of performance entries should be the same!"
        
    def sortStats(self):
        """
        Convert and sort the stats with respect to performance
        """
        self.files = np.array(self.files)
        self.nprocs = np.array([int(proc) for proc in self.nprocs])
        self.nthreads = np.array([int(thread) for thread in self.nthreads])
        self.performance = np.array([float(perf) for perf in self.performance])
        
        indeces = np.argsort(self.performance)
        self.files = list(self.files[indeces])
        self.nprocs = list(self.nprocs[indeces])
        self.nthreads = list(self.nthreads[indeces])
        self.performance = list(self.performance[indeces])
        self.ncpus = [self.nprocs[i] * self.nthreads[i] for i in range(len(self.nprocs))]
        
        if self.gpuruns:
            self.ngpus = np.array([int(gpu) for gpu in self.ngpus])
            self.ngpus = list(self.ngpus[indeces])
                
    def subdivideomp(self):
        """
        Subdivides stats into subarrays corresponding
        to different number of OpenMP threads
        """

        if isinstance(self.nthreads[0], list):
            print("Performance stats are subdivided already!")
        else:
            ompset = set(self.nthreads)
            print(f"Subdividing the stats into {len(ompset)} sets...")
            if len(ompset) > 1:
                self.files = [self.files]
                self.nprocs = [self.nprocs]
                self.nthreads = [self.nthreads]
                self.performance = [self.performance]
                self.ncpus = [self.ncpus]
                if self.gpuruns:
                    self.ngpus = [self.ngpus]

                for ompId, omp in enumerate(ompset):
                    self.files.append([])
                    self.nprocs.append([])
                    self.nthreads.append([])
                    self.performance.append([])
                    self.ncpus.append([])
                    if self.gpuruns:
                        self.ngpus.append([])
                    for statId, stat in enumerate(self.nthreads[0]):
                        if stat == omp:
                            self.nthreads[-1].append(stat)
                            self.files[-1].append(self.files[0][statId])
                            self.nprocs[-1].append(self.nprocs[0][statId])
                            self.ncpus[-1].append(self.ncpus[0][statId])
                            self.performance[-1].append(self.performance[0][statId])
                            if self.gpuruns:
                                self.ngpus[-1].append(self.ngpus[0][statId])

                del(self.files[0])
                del(self.nprocs[0])
                del(self.nthreads[0])
                del(self.performance[0])
                del(self.ncpus[0])
                if self.gpuruns:
                    del(self.ngpus[0])
            else:
                print("No subdivision is performed.")