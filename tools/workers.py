#para checkear el número de procesadores para determinar el número máximo de workers
#12
import multiprocessing

print(multiprocessing.cpu_count())