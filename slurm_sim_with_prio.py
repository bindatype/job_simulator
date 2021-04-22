debug = False
#tiny      8:00:00 = 480 minutes
#short  1-00:00:00 = 1440
#defq* 14-00:00:00 =

import sys
import pandas as pd
pd.options.mode.chained_assignment = None 
import simpy
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total Nodes for CPU jobs = 160


NUM_OF_DEFQ_NODES=32
NUM_OF_SHORT_NODES=32
NUM_OF_TINY_NODES = 96
NUM_OF_DEFQ_NODES=int(sys.argv[2])
NUM_OF_SHORT_NODES=int(sys.argv[3])
NUM_OF_TINY_NODES = int(sys.argv[4])

TOTAL_NUM_OF_NODES = NUM_OF_DEFQ_NODES+NUM_OF_SHORT_NODES+NUM_OF_TINY_NODES

MAX_SLURM_PRIO = 4294967295

def ReadData():
    df = pd.read_csv('job14.input')
    return df


df_init  = ReadData()

#if len (sys.argv) != 2 :
#	print("Must specify user or auto");
#	exit(1)

if sys.argv[1] == 'auto':
	if rank == 0:
		df = df_init[df_init.Timelimit > 1440]
	if rank == 1:
		df = df_init[(df_init.Timelimit <= 1440) & (df_init.Timelimit > 480)]
	if rank == 2:
		df = df_init[df_init.Timelimit <= 480]
elif sys.argv[1] == 'user':
	if rank == 0:
		df = df_init[df_init.Partition=='defq']
	if rank == 1:
		df = df_init[df_init.Partition=='short']
	if rank == 2:
		df = df_init[df_init.Partition=='tiny']
else: 
	print("Unknown option")
	exit(1)

df.reset_index(drop=True,inplace=True)
df['SubmitTime']=df['SubmitTime']-df['SubmitTime'].min()
df['InterArrival']=df['SubmitTime'].diff(periods=1)
df["InterArrival"][0]=df["SubmitTime"][0]

if debug:
	with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,'display.width',None,
                       'display.precision', 3,
                       ):
		print(df)
if debug:
    print(df[df['InterArrival'].isnull()]) 

NUM_OF_JOBS = len(df)

start_time  = []
wait_time  = []
run_time = []

def job(env,JobID,SubmitTime,RunTime,Priority,Partition,Timelimit,InterArrival,NCPUS,node):
#    SubmitTime = env.now
    #print("Job %s submitted at time = %.1f" % (JobID, SubmitTime))
    NumberOfNodes = NCPUS%40+1
    with node.request(priority=np.absolute(Priority-MAX_SLURM_PRIO)) as req:
        yield req
        yield env.timeout(InterArrival)
        #yield env.timeout(0)
        job_start = env.now+SubmitTime
        #job_start = env.now
#        print("JobID %s -- Start %d -- Run %d -- Submit %d -- Now %d -- Q %s -- Prio %d" % (JobID, job_start,RunTime,SubmitTime,env.now,Partition,Priority))
        yield env.timeout(RunTime)
        if rank == 2:
            wait_time.append(job_start-SubmitTime)         
        elif rank == 1 :
            wait_time.append(job_start-SubmitTime)
        elif rank == 0:
            wait_time.append(job_start-SubmitTime)
        run_time.append(RunTime)
        if debug:
              print(">> JobID %s | submit %d | wait %d | start %d | Run %d |Timelimit %d |Priority %f |Partition %s | Rank %d | finish %d" % (JobID,SubmitTime,job_start-SubmitTime,job_start,RunTime,Timelimit,Priority,Partition,rank,env.now))



def run_jobs(env):
    for i in range(NUM_OF_JOBS):
    	yield env.timeout(0) # Change to waittime at some point?
    	env.process(job(env, df['JobID'][i],df['SubmitTime'][i],df['RunTime'][i],df['Priority'][i],df['Partition'][i],df['Timelimit'][i],df['InterArrival'][i],df['NCPUS'], node=node))    

env = simpy.Environment()
if rank == 0:
	node = simpy.PriorityResource(env, capacity = NUM_OF_DEFQ_NODES)
if rank == 1:
	node = simpy.PriorityResource(env, capacity = NUM_OF_SHORT_NODES)
if rank == 2:
	node = simpy.PriorityResource(env, capacity = NUM_OF_TINY_NODES)

env.process(run_jobs(env))
env.run()
#env.run(until=1000)

wait_sum = np.sum(wait_time)
wait_mean = np.average(wait_time)
wait_med = np.median(wait_time)

#if rank == 0:
#        print("DefQ sum %d mean %d wait %d " %(wait_sum,wait_mean,wait_med))
#if rank == 1:
#        print("Short sum %d mean %d wait %d " %(wait_sum,wait_mean,wait_med))
#if rank == 2:
#        print("Tiny sum %d mean %d wait %d " %(wait_sum,wait_mean,wait_med))

if rank == 0:
        print("Number of Jobs in defq %d,  median wait %.2f [hrs]" %(len(df),wait_med/3600))
if rank == 1:
        print("Number of Jobs in short %d, median wait %.2f [hrs] " %(len(df),wait_med/3600))
if rank == 2:
        print("Number of Jobs in tiny %d, median wait %.2f [hrs] " %(len(df),wait_med/3600))


#print("DefQ Wait %d/%d -- Short Wait %d/%d -- Tiny Wait %d/%d -- Total Wait %d " %(defq_sum,NUM_OF_DEFQ_NODES,short_sum,NUM_OF_SHORT_NODES,tiny_sum,NUM_OF_TINY_NODES,defq_sum+short_sum+tiny_sum))

#print("DefQ Wait %.1f -- Short Wait %.1f -- Tiny Wait %.1f  " %(defq_mean,short_mean,tiny_mean))


#print("DefQ Wait %.1f -- Short Wait %.1f -- Tiny Wait %.1f " %(defq_mean/NUM_OF_DEFQ_NODES,short_mean/NUM_OF_SHORT_NODES,tiny_mean/NUM_OF_TINY_NODES))


#print("DefQ Wait %.1f -- Short Wait %.1f -- Tiny Wait %.1f " %(defq_med,short_med,tiny_med))

#print("Defq %d -- Short %d-- Tiny %d = SUM %d)" %(len(defq_wait_time),len(short_wait_time),len(tiny_wait_time),len(tiny_wait_time)+len(defq_wait_time)+len(short_wait_time)))

print("Mode %s, Defq %d, Short %d, Tiny %d" %(sys.argv[1],NUM_OF_DEFQ_NODES,NUM_OF_SHORT_NODES,NUM_OF_TINY_NODES))
