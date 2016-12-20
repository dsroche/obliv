#!/usr/bin/env python3

"""Module for a multi threaded sftp-based backend for ORAM storage.

The mt_ssh_store class works similarly to the fstore class to
provide a list-like functionality that in fact stores each list
element as a separate file. The difference is that these will
be stored on a remote server with file transfer accomplished
using multiple, persistent SFTP connections.
"""

import os
from os import path
from collections import MutableSequence
import sys
import threading
from queue import Queue, Empty
from collections import Counter, namedtuple, OrderedDict
import weakref
from enum import Enum
import logging

import paramiko

from . import sftp

logger = logging.getLogger(__name__)

class CopyOp(Enum):
    """The kinds of tasks that threads perform."""
    GET = 1
    PUT = 2
    DEL = 3
    DIE = 4

taskcount = 0

"""An object of this type represents a single pending
operation to be performed over SFTP."""
CopyTask = namedtuple('CopyTask', 
        ['command', 'taskid', 'index', 'filename', 'contents'])

def makeTask(command, index=None, filename=None, contents=None):
    """Creates a CopyTask object. 

    The command must be from the enum type CopyOp above.
    index is the index in the buffer that is being worked on.
    filename is the corresponding backend filename to read/write.
    contents is the contents that will be written, for a PUT.
    """
    global taskcount
    myid = taskcount
    taskcount += 1
    return CopyTask(command, myid, index, filename, contents)

    
"""Response indicating the task was completed."""
CopyResponse = namedtuple('CopyResponse',
        ['command', 'taskid', 'index', 'contents'])

def makeResponse(task, contents=None):
    """Creates a CopyRespones based on the given CopyTask.

    In the case of a GET task, the contents should be set accordingly.
    """
    return CopyResponse(task.command, task.taskid, task.index, contents)


class Copier(threading.Thread):
    """A Copier is a single thread maintaining an open SFTP connection
    and performing copying tasks on demand.

    Tasks are grabbed from the global task queue that exists in 
    the "parent" mt_ssh_store instance.
    """

    def __init__(self, parent, threadid):
        super().__init__()
        self.setDaemon(True)
        self._sftp = sftp.SFTP(parent.ssh_info, parent.subdir)
        self._taskq = weakref.ref(parent._taskq)
        self._resq = weakref.ref(parent._resq)
        self._tid = threadid

    def run(self):
        logger.info("Copier thread {} initializing".format(self._tid))

        with self._sftp:
            logger.debug("Copier thread {} ready".format(self._tid))
            die = False # flag to indicate completion

            # loop to wait for and process commands
            while not die:
                # dereference these objects; will return None if 
                # the parent class is destroyed
                taskq, resq = self._taskq(), self._resq()
                if not taskq or not resq:
                    logger.warning("Copier thread {} stopping abnormally".format(self._tid))
                    return

                try:
                    task = taskq.get(timeout=2)
                except Empty:
                    continue
                logger.debug("thread {} takes task {}".format(self._tid,task))
                response = None

                if task.command == CopyOp.GET:
                    response = makeResponse(task, self._sftp.read(task.filename))
                elif task.command == CopyOp.PUT:
                    self._sftp.write(task.filename, task.contents)
                    response = makeResponse(task)
                elif task.command == CopyOp.DEL:
                    self._sftp.delete(task.filename)
                    response = makeResponse(task)
                elif task.command == CopyOp.DIE:
                    logger.info("Copier thread {} stopping".format(self._tid))
                    response = makeResponse(task)
                    die = True
                else:
                    logger.warn("Copier thread got invalid command: {}".format(task))

                if response is not None:
                    resq.put(response)
                taskq.task_done()

        logger.info("Copier thread {} finished".format(self._tid))


class mt_ssh_store(MutableSequence):
    """Provides a list-like interface which stores bytes over sftp.
    
    Files are stored in a given directory with names like 132.fstore.

    Note: this class spawns and uses threads, but is very much NOT
    thread-safe itself!

    Instances of this class CAN be effectively pickled and unpickled.
    The SFTP connection is persistent for the duration of the object's
    lifetime.

    At object creation, a number of Copier threads are started up,
    each with their own persistent SFTP connection. A global task queue
    is stored in this object, and the Copier threads pull tasks from
    that queue and put the results in a separate response queue.
    """

    def __init__(self, ssh_info, dirname, nthreads=5, size=0, bufsize=10):
        """Creates a list view on storage files in the given directory.
        
        If the size argument is not zero, that many files (with proper names)
        must already exist on the server.
        """
        self._ssh_info = ssh_info
        self._subdir = dirname
        self._len = size
        self._desired_threads = nthreads
        self._nthreads = 0
        self._bufsize = bufsize
        self._taskq = Queue() # tasks being sent to Copier threads
        self._resq = Queue() # responses from Copier threads
        self._count = Counter() # counts files transferred by size

        # This is the most important data structure here.
        # It's a map from index to file contents, that keeps track of
        # access time so that popitem(False) will return the least recently
        # used element.
        self._buffer = OrderedDict()

        # these dictionaries keep track of pending operations in the taskq.
        # they are each a map from (index) to (taskid).
        # Every index in either of these is also in self._buffer,
        # but no index should simultaneously be in both pending lists to
        # avoid race conditions among the backend threads.
        self._pend_reads = {}
        self._pend_writes = {}

    @property
    def ssh_info(self):
        """The ssh_info object that describes the sftp connection"""
        return self._ssh_info

    @property
    def subdir(self):
        """The folder on the remote host where this store's files will go"""
        return self._subdir

    @property
    def nthreads(self):
        """How many threads are running concurrently to copy data"""
        return self._nthreads

    @property
    def bufsize(self):
        """The maximum number of blocks that can be prefetched"""
        return self._bufsize

    @property
    def current_bufsize(self):
        """The size of the current buffer, accounting for pending read/write operations."""
        return len(self._buffer) - len(self._pend_writes)

    def set_len(self, newlen):
        """Expand the storage to a length, assumes that data will be found at that length <- UGLY HACK"""
        self._len = newlen
        for ind in self._buffer:
            if ind >= newlen:
                logger.warning("Dropping data at index {}".format(ind))
                del self._buffer[ind]
                self._pend_reads.pop(ind, None)
                self._pend_writes.pop(ind, None)

    def __len__(self):
        return self._len

    def open(self):
        """Called when entering a with block to get stuff going"""
        if self.nthreads:
            logger.warning("Threads already running; open() is doing nothing.")
        else:
            logger.info("Starting up multithreaded sftp storage")
            try:
                with sftp.SFTP(self.ssh_info, self.subdir):
                    pass # just ensures that the directory exists to avoid race cond.
            except Exception as e:
                raise RuntimeError("ERROR connecting to {}".format(self.ssh_info.hostname)
                        ) from e
            self._copiers = []
            for i in range(self._desired_threads):
                self._copiers.append(Copier(self, i))
                self._copiers[-1].start()
                self._nthreads += 1

    def close(self):
        """Writes everything back to the sftp store and stops all the threads."""
        if self.nthreads:
            logger.info("Shutting down multithreaded sftp storage")
            self.flush()
            for _ in range(self.nthreads):
                self._taskq.put(makeTask(CopyOp.DIE))
            self._complete_tasks()
            assert self.nthreads == 0
            del self._copiers
        else:
            logger.warning("Threads already stopped; close() is doing nothing.")

    def _assert_running(self):
        """Raises an AssertionError if there are no running threads."""
        if not self.nthreads:
            raise RuntimeError("There are no running ssh threads. Use a with statement or the open() method.")

    def __getitem__(self, ind):
        """Retrieves data at the given index.

        If the data is not already in the buffer, it is read from one of the
        open SFTP connections.
        """
        actind = self._ind(ind)
        self.prefetch([actind])

        # wait until the required read actually occurs, if necessary
        while actind in self._pend_reads:
            self._assert_running()
            self._process(self._resq.get())

        return self._buffer[actind]

    def __setitem__(self, ind, value):
        """Assigns data to the given index in the buffer.

        If the buffer is already full, the least recently used item will be
        written back to the backend.
        """
        if not isinstance(value, bytes):
            raise TypeError("can only store bytes data.")

        actind = self._ind(ind)

        # remove the effects from any pending reads or writes to this index
        self._clear_pending(actind)

        self._buffer[actind] = value
        self._buffer.move_to_end(actind)

        # check overflow
        self._handle_overflow()

    def __delitem__(self, ind):
        """Removes the data at the given index, which MUST equal len() - 1."""

        actind = self._ind(ind)
        if actind != self._len - 1:
            raise IndexError("You can only remove from the end.")

        # remove the effects from any pending reads or writes to this index
        self._clear_pending(actind)

        task = makeTask(CopyOp.DEL, actind, self._filename(actind))
        self._taskq.put(task)
        self._buffer[actind] = None
        self._pend_writes[actind] = task.taskid
        self._len -= 1

    def insert(self, ind, value):
        """Inserts value at index ind, which MUST equal len()."""
        if ind == len(self):
            self._len += 1
            self.__setitem__(ind, value)
        else:
            raise IndexError("You can only insert at the end.")

    def flush(self):
        """Flushes the buffer; writes all files back to the sftp store."""
        for ind in self._pend_reads:
            del self._buffer[ind]
        self._pend_reads.clear()
        for ind, contents in self._buffer.items():
            if ind not in self._pend_writes:
                self._writeback(ind)

    def prefetch(self, indices):
        """Assigns thread to fetch data at the given indices into the buffer.
        
        Existing buffer data may be flushed out to make room, depending on bufsize.
        If len(indices) > bufsize, a ValueError is raised.

        After this operation, everything in the indices list will
        either be in the buffer already, or awaiting completion in the task queue.
        """

        if len(indices) > self.bufsize:
            raise ValueError("Prefetch size {} exceeds buffer size {}".format(len(indices), self.bufsize))

        self._sync() # process any pending buffer updates

        # add indices to pending operations as necessary
        for actind in map(self._ind, indices):
            if actind not in self._buffer:
                task = makeTask(CopyOp.GET, actind, self._filename(actind))
                self._taskq.put(task)
                self._buffer[actind] = None
                self._pend_reads[actind] = task.taskid
            else:
                self._buffer.move_to_end(actind)

        # check overflow
        self._handle_overflow()

    def _filename(self, ind):
        return str(ind) + '.fstore'

    def _ind(self, ind):
        """Computes the actual index and checks out of bounds."""
        posind = ind if ind >= 0 else len(self) + ind
        if 0 <= posind < len(self):
            return posind
        else:
            raise IndexError("index out of bounds: {}".format(ind))

    def _clear_pending(self, index):
        """Ignores any pending reads from, and waits for any pending writes to,
        the given index.

        After this, it should be safe to add a new operation for the specified index
        onto the queue without the risk of any race conditions.
        """
        self._sync() # process any pending buffer updates
        self._pend_reads.pop(index, None)
        while index in self._pend_writes:
            self._assert_running()
            self._process(self._resq.get())

    def _handle_overflow(self):
        """Checks for the buffer getting to large, and if it is, queues up
        writeback operations as necessary."""
        logger.debug("Check overflow: {} used out of {}".format(self.current_bufsize, self.bufsize))
        while self.current_bufsize > self.bufsize:
            index, contents = self._buffer.popitem(False)
            if index in self._pend_writes:
                # send to the back of the list; it's already being written back.
                self._buffer[index] = contents
            elif index in self._pend_reads:
                # ignore the value whenever it comes in, but don't re-write it.
                del self._pend_reads[index]
            else:
                # flush this item to the backend
                self._buffer[index] = contents
                self._writeback(index)

    def _writeback(self, index):
        """Creates a task to write back the given index."""
        assert index in self._buffer
        assert index not in self._pend_writes and index not in self._pend_reads
        if not self.nthreads:
            logger.warning("PUT operation is queued but no threads are running. Use with ...: or open().")
        task = makeTask(CopyOp.PUT, index, self._filename(index), self._buffer[index])
        self._taskq.put(task)
        self._pend_writes[index] = task.taskid
        self._count[len(self._buffer[index])] += 1

    def _process(self, resp):
        """Processes a single CopyResponse."""
        logger.debug("processing {}".format(resp))
        if resp.command == CopyOp.GET:
            if self._pend_reads.get(resp.index, None) == resp.taskid:
                self._buffer[resp.index] = resp.contents
                del self._pend_reads[resp.index]
            self._count[len(resp.contents)] += 1

        elif resp.command in {CopyOp.PUT, CopyOp.DEL}:
            if self._pend_writes.get(resp.index, None) == resp.taskid:
                del self._buffer[resp.index]
                del self._pend_writes[resp.index]

        elif resp.command == CopyOp.DIE:
            self._nthreads -= 1

        else:
            logger.warning("response queue got invalid response: {}".format(resp))

        self._resq.task_done()

    def _sync(self):
        """Processes any pending responses to sync the buffer."""
        while not self._resq.empty():
            self._process(self._resq.get_nowait())

    def _complete_tasks(self):
        """Waits to complete all pending tasks in the queue."""

        if not self._taskq.empty() or not self._resq.empty():
            self._assert_running()

            # wait for tasks to complete
            self._taskq.join()

            # process responses
            self._sync()

        assert self._taskq.empty() and self._resq.empty()
        assert not self._pend_reads and not self._pend_writes

    def __enter__(self):
        """magic method called at the beginning of a with statement"""
        self.open()
        return self

    def __exit__(self, extype, exval, tb):
        """magic method called at the end of a with block"""
        self.close()

    def __getstate__(self):
        """Used to pickle this object."""
        self.flush()
        self._complete_tasks()
        state = self.__dict__.copy()
        if self.nthreads:
            del state['_copiers']
            state['_nthreads'] = 0
        del state['_taskq']
        del state['_resq']
        return state

    def __setstate__(self, state):
        """Used to unpickle this object."""
        if '_nthreads' in dir(self) and self.nthreads:
            logger.warning("Unpickling over running sftp storage object!")
            self.close()
        self.__dict__.update(state)
        self._taskq = Queue()
        self._resq = Queue()

    def clear_counts(self):
        self._count.clear()

    def print_counts(self):
        if len(self._count) > 10:
            print("More than 10 different block sizes; THERE IS A PROBLEM!!!")
        else:
            for bs in sorted(self._count):
                print("Transfered {} files of size {}".format(self._count[bs], bs))
            if not self._count:
                print("ZERO transfers performed.")
