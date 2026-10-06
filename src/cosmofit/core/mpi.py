"""
Processes that run one sampler together.

A parallel sampler needs two collective operations from its processes:
every process handing something to the first (``gather``), and the
first handing something to all (``bcast``). Three communicators
provide them:

:class:`Serial`
    One process, which is all of them; nothing is sent anywhere.
:class:`MPIComm`
    ``mpi4py``'s ``COMM_WORLD``, for a run started under ``mpirun``:
    every process runs the same input, and :func:`world` hands each its
    place in it.
:class:`PipeComm`
    Processes this one starts itself, with :func:`spawn`, connected by
    pipes. Started with the ``spawn`` method -- the only one Windows
    has, and the safe one everywhere -- so each rebuilds the model from
    the input rather than inheriting it.

A process that fails in a :class:`PipeComm` sends the failure in place
of what it owed, and the first process raises it, so an error in one
process stops the run instead of leaving the rest waiting.
"""

from __future__ import annotations

import multiprocessing
import traceback


__all__ = ["Serial", "MPIComm", "PipeComm", "world", "spawn", "WorkerError", "Aborted"]


class WorkerError(RuntimeError):
    """A process of a parallel run failed; the message is its traceback."""


class Aborted(RuntimeError):
    """Raised in the processes that did not fail, when another one did."""


class _Failure:
    """What a failed process sends in place of its share."""

    def __init__(self, text: str):
        self.text = text


class Serial:
    """A run of one process."""

    rank = 0
    size = 1

    def gather(self, obj) -> list:
        return [obj]

    def bcast(self, obj):
        return obj


class MPIComm:
    """
    ``mpi4py``'s ``COMM_WORLD``.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
    """

    def __init__(self, comm):

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def gather(self, obj):
        return self.comm.gather(obj, root=0)

    def bcast(self, obj):
        return self.comm.bcast(obj, root=0)


def world():
    """
    The communicator this process runs in: :class:`MPIComm` when started
    under ``mpirun`` with more than one process and ``mpi4py``
    installed, :class:`Serial` otherwise.
    """

    try:
        from mpi4py import MPI
    except ImportError:
        return Serial()

    if MPI.COMM_WORLD.Get_size() < 2:
        return Serial()

    return MPIComm(MPI.COMM_WORLD)


class PipeComm:
    """
    Processes connected by pipes to the first.

    Parameters
    ----------
    rank, size : int
    connections : list
        For the first process, one connection to each of the others, in
        rank order; for the others, the one connection to the first.
    processes : list, optional
        The first process's handles on the others, to tell a process
        that died from one that is slow.
    """

    #: Seconds between checks that a silent process is still alive.
    _POLL = 1.0

    def __init__(self, rank: int, size: int, connections: list, processes=None):

        self.rank = rank
        self.size = size
        self.connections = connections
        self.processes = processes or []

    def _receive(self, index: int):

        connection = self.connections[index]

        while not connection.poll(self._POLL):

            if self.processes and not self.processes[index].is_alive():
                raise WorkerError(
                    f"Process {index + 1} of the run exited without "
                    f"reporting (exit code {self.processes[index].exitcode})."
                )

        obj = connection.recv()

        if isinstance(obj, _Failure):
            raise WorkerError(f"Process {index + 1} of the run failed:\n{obj.text}")

        return obj

    def gather(self, obj):

        if self.rank != 0:
            self.connections[0].send(obj)
            return None

        return [obj] + [self._receive(i) for i in range(len(self.connections))]

    def bcast(self, obj):

        if self.rank == 0:

            for connection in self.connections:
                connection.send(obj)

            return obj

        return self.connections[0].recv()

    def fail(self, error: BaseException) -> None:
        """From a process other than the first: report ``error`` and stop."""

        text = "".join(traceback.format_exception(error))

        try:
            self.connections[0].send(_Failure(text))
        except (OSError, EOFError):
            pass

    def close(self) -> None:
        """From the first process: stop the others."""

        for process in self.processes:

            process.join(timeout=5.0)

            if process.is_alive():
                process.terminate()


def _worker(target, rank, size, connection, args):

    comm = PipeComm(rank, size, [connection])

    try:
        target(comm, *args)
    except Aborted:
        # Another process failed, and the first already knows.
        pass
    except BaseException as error:  # noqa: BLE001 -- reported, then re-raised
        comm.fail(error)
        raise
    finally:
        connection.close()


def spawn(size: int, target, args: tuple = ()) -> PipeComm:
    """
    Start ``size - 1`` processes, each calling ``target(comm, *args)``
    with its own :class:`PipeComm`, and return the first process's.

    ``target`` and ``args`` must pickle: the new processes import
    ``target`` by name and rebuild everything else from ``args``.
    """

    context = multiprocessing.get_context("spawn")

    connections, processes = [], []

    for rank in range(1, size):

        mine, theirs = context.Pipe()

        process = context.Process(
            target=_worker, args=(target, rank, size, theirs, args), daemon=True,
        )
        process.start()

        theirs.close()

        connections.append(mine)
        processes.append(process)

    return PipeComm(0, size, connections, processes)
