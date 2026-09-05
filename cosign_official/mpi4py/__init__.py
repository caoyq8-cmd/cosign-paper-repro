"""
Dummy mpi4py MPI stub for single‑GPU runs.
Drop‑in replacement when real mpi4py is not available / single gpu.
Compatible with both .rank / Get_rank(), .size / Get_size()
"""
class _FakeComm:
    rank = 0
    size = 1

    def Get_rank(self):
        return self.rank
    def Get_size(self):
        return self.size
    def Barrier(self):
        pass
    def bcast(self, obj, root=0):
        return obj


class MPI:
    COMM_WORLD = _FakeComm()


__all__ = ["MPI"]
