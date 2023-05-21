import numpy as np

from wingman import ReplayBuffer


def test_bulk():
    """Tests repeatedly bulking the buffer and whether it rollovers correctly."""
    bulk_size = 7
    mem_size = 20
    element_shapes = [(3,), ()]
    memory = ReplayBuffer(mem_size=mem_size)

    for iteration in range(10):
        # try to stuff:
        # a) (bulk_size, 3) array
        # b) (bulk_size,) array
        all_tuples = []
        for shape in element_shapes:
            all_tuples.append(np.random.randn(bulk_size, *shape))
        memory.push(all_tuples, bulk=True)

        for step, data in enumerate(all_tuples):
            output = memory.__getitem__((iteration * bulk_size + step) % mem_size)
            for idx in range(len(all_tuples)):
                assert (output[idx] == data[idx]).all(), f"Something went wrong with rollover at iteration {iteration}, expected {data[idx]}, got {output[idx]}."


def test_rollover():
    memory = ReplayBuffer(mem_size=3)
    for i in range(20):
        data1 = np.random.randn(3)
        data2 = np.random.randn()
        memory.push([data1, data2])

        output = memory.__getitem__(i % 3)
        assert (output[0] == data1).all(), f"Something went wrong with rollover."
        assert (output[1] == data2).all(), f"Something went wrong with rollover."
