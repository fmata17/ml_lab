import numpy as np
from scipy.linalg import null_space

# dim = 9
# rand_vec = np.random.randint(0, 2, dim)

# M = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 4, -1, -1, -1, -1, 0, 0],
#               [0, 0, -1, 1, 0, 0, 0, 0, 0],
#               [0, 0, -1, 0, 1, 0, 0, 0, 0],
#               [0, 0, -1, 0, 0, 1, 0, 0, 0],
#               [0, 0, -1, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# print(rand_vec.T @ M @ rand_vec)

# my_vec = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

# print(my_vec.T @ M @ my_vec)


# ###############################
# for _ in range(10000):
#     rand_vec = np.random.randint(0, 2, dim)

#     res = rand_vec.T @ M @ rand_vec

#     if res == 0:
#         print(rand_vec)


def create_image():
    """Guide the user through selecting their vector."""
    D: int = int(input("How many bits will your image have?\n"))
    bits: str = input(f"Enter {D} consecutive bits.\n")
    bits_vec: np.ndarray = np.array(list(bits))
    bits_vec.reshape(-1, 2)

    # ortho = null_space(bits_vec)

    print(bits_vec.size)
    # print(ortho, type(ortho))


create_image()
