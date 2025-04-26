

def apply_mask0(matrix, data_bit_positions):
    # (row + col) % 2 == 0
    for (x, y) in data_bit_positions:
        if (x + y) % 2 == 0:
            matrix[x][y] = 3 - matrix[x][y]  # Flip 1 <-> 2
    print("after applying mask pattern 0")
    return matrix


def apply_mask1(matrix, data_bit_positions):
    
    # row % 2 == 0
    for (x, y) in data_bit_positions:
        if x % 2 == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 1")
    return matrix
    


def apply_mask2(matrix, data_bit_positions):
    # col % 3 == 0
    for (x, y) in data_bit_positions:
        if y % 3 == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 2")
    return matrix

def apply_mask3(matrix, data_bit_positions):
    # (row + col) % 3 == 0
    for (x, y) in data_bit_positions:
        if (x + y) % 3 == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 3")
    return matrix

def apply_mask4(matrix, data_bit_positions):
    # (floor(row/2) + floor(col/3)) % 2 == 0
    for (x, y) in data_bit_positions:
        if ((x // 2) + (y // 3)) % 2 == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 4")
    return matrix

def apply_mask5(matrix, data_bit_positions):
    # ((row * col) % 2 + (row * col) % 3) == 0
    for (x, y) in data_bit_positions:
        if ((x * y) % 2 + (x * y) % 3) == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 5")
    return matrix

def apply_mask6(matrix, data_bit_positions):
    # (((row * col) % 2 + (row * col) % 3) % 2) == 0
    for (x, y) in data_bit_positions:
        if (((x * y) % 2 + (x * y) % 3) % 2) == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 6")
    return matrix

def apply_mask7(matrix, data_bit_positions):
   
    # (((row + col) % 2 + (row * col) % 3) % 2) == 0
    for (x, y) in data_bit_positions:
        if (((x + y) % 2 + (x * y) % 3) % 2) == 0:
            matrix[x][y] = 3 - matrix[x][y]
    print("after applying mask pattern 7")
    return matrix
