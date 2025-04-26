import numpy as np
from image import create_qr_image 
from applying_masks import*
import os
def create_matrix(version):
    matrix = np.zeros((25,25))
    print("initial matrix:")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)
    add_finder_patterns(matrix,version)
    return matrix
   # return matrix

def add_finder_patterns(matrix, version):
    # Calculate the size of the QR code based on the version
    size = ((version - 1) * 4) + 21

    # Top-left finder pattern (always at (0,0))
    for y in range(7):
        for x in range(7):
            if (x in [0, 6] or y in [0, 6]) or (2 <= x <= 4 and 2 <= y <= 4):
                matrix[y][x] = 2  # black module
            else:
                matrix[y][x] = 1  # white module
    
    # Draw the top-right finder pattern starting at (0, 17)
    for y in range(7):
     for x in range(7):
        if (x in [0, 6] or y in [0, 6]) or (2 <= x <= 4 and 2 <= y <= 4):
            matrix[y][18 + x] = 2  # black
        else:
            matrix[y][18 + x] = 1  # white

    # Bottom-left finder pattern starting at (18, 0)
    for y in range(7):
     for x in range(7):
        if (x in [0, 6] or y in [0, 6]) or (2 <= x <= 4 and 2 <= y <= 4):
            matrix[18 + y][x] = 2  # black
        else:
            matrix[18 + y][x] = 1  # white


    # Call any additional functions (e.g., adding separators)
    print("matrix with finder patterns: ")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)

    add_seperators(matrix,version)
    # If applicable, add alignment patterns, etc.

def add_seperators(matrix, version):
    """
    Separators are lines of white modules, one module wide, that are placed beside the finder patterns
    to separate them from the rest of the QR code. The separators are only placed beside the edges of the finder patterns
    that touch the inside of the QR code.
    """
    # Calculate the size of the QR code based on the version
    size = ((version - 1) * 4) + 21

    # Top-left separator (between finder and the rest of the QR code)
    for row in range(8):  # This will cover the top-left finder pattern's right side and bottom side
        matrix[row][7] = 1  # Vertical separator
        matrix[7][row] = 1  # Horizontal separator
    for row in range(8):
        matrix[row][17] = 1
    for row in range(17,24):
        matrix[7][row] = 1
   # matrix[7][17] = 1

    for row in range(18,25):
        matrix[row][7] = 1

    for row in range(8):
      matrix[17][row] = 1
     
    print("With separators:")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)

    
    add_alignment_patterns(matrix,version)

    #add_timing_patterns(matrix,version)
def add_alignment_patterns(matrix, version):
    """
    Adds alignment patterns to the QR code matrix.
    For version 2, only one alignment pattern is needed at (18, 18).
    """
    alignment_positions = [6, 18]

        # Avoid overlapping with finder patterns at (6,6), (6,18), and (18,6)
    for cy in alignment_positions:
            for cx in alignment_positions:
                # Skip corners where finder patterns are already placed
                if (cy == 6 and cx == 6) or (cy == 6 and cx == 18) or (cy == 18 and cx == 6):
                    continue

                # Draw 5x5 alignment pattern centered at (cy, cx)
                for y in range(cy - 2, cy + 3):
                    for x in range(cx - 2, cx + 3):
                        if x == cx - 2 or x == cx + 2 or y == cy - 2 or y == cy + 2:
                            matrix[y][x] = 2  # Outer black border
                        elif x == cx and y == cy:
                            matrix[y][x] = 2  # Center black dot
                        else:
                            matrix[y][x] = 1  # Inner white

    print("Matrix with alignment patterns added (version 2):")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)
    add_timing_patterns(matrix,version)


def add_timing_patterns(matrix,version):
    """
    The timing patterns are two lines of alternating dark and light modules.
    They should extend across the entire QR code.
    """
    for i in range(8, 17):
     matrix[6][i] = 2 if (i % 2 == 0) else 1  # Horizontal timing pattern

    for i in range(8,17):
      matrix[i][6] = 2 if (i % 2 == 0) else 1 
    print("matrix with timing patterns")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)
    reserve_format_information(matrix,version)


def reserve_format_information(matrix,version):
    """
    Reserve format information areas and place the dark module.
    The dark module is always at (8, (4 * version) + 9).
    """
    # Place the dark module
    matrix[17][8] = 2
    for i in range(18,25):
        matrix[i][8] = 3
    for i in range(6):
       matrix[8][i] = 3
    matrix[8][7] = 3
    matrix[8][8] = 3

    for i in range(6):
      matrix[i][8] = 3
    matrix[7][8] = 3
    matrix[8][8] = 3
    
    for i in range(17,25):
      matrix[8][i] = 3

    print("Matrix with reserved format info and dark module:")
    print(matrix)
    for row in matrix:
      print(row.tolist())  # or just print(row)
    return matrix
    
def place_data_bits(matrix, bitstream):
    data_bit_positions = []
    converted_bits = []

    # Convert bitstream to 1 (white) and 2 (black)
    for byte in bitstream:
        for bit in byte:
            converted_bits.append(1 if bit == '0' else 2)

    size = len(matrix)
    y = size - 1
    direction_up = True

    while y > 0 and converted_bits:
        if y == 6:
            y -= 1

        for dx in [0, -1]:
            col = y + dx
            if col < 0 or col >= size:
                continue

            row_range = range(size - 1, -1, -1) if direction_up else range(size)
            for row in row_range:
                if matrix[row][col] >= 1:
                    continue
                if not converted_bits:
                    break
                matrix[row][col] = converted_bits.pop(0)
                data_bit_positions.append((row, col))  # fixed here

        y -= 2
        direction_up = not direction_up

    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 0:
                matrix[i][j] = 1

    print("after data bits")
    print("Remaining bits:", len(converted_bits))
    for row in matrix:
        print(row)
    return matrix, data_bit_positions

import copy
import copy

def lowest_penalty(matrix, data_bit_positions):
    best_mask = None
    lowest_score = float('inf')
    best_matrix = None
    best_format_bits = None

    format_bits_map = {
        0: "111011111000100",
        1: "111001011110011",
        2: "111110110101010",
        3: "111100010011101",
        4: "110011000101111",
        5: "110001100011000",
        6: "110110001000001",
        7: "110100101110110"
    }
    
    for i in range(8):
        test_matrix = copy.deepcopy(matrix)
        mask_func = globals()[f'apply_mask{i}']
        test_matrix = mask_func(test_matrix, data_bit_positions)

        p1 = determine_mask_1(test_matrix)
        p2 = determine_mask_2(test_matrix)
        p3 = determine_mask3(test_matrix)
        p4 = determine_mask4(test_matrix)
        total_penalty = p1 + p2 + p3 + p4

        print(f"Mask {i} penalty score: {total_penalty}")

        if total_penalty < lowest_score:
            lowest_score = total_penalty
            best_mask = i
            best_matrix = test_matrix
            best_format_bits = format_bits_map[i]

    print(f"\nBest mask pattern is: {best_mask} with penalty score: {lowest_score}")
    print(f"Format bits to embed: {best_format_bits}")
    return best_matrix, best_mask, best_format_bits


def determine_mask_1(matrix):
    total_penalty = 0
    
    # Horizontal check
    for i in range(25):  # Iterate over each row (25 rows)
        j = 0
        while j < 21:  # Only check up to column 21 (because we need at least 5 elements for a pattern)
            consecutive_elements = matrix[i][j:j+5]  # Get 5 consecutive elements
            if all(x == consecutive_elements[0] for x in consecutive_elements):  # Check if they are all the same
                total_penalty += 3  # Add penalty for 5 consecutive elements
                
                # Check for additional consecutive elements
                k = j + 5
                while k < 25 and matrix[i][k] == consecutive_elements[0]:
                    total_penalty += 1  # Add penalty for each additional consecutive element
                    k += 1
                
               # print(f"Horizontal: Found consecutive elements starting at row {i}, column {j}: {consecutive_elements}")
                
                # Move to the next non-overlapping pattern by skipping the current consecutive sequence
                j = k  # Skip over the consecutive sequence to avoid overlap
            else:
                j += 1  # Move to the next column if no pattern is found
    
    # Vertical check
    for j in range(25):  # Iterate over each column (25 columns)
        i = 0
        while i < 21:  # Only check up to row 21 (because we need at least 5 elements for a pattern)
            consecutive_elements = [matrix[i+k][j] for k in range(5)]  # Get 5 consecutive elements in the column
            if all(x == consecutive_elements[0] for x in consecutive_elements):  # Check if they are all the same
                total_penalty += 3  # Add penalty for 5 consecutive elements
                
                # Check for additional consecutive elements
                k = i + 5
                while k < 25 and matrix[k][j] == consecutive_elements[0]:
                    total_penalty += 1  # Add penalty for each additional consecutive element
                    k += 1
                
                #print(f"Vertical: Found consecutive elements starting at row {i}, column {j}: {consecutive_elements}")
                
                # Move to the next non-overlapping pattern by skipping the current consecutive sequence
                i = k  # Skip over the consecutive sequence to avoid overlap
            else:
                i += 1  # Move to the next row if no pattern is found

    print(f"Total penalty: {total_penalty}")
    return total_penalty

   


def determine_mask_2(matrix):
    """
    For Second Evaluation Condition , look for areas of the same colour that are at least 2x2 modules or larger,
    The QR code specification says that for a solid-color block of size m*m the penalty score is 3 x (m - 1) x (n -1)

    However, the QR code specification does not specify how to calculate the penalty when there are multiple ways of dividing up the solid-color blocks.

    Therefore, rather than looking for solid-color blocks larger than 2x2,
    simply add 3 to the penalty score for every 2x2 block of the same color in the QR code, making sure to count overlapping 2x2 blocks. 
    For example,a 3x2 block of the same color should be counted as two 2x2 blocks, one overlapping the other.
    """
    penalty = 0
    # Iterate over the matrix to find 2x2 blocks
    for i in range(24):  # Only go up to 24 (because 25th row/column will not have enough room for 2x2 block)
        for j in range(24):  # Similarly, only go up to 24 columns
            # Check if the 2x2 block starting at (i, j) is all the same color
            if (matrix[i][j] == matrix[i+1][j] == matrix[i][j+1] == matrix[i+1][j+1]):
                penalty += 3  # Add 3 for each valid 2x2 block
               # print(f"2x2 block found at ({i}, {j})")

    print(f"Total penalty for 2x2 blocks: {penalty}")
    return penalty
    

def determine_mask3(matrix):
    total_penalty = 0
    # Check horizontally (in rows)
    for i in range(25):  # Iterate over each row (25 is the size of the matrix)
        for j in range(19):  # Check for patterns starting at column 0 to column 18 (because the pattern is 7 elements long)
            # Check if the pattern matches the "2 1 2 2 2 1 2" or "1 1 1 1 2 1 2" patterns
            horizontal_pattern = matrix[i][j:j+7]  # Slice out 7 consecutive elements from the row
            
            # Use np.array_equal to compare arrays if using NumPy arrays, or simple list comparison otherwise
            if np.array_equal(horizontal_pattern, [2, 1, 2, 2, 2, 1, 2]) or np.array_equal(horizontal_pattern, [1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2]):
                total_penalty += 40  # Add 40 to the penalty for each match
               # print(f"Horizontal pattern found at row {i}, starting column {j}")
    
    # Check vertically (in columns)
    for j in range(25):  # Iterate over each column (25 is the size of the matrix)
        for i in range(19):  # Check for patterns starting at row 0 to row 18 (because the pattern is 7 elements long)
            # Check if the pattern matches the "2 1 2 2 2 1 2" or "1 1 1 1 2 1 2" patterns
            vertical_pattern = [matrix[i+k][j] for k in range(7)]  # Get 7 consecutive elements in the column
            
            # Use np.array_equal to compare arrays if using NumPy arrays, or simple list comparison otherwise
            if np.array_equal(vertical_pattern, [2, 1, 2, 2, 2, 1, 2]) or np.array_equal(vertical_pattern, [1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2]):
                total_penalty += 40  # Add 40 to the penalty for each match
                #print(f"Vertical pattern found at column {j}, starting row {i}")
    
    print(f"Total penalty for third rule: {total_penalty}")
    return total_penalty


def determine_mask4(matrix):
    """
    The Final evaluation condition is based on the ratio of light modules to dark modules to calculate this penalty rule.
    """
    num_of_modules = 0
    dark_modules = 0
    
    # Count the total number of modules in the matrix
    for row in matrix:
        num_of_modules += len(row)
    
    # Count the dark modules (assuming dark modules are represented by 2)
    dark_modules = np.sum(matrix == 2)

    print("num of modules in matrix is:", num_of_modules)
    print("num of dark modules is:", dark_modules)

    # Calculate the percentage of dark modules
    dark_percent = (dark_modules / num_of_modules) * 100
    print("percent of modules in the matrix that are dark is:", dark_percent)

    # Step 4: Determine previous and next multiples of 5
    prev_multiple_of_5 = (dark_percent // 5) * 5
    next_multiple_of_5 = prev_multiple_of_5 + 5 if dark_percent % 5 != 0 else prev_multiple_of_5
    
    print("Previous multiple of 5:", prev_multiple_of_5)
    print("Next multiple of 5:", next_multiple_of_5)

    # Step 5: Subtract 50 from each of these multiples of five
    prev_multiple_of_5 = abs(prev_multiple_of_5 - 50)
    next_multiple_of_5 = abs(next_multiple_of_5 - 50)
    
    print("After subtracting 50 and taking absolute value:")
    print("Previous multiple of 5:", prev_multiple_of_5)
    print("Next multiple of 5:", next_multiple_of_5)

    # Step 6: Divide each by 5
    prev_multiple_of_5_divided = prev_multiple_of_5 / 5
    next_multiple_of_5_divided = next_multiple_of_5 / 5

    print("Divided by 5:")
    print("Previous multiple of 5:", prev_multiple_of_5_divided)
    print("Next multiple of 5:", next_multiple_of_5_divided)

    # Step 7: Take the smallest of the two numbers and multiply by 10
    penalty_score = min(prev_multiple_of_5_divided, next_multiple_of_5_divided) * 10

    print("Penalty score #4:", penalty_score)
    return penalty_score



def apply_mask(matrix,data_bit_positions):
   """
   apply one masking pattern (pattern 0)
   to reduce large blocks of the same coloured modules

   mask number 0 has formula (row + column) mod 2 == 0
   Each mask pattern uses a formula to determine whether or not to change the color of the current bit. 
   You put the coordinates of the current bit into the formula, and if the result is 0, you use the opposite bit at that coordinate. 
   For example, if the bit for coordinate (0,3) is 1, and the formula is equal to 0 for that coordinate, 
   then you put a 0 at (0,3) instead of a 1.
   """
   determine_mask_1(matrix)
   return matrix

def format_info(matrix, format_bits):
    def format_value(bit):
        return 2 if bit == '1' else 1

    # Place format bits 0–6 in column 8 from row 24 to 18
    for i in range(7):  # i = 0 to 6
        row = 24 - i
        matrix[row][8] = format_value(format_bits[i])
   
   # for i in range(6)
    count = 0
    for i in range(6):
          matrix[8][i] = format_value(format_bits[count])
          count += 1
    matrix[8][7] = format_value(format_bits[6])
    matrix[8][8] = format_value(format_bits[7])
    matrix[7][8] = format_value(format_bits[8])

    #move from 5 to 1/0
    count2 = 9
    for i in range(5, -1, -1):  # From 5 down to 0
       matrix[i][8] =  format_value(format_bits[count2])
       count += 1

    count3 = 7
    for i in range(17,25):
        matrix[8][i] =  format_value(format_bits[count3])
        count3 += 1

    # Print matrix for debug
    for row in matrix:
        print(row.tolist())


    return matrix


from PIL import Image

def save_qr_image(matrix, pixel_size=10, file_name="qr_output.png", quiet_zone=4):
    """
    Converts a QR matrix (with 1 for white, 2 for black) into a PNG image with a quiet zone.

    :param matrix: 2D list with values 1 (white) and 2 (black)
    :param pixel_size: Size of each QR module (square)
    :param file_name: Output file name
    :param quiet_zone: Number of white modules to pad around the QR
    """
    size = len(matrix)
    padded_size = size + quiet_zone * 2  # Total image size with borders

    # Create a white image
    img = Image.new("RGB", (padded_size * pixel_size, padded_size * pixel_size), "white")
    pixels = img.load()

    # Fill in the QR code content
    for i in range(size):
        for j in range(size):
            color = (255, 255, 255) if matrix[i][j] == 1 else (0, 0, 0)
            for x in range(pixel_size):
                for y in range(pixel_size):
                    px = (j + quiet_zone) * pixel_size + x
                    py = (i + quiet_zone) * pixel_size + y
                    pixels[px, py] = color

    img.save(file_name)
    print(f"QR code with quiet zone saved to {file_name}")
    img.show()  
