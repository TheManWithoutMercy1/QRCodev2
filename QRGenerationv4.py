from Qrpt2v2 import data_words
from Qrpt2v2 import integrate_reed_solo
from matrix2 import create_matrix, place_data_bits, apply_mask, format_info, determine_mask3, determine_mask_1, determine_mask_2,determine_mask4,lowest_penalty,save_qr_image
from image import create_qr_image
import os
class QR_Generation:
    def __init__(self, input_string):
        self.string = input_string
        self.bit_string = ''  # Final bit string to be generated
        self.required_bits = None
        self.version = None
    
    def encoding_string(self):
        """
        Encode string into byte mode bits (Version 1-9).
        """
        length = len(self.string)
        if length <= 32:
            self.version = 2
            self.required_bits = 272
        else:
            print("input too long, only Versions 1 and 2 available.")
        byte_mode = '0100'  # Byte mode indicator
        print(f"Mode indicator: {byte_mode}")

        char_count = len(self.string)
        print(f"Character count: {char_count}")

        # Convert character count to 8-bit binary
        char_count_bin = format(char_count, '08b')
        print(f"Character count in binary (8-bit padded): {char_count_bin}")

        self.bit_string = byte_mode + char_count_bin  # Start bit string

    def byte_mode_encoding(self):
        """
        Convert each character to its 8-bit binary representation.
        """
        print(f"Original string: {self.string}")
        binary_data = ''.join(format(ord(i), '08b') for i in self.string)
        self.bit_string += binary_data

        print("Character to Binary Conversion:")
        for char in self.string:
            char_binary = format(ord(char), '08b')
            print(f"{char} → {char_binary}")

        print(f"Bit string after encoding data: {self.bit_string}")

    def determine_bits(self):
        """
        Calculate total bit length and check against required capacity.
        """
        num_total_bits = len(self.bit_string)
        print(f"Total bits so far: {num_total_bits}")
        if num_total_bits < self.required_bits:
            print(f"Bit string is {self.required_bits - num_total_bits} bits too short. Padding needed.")
        else:
            print("No padding needed. Bit string meets required length.")

    @staticmethod
    def add_terminators(bit_string, num_total_bits, required_bits):
        diff = required_bits - num_total_bits
        if diff <= 0:
            print("No terminator needed.")
            return bit_string
        elif diff >= 4:
            print("Adding 4-bit terminator.")
            return bit_string + '0000'
        else:
            print(f"Adding {diff}-bit terminator.")
            return bit_string + ('0' * diff)

    @staticmethod
    def add_zero_multiple_8(bit_string):
        remainder = len(bit_string) % 8
        if remainder == 0:
            print("Bit string already a multiple of 8.")
            return bit_string
        padding = 8 - remainder
        print(f"Adding {padding} zero(s) to align to byte boundary.")
        return bit_string + ('0' * padding)

    @staticmethod
    def pad_bytes(bit_string,required_bits):
        pad1 = '11101100'
        pad2 = '00010001'
        toggle = True

        while len(bit_string) + 8 <= required_bits:
            bit_string += pad1 if toggle else pad2
            toggle = not toggle

        print(f"Padded to final bit length: {len(bit_string)}")
        return bit_string

    def generate_final_bit_stream(self):
        """
        Full end-to-end bit stream generation for the input string.
        """
        self.encoding_string()
        self.byte_mode_encoding()
        self.determine_bits()
        # Add terminators
        self.bit_string = self.add_terminators(self.bit_string, len(self.bit_string), self.required_bits)
        # Make bit string a multiple of 8
        self.bit_string = self.add_zero_multiple_8(self.bit_string)
        # Add pad bytes if necessary
        self.bit_string = self.pad_bytes(self.bit_string,self.required_bits)
        print("Final Bit String:")
        print(self.bit_string)
        print(f"Total length: {len(self.bit_string)} bits")
   
        data_words_list = data_words(self.bit_string)
        ir = integrate_reed_solo(data_words_list,self.version)
        matrix = create_matrix(self.version)
        matrix2, data_bit_positions = place_data_bits(matrix, ir)
        matrix3,best_mask,format_bits = lowest_penalty(matrix2, data_bit_positions)
        matrix = format_info(matrix3,format_bits)
        save_qr_image(matrix, pixel_size=10, file_name="my_qr.png")




       # penalty1 = determine_mask_1(matrix2)
       # penalty2 = determine_mask_2(matrix2)
       # penalty3 = determine_mask3(matrix2)
       # penalty4 = determine_mask4(matrix)
       # total_penalty = penalty1+penalty2+penalty3+penalty4
       # print("the total penalty score is: ", total_penalty)

        #matrix4 = format_info(matrix3)
        #img = create_qr_image(matrix4)
        #downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        #img_path = (os.path.join(downloads_path, "qr_code.png"))
        #img.save(img_path)
        #img.show()  # Show the image for verification
        #return img_path

        #return self.bit_string
    
# Example usagegenerate_final_bit_stream()
string = input("what string do you want to put?")
qr_gen = QR_Generation(string)
qr_gen.generate_final_bit_stream()
