import reedsolo as rs

def data_words(binarystream):
    print("in data words function")
    data_words_list = []
    # Split the bitstream into chunks of 8 bits
    for i in range(0, len(binarystream), 8):
        # Take each 8-bit chunk and convert to integer
        word = binarystream[i:i+8]
        data_words_list.append(word)  # Store binary strings    
    print("Data words:", data_words_list)
    return data_words_list


def integrate_reed_solo(data_words_list, version):
    print("integrating reed solo..")
    # Convert binary strings to integers (bytes)
    data_bytes = [int(byte, 2) for byte in data_words_list]

    num_ec_codewords = 10  # Version 2-L needs 10 EC codewords

    rs_codec = rs.RSCodec(num_ec_codewords)

    full_encoded = rs_codec.encode(data_bytes)

    # Correct: last 10 bytes are the error correction codewords
    ec_bytes = full_encoded[-num_ec_codewords:]

    # Final codeword sequence = original data + parity bytes
    final_bytes = data_bytes + list(ec_bytes)

    # Convert back to binary strings
    encoded_binary = [format(byte, '08b') for byte in final_bytes]

    print(f"Final encoded (data + EC): {encoded_binary}")
    return encoded_binary
