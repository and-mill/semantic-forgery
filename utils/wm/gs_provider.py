"""
Original by https://github.com/lthero-big/A-watermark-for-Diffusion-Models and heavily modified for debugging purposes and getting access to internals.
Please give them credit and adhere to their license agreement.
"""

import typing

import argparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend

import numpy as np
from scipy.stats import norm

import torch

from PIL import Image

from .wm_provider import WmProvider

from utils.image_utils import torch_to_PIL



parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--l', default=1, type=int, help="The size of slide windows for m")
parser.add_argument('--num_replications', default=64, type=int, help="The number of replications of the message bits to get barcode image")
parser.add_argument('--message_width_in_bytes', default=32, type=int, help="Message width in bytes")



class GsProvider(WmProvider):
    """
    Original by https://github.com/lthero-big/A-watermark-for-Diffusion-Models and heavily modified for debugging and getting access to internals.
    """

    def __init__(self,
                 message_width_in_bytes: int = 32,  # (channels/f_c) * f_h * f_w // 8
                 num_replications: int = 64,
                 l: int = 1,
                 offset: int = 0,
                 message: typing.Optional[str] = None,
                 key: str = None,
                 nonce: str = None,
                 **kwargs):
        """
        This provider uses a fixed list of keys, messages, and nonces for the watermarking process to eliminate every possibliy of false negatives due to wrong seeds.
        Generate more with "generate_secrets.py" as needed.

        These are the params for the the watermark:
        - message_width_in_bytes = Num users to distinguish
        - num_replications = strength of error correction
        - l = window size, aka the leeway given for errors in reconstructing original noise vector and it counteracts the error correction

        The "barcode" refers to the barcode image (zeros and ones, arranged in columns) that we have right before encrypting (during the watermark-generating stage) / right after decrypting (verifying a watermark stage).
        During verification, this gives a good vidual represenation of error introduced by the inversion process. See Section A in our Appendix, where we explain Gaussian Shading in detail.

        The barcode dimensions flattened divided by l must match the latent dimension flattened:
        
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        -> message_width_in_bytes * 8 * num_replications / l == num_channels * latent_resolution * latent_resolution
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        Example:
        -> 32 (message_width_in_bytes) * 8 * 64 (num_replication) / 1 (l) = 16384 = 4 (num_channels) * 64 (latent_resolution) * 64 (latent_resolution)

        @param message_width_in_bytes: width of the message in bytes
        @param num_replications: num_replications
        @param l: l
        @param offset: offset in the list of messages, keys, and nonces
        @param message: message to use all batch_size. This is useful if you want to simulate a user generating mutiple images.
        @param key: key to use all batch_size
        @param nonce: nonce to use for all batch_size
        """
        super().__init__(**kwargs)

        self.message_width_in_bytes = message_width_in_bytes
        self.message_width_in_bits = int(message_width_in_bytes * 8)
        self.barcode_width = self.message_width_in_bits

        self.num_replications = num_replications
        self.l = l
        self.offset = offset  # the exact messages, keys, nonces used when calling get_wm_latents are decided by batch_size starting from offset

        assert self.message_width_in_bits * self.num_replications // self.l == self.num_channels * self.latent_resolution * self.latent_resolution

        # use a predefined list of crypto params
        if message is None:
            from .messages import MESSAGES
            MESSAGES = [m[:self.message_width_in_bytes] for m in MESSAGES]  # trim to message_width_in_bytes to fit our required message length
            from .keys import KEYS
            from .nonces import NONCES
            self.messages = MESSAGES
            self.keys = KEYS
            self.nonces = NONCES

        # if only one param is given, just reuse it every time
        else:
            assert key is not None and nonce is not None
            self.messages = [message for _ in range(self.batch_size)]
            self.keys = [key for _ in range(self.batch_size)]
            self.nonces = [nonce for _ in range(self.batch_size)]


    def get_wm_type(self) -> str:
        return "GS"


    def __character_str_to_bytes(self,
                                 message: str) -> bytes:
        """
        Convert a string to a byte string

        @param message: message to convert
        @param padded_message_length: length of the byte string

        @return: byte string
        """
        # Convert the message to a byte string
        message_bytes = message.encode()

        # Ensure the encoded message is 256 bits (32 bytes)
        if len(message_bytes) < self.message_width_in_bytes:
            padded_message = message_bytes + b'\x00' * (self.message_width_in_bytes - len(message_bytes))
        else:
            padded_message = message_bytes[:self.message_width_in_bytes]
        return padded_message

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, any]:
        """
        Get Watermarked latents and barcodes
                        
        @return: dict
        """
        # iterate message, keys, nonces
        latents_torch = []
        barcodes_torch = []
        message_bits_str_list = []
        for i in range(self.offset,
                       self.batch_size + self.offset):
            # get the message, key and nonce
            message_bytes = self.messages[i]  # message_bytes = k
            # remember the message bits as string
            message_bits_str_list.append(''.join(format(byte, '08b') for byte in self.messages[i]))

            key_bytes = self.keys[i]
            nonce_bytes = self.nonces[i]

            # ----------------------------------------------------- STEP 1: Replicate the message to get a barcode -----------------------------------------------------
            replicated_message_bytes = message_bytes * self.num_replications  # replicated_message_bytes = s_d, 2048 bytes in old setup
            replicated_message_bits_str = ''.join(format(byte, '08b') for byte in replicated_message_bytes)  # replicated_message_bits_flat = s_d in binary, 16384 bits in old setup
            # Make barcode as np from s_d
            barcode_ints_2d_np = np.array([int(b, 2) for b in replicated_message_bits_str]).reshape(self.num_replications, self.message_width_in_bits)
            barcode_ints_2d_torch = torch.tensor(barcode_ints_2d_np, dtype=torch.uint8, device=self.device)
            barcodes_torch.append(barcode_ints_2d_torch)

            # ------------------------------------------------------------- STEP 2: Encrypt the barcode -------------------------------------------------------------
            # set up cipher
            cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted_bytes = encryptor.update(replicated_message_bytes) + encryptor.finalize()  # encrypted_bytes = m
            encrypted_bits_str = ''.join(format(byte, '08b') for byte in encrypted_bytes)  # encrypted_bits_str = m_bits
    
            # ------------------------------------- STEP 3: Embed the encrypted message into the latent space via Gaussian sampling -------------------------------------
            # sampling from uniform into gaussian
            # Traverse the binary representation of m, cutting according to window size l
            pixel_values = []
            for i in range(0, len(encrypted_bits_str), self.l):
                window = encrypted_bits_str[i:i + self.l]
                y = int(window, 2)  # Convert the binary sequence (base 2) inside the window into integer y
                u = np.random.uniform(0, 1)  # Generate random u
                # Calculate z^s_T
                pixel_value = norm.ppf((u + y) / 2**self.l)
                pixel_values.append(pixel_value)
            pixel_values = torch.tensor(pixel_values, dtype=self.dtype, device=self.device)
            latent_torch = pixel_values.reshape(self.latent_shape[1:])  # latent shape without batch dim
            latents_torch.append(latent_torch)
        
        # finalize
        latents_torch = torch.stack(latents_torch, dim=0)
        barcodes_torch = torch.stack(barcodes_torch, axis=0)

        latents_PIL = torch_to_PIL(latents_torch)

        barcodes_PIL = torch_to_PIL(barcodes_torch)

        results_dict = {"zT_torch": latents_torch,
                        "zT_PIL": latents_PIL,
                        "zT": latents_PIL,
                        "barcodes_torch": barcodes_torch,
                        "barcodes_PIL": barcodes_PIL,
                        "barcodes": barcodes_PIL,
                        "message_bits_str_list": message_bits_str_list
                        }
    
        return results_dict
    

    def __recover_messages_from_latents(self,
                                        latents: torch.Tensor) -> typing.List[typing.Union[str, Image.Image]]:
        """
        Recover messages from latents

        @param latents: latent either tensor with batch dim
        @return: dict
        """
        # lets work on numpies only
        latents = latents.cpu().numpy()

        # iterate keys, nonces
        recovered_messages_bits_str = []
        recovered_barcodes_torch = []
        for i in range(self.offset,
                       self.batch_size + self.offset):
            # get the message, key and nonce
            key_bytes = self.keys[i]
            nonce_bytes = self.nonces[i]
            latent = latents[i - self.offset]  # This is ugly
        
            # ------------------------- STEP 3 reverse: Extract the encrypted message from the latent space via reverse Gaussian Sampling -------------------------
            # Reconstruct m from reversed_latents
            encrypted_numbers = []
            for zsT_pixel_value in np.nditer(latent):
                # Use the inverse operation of norm.ppf to recover the original y value
                y_reconstructed = norm.cdf(zsT_pixel_value) * 2**self.l
                y_reconstructed = int(y_reconstructed)
                # in some very rare and sporadic cases, y_reconstructed can be a 2 for l=1.
                if y_reconstructed == 2**self.l:
                    y_reconstructed -= 1
                encrypted_numbers.append(y_reconstructed)
    
            # For l > 1 we need another step that unfold number up to 2**l -1 to binary strings
            encrypted_bits = []
            for n in encrypted_numbers:
                # Convert the integer to binary and pad with zeros
                encrypted_bits.append(bin(n)[2:].zfill(self.l))
            encrypted_bits_str = ''.join(encrypted_bits)  # encrypted_bits_str = reconstructed_m_bits

            # Convert binary bits to bytes
            # This is hell for my smooth brain
            encrypted_bytes = bytes(
            int(
                    ''.join(
                        str(bit) for bit in encrypted_bits_str[j:j+8]
                        ),
                        2
                    ) for j in range(0, len(encrypted_bits_str), 8)
            )
    
            # ---------------------------------------- STEP 2 reverse: Decrypt the message to retrieve barcode ----------------------------------------
            # Decrypt m to recover the data s_d before diffusion
            # initiate the Cipher
            cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            replicated_message_bytes = decryptor.update(encrypted_bytes) + decryptor.finalize()  # replicated_message_bytes = s_d_reconstructed
            
            # Convert the decrypted byte string into a binary representation
            replicated_message_bits_str = ['{:08b}'.format(byte) for byte in replicated_message_bytes]  # replicated_message_bits_str = bits_list
            # Merge the binary strings into a long string
            all_bits = ''.join(replicated_message_bits_str)
            # list of strings with len self.message_width_in_bits
            barcode_string_1d = [all_bits[i:i+self.message_width_in_bits] for i in range(0,
                                                                                         len(all_bits),
                                                                                         self.message_width_in_bits)]  # barcode_string_2d = segments
        
            # make torch
            # 1st dim: number of elements in barcode_string_1d
            # 2nd dim: number of character ("0" or "1") in each element, these are self.message_width_in_bits
            barcode_int_2d_torch = torch.tensor([list(map(int, list(row))) for row in barcode_string_1d], dtype=torch.uint8, device=self.device)
    
            # ---------------------------------------- STEP 1 reverse: Recover the original message from the barcode -----------------------------------
            # Voting mechanism to determine each bit
            message_bits_str = ''
            for i in range(self.message_width_in_bits):
                # Calculate the count of '1's for each bit across all lines
                count_1 = sum(column[i] == '1' for column in barcode_string_1d)
                # for l>1, we might get sporadic errors becasue of numerical
                message_bits_str += '1' if count_1 > len(barcode_string_1d) / 2 else '0'

            # append
            recovered_messages_bits_str.append(message_bits_str)
            recovered_barcodes_torch.append(barcode_int_2d_torch)

        # finalize
        recovered_barcodes_torch = torch.stack(recovered_barcodes_torch, dim=0)
        recovered_barcodes_PIL = torch_to_PIL(recovered_barcodes_torch)

        return {"messages_bits_str": recovered_messages_bits_str,
                "barcodes_torch": recovered_barcodes_torch,
                "barcodes_PIL": recovered_barcodes_PIL}
    

    def __calculate_bit_accuracy(self,
                                 original_message_hex: any,  # no idea what datatype
                                 extracted_message_bits_str: str) -> float:
        """
        Gett bit accuracy between extracted bits and the original message hex

        @param original_message_hex: original message in hex
        @param extracted_message_bits_str: extracted message in bits

        @return: bit accuracy
        """
        # Convert the original hex message to binary
        original_message_bin = ''.join(format(byte, '08b') for byte in original_message_hex)

        # Ensure both binary strings are of the same length
        min_length = min(len(original_message_bin), len(extracted_message_bits_str))
        original_message_bin = original_message_bin[:min_length]
        extracted_message_bits_str = extracted_message_bits_str[:min_length]
        
        # Calculate bit accuracy
        matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bits_str) if x == y)
        bit_accuracy = matching_bits / min_length
        
        return bit_accuracy
    

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.array]) -> typing.Dict[str, any]:
        """
        Get bit accuracy between original and extracted messages

        @param latents: latent either tensor with batch dim or numpy with batch dim
        @return: dict
        """
        # get the extracted message
        recovered = self.__recover_messages_from_latents(latents)
        recovered_messages_bits_str = recovered["messages_bits_str"]
        recovered_barcodes_torch = recovered["barcodes_torch"]
        recovered_barcodes_PIL = recovered["barcodes_PIL"]

        # iterate and calulate bit accuracies
        bit_accuracies = []
        recovered_message_bits_str_list = []
        for i in range(self.offset, self.batch_size + self.offset):
            original_message_hex = self.messages[i]
            recovered_message_bits_str = recovered_messages_bits_str[i - self.offset]

            # get the bit accuracy
            bit_accuracy = self.__calculate_bit_accuracy(original_message_hex, recovered_message_bits_str)
            bit_accuracies.append(bit_accuracy)
            recovered_message_bits_str_list.append(recovered_message_bits_str)

        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "barcodes_torch": recovered_barcodes_torch,
            "barcodes_PIL": recovered_barcodes_PIL,
            "barcodes": recovered_barcodes_PIL,
            "message_bits_str_list": recovered_message_bits_str_list
        }
    

    def wiggle_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Resample latents

        @param latents: latent tensor with batch dim
        @return: torch.Tensor with batch dim 
        """

        # reverse sampling back to barcode pixels in [0, 2**self.l - 1]
        latents = latents.detach().cpu().numpy()
        latents = norm.cdf(latents) * 2**self.l
        latents = latents.astype(np.int32)
        # fix bug where we sometimes get 2**l
        latents[latents == 2**self.l] = 2**self.l - 1
        # latents is now integers in [0, 2**self.l - 1]

        # forward sampling with randomnes
        # y we already have
        y = latents
        # u we draw
        u = np.random.uniform(low=0, high=1, size=y.shape).astype(np.float32)
        # sampling a gaussian
        new_latent = norm.ppf((u + y) / 2**self.l)
        
        return torch.tensor(new_latent, dtype=self.dtype, device=self.device)
