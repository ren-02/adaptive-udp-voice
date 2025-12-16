"""
Adaptive Voice Link Simulation
==============================

A real-time UDP voice application that simulates a Physical Layer (PHY)
transmission chain including:
- Forward Error Correction (Convolutional Coding K=3, R=1/2)
- Modulation (16-QAM)
- Channel Simulation (Rayleigh Fading + AWGN)
- Adaptive Coding (Viterbi Decoding) based on SNR feedback.

Author: Vincent Rene U. Arce
License: MIT
"""

import argparse
import asyncio
import json
import socket
import struct
import threading
import zlib
import numpy as np
import sounddevice as sd
import math

# ====================== CONFIG ======================
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
UDP_PORT = 8765
JITTER_BUFFER_MS = 80
AUDIO_DTYPE = np.int16
AUDIO_MAX_VAL = 32767.0

# PHY parameters
QAM_ORDER = 16
BITS_PER_SYMBOL = int(math.log2(QAM_ORDER))

# Adaptation parameters (using FEC rates)
# FEC Rates: (1, 1) -> No coding (High throughput)
#            (1, 2) -> Rate 1/2 coding (High redundancy/protection)
ADAPT_MODE_LOW = 10.0  # below this -> use stronger FEC (e.g., R=1/2)
ADAPT_MODE_HIGH = 16.0 # above this -> use weaker FEC (e.g., R=1/1)
# ===================================================

# ---------------- CRC32 Helper ----------------
def compute_crc32(data: bytes) -> int:
    """Computes the CRC32 checksum for a byte array."""
    return zlib.crc32(data) & 0xFFFFFFFF

# ---------------- Helpers: bit/byte conversions ----------------
def bytes_to_bits(b: bytes):
    """Converts a byte array to a numpy array of uint8 bits."""
    if len(b) == 0:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)

def bits_to_bytes(bits: np.ndarray):
    """Converts a numpy array of uint8 bits back to a byte array."""
    nbits = bits.size
    pad = (-nbits) % 8
    if pad:
        # Pad with zeros to the next byte boundary
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    arr = np.packbits(bits)
    return arr.tobytes()

# ---------------- Convolutional Coding (REAL) ----------------

# Standard K=3, R=1/2 generators
# G1 = 7 (octal) = [1, 1, 1]
# G2 = 5 (octal) = [1, 0, 1]
FEC_K = 3
FEC_GENERATORS = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8)
FEC_NUM_STATES = 2**(FEC_K - 1) # 4 states for K=3

def convolutional_encode(bits: np.ndarray, rate_num: int, rate_den: int):
    """
    Real Convolutional Encoder (K=3, R=1/2, G=[7, 5]).
    Appends (K-1) tail bits (zeros) to flush the encoder.
    """
    if rate_num == 1 and rate_den == 1:
        return bits # Rate 1/1 (No coding)

    if rate_num != 1 or rate_den != 2:
        # This implementation only supports R=1/2 and R=1/1
        raise NotImplementedError(f"Unsupported rate: {rate_num}/{rate_den}")

    # K-1 tail bits (zeros) for trellis termination
    tail = np.zeros(FEC_K - 1, dtype=np.uint8)
    padded_bits = np.concatenate([bits, tail])
    
    n_bits = padded_bits.size
    encoded_bits = np.empty(n_bits * 2, dtype=np.uint8)
    
    # Encoder state register (K-1 bits)
    register = np.zeros(FEC_K - 1, dtype=np.uint8)
    
    for i in range(n_bits):
        in_bit = padded_bits[i]
        
        # Form the full register state [in_bit, reg1, reg2, ...]
        full_state = np.concatenate([[in_bit], register])
        
        # Calculate output bits by dot product with generators (mod 2)
        out1 = np.dot(FEC_GENERATORS[0], full_state) % 2
        out2 = np.dot(FEC_GENERATORS[1], full_state) % 2
        
        encoded_bits[2*i] = out1
        encoded_bits[2*i + 1] = out2
        
        # Shift register
        register = np.roll(register, 1)
        register[0] = in_bit
        
    return encoded_bits

def _hamming_distance(b1, b2):
    """Calculates the Hamming distance between two bit arrays."""
    return np.sum(b1 != b2)

def viterbi_decode(encoded_bits: np.ndarray, rate_num: int, rate_den: int):
    """
    Real Viterbi Decoder (K=3, R=1/2, G=[7, 5]) using hard-decision.
    """
    if rate_num == 1 and rate_den == 1:
        return encoded_bits # Rate 1/1 (No decoding)

    if rate_num != 1 or rate_den != 2:
        raise NotImplementedError(f"Unsupported rate: {rate_num}/{rate_den}")

    # Pre-calculate expected outputs for all 8 transitions
    # (state, input) -> (output_pair)
    # States: 0=(00), 1=(01), 2=(10), 3=(11)
    # Note: State is (b_k-1, b_k-2)
    expected_outputs = {
        # (prev_state, input_bit): [out1, out2]
        (0, 0): [0, 0], # State 0 (00), In 0 -> Reg [0,0,0] -> Out (0,0) -> Next 0 (00)
        (0, 1): [1, 1], # State 0 (00), In 1 -> Reg [1,0,0] -> Out (1,1) -> Next 2 (10)
        (1, 0): [1, 1], # State 1 (01), In 0 -> Reg [0,0,1] -> Out (1,1) -> Next 0 (00)
        (1, 1): [0, 0], # State 1 (01), In 1 -> Reg [1,0,1] -> Out (0,0) -> Next 2 (10)
        (2, 0): [1, 0], # State 2 (10), In 0 -> Reg [0,1,0] -> Out (1,0) -> Next 1 (01)
        (2, 1): [0, 1], # State 2 (10), In 1 -> Reg [1,1,0] -> Out (0,1) -> Next 3 (11)
        (3, 0): [0, 1], # State 3 (11), In 0 -> Reg [0,1,1] -> Out (0,1) -> Next 1 (01)
        (3, 1): [1, 0], # State 3 (11), In 1 -> Reg [1,1,1] -> Out (1,0) -> Next 3 (11)
    }
    
    # Trellis transitions (for ACS)
    # next_state -> [ (prev_state, input_bit), (prev_state, input_bit) ]
    transitions = {
        0: [(0, 0), (1, 0)], # To state 0 (00)
        1: [(2, 0), (3, 0)], # To state 1 (01)
        2: [(0, 1), (1, 1)], # To state 2 (10)
        3: [(2, 1), (3, 1)], # To state 3 (11)
    }

    n_pairs = encoded_bits.size // 2
    
    # path_metrics: (num_states) - cost to reach each state
    path_metrics = np.full(FEC_NUM_STATES, np.inf)
    path_metrics[0] = 0.0 # Start at state 0
    
    # trellis: (num_states, n_pairs + 1) - stores survivor path (previous state)
    trellis = np.zeros((FEC_NUM_STATES, n_pairs + 1), dtype=np.int32)

    # === 1. Add-Compare-Select (ACS) ===
    for t in range(n_pairs):
        rcvd_pair = encoded_bits[2*t : 2*t + 2]
        new_path_metrics = np.full(FEC_NUM_STATES, np.inf)
        
        for next_state in range(FEC_NUM_STATES):
            min_metric = np.inf
            best_prev_state = -1
            
            # Check the two possible transitions that lead to this next_state
            for prev_state, input_bit in transitions[next_state]:
                # Branch metric: Hamming distance
                expected_out = expected_outputs[(prev_state, input_bit)]
                branch_metric = _hamming_distance(rcvd_pair, expected_out)
                
                # Path metric = old_path_metric + branch_metric
                metric = path_metrics[prev_state] + branch_metric
                
                if metric < min_metric:
                    min_metric = metric
                    best_prev_state = prev_state
            
            new_path_metrics[next_state] = min_metric
            trellis[next_state, t + 1] = best_prev_state
            
        path_metrics = new_path_metrics

    # === 2. Traceback ===
    decoded_bits = []
    
    # Find the best final state (should be 0 if tail bits worked)
    current_state = np.argmin(path_metrics)
    
    # Trace back from the end
    for t in range(n_pairs, 0, -1):
        prev_state = trellis[current_state, t]
        
        # Find the input bit that caused this transition
        # This is a property of this specific trellis structure:
        # Input 0 transitions to states 0 or 1
        # Input 1 transitions to states 2 or 3
        input_bit = 1 if current_state in [2, 3] else 0
        decoded_bits.append(input_bit)
        
        current_state = prev_state

    # Reverse the decoded bits (since we traced backward)
    decoded_bits.reverse()
    
    # Remove the (K-1) tail bits that were decoded
    num_data_bits = n_pairs - (FEC_K - 1)
    
    return np.array(decoded_bits[:num_data_bits], dtype=np.uint8)

# ---------------- 16-QAM Gray mapping/demap ----------------
_LEVELS = np.array([-3, -1, 1, 3], dtype=float)
_LEVEL_BITS = [(0,0),(0,1),(1,1),(1,0)]
_CONST_SYM = []
_CONST_BITS = []
for i_idx, ib in enumerate(_LEVEL_BITS):
    for q_idx, qb in enumerate(_LEVEL_BITS):
        I = _LEVELS[i_idx]
        Q = _LEVELS[q_idx]
        _CONST_SYM.append(complex(I, Q))
        _CONST_BITS.append([ib[0], ib[1], qb[0], qb[1]])
_CONST_SYM = np.array(_CONST_SYM, dtype=np.complex64)
# Normalize constellation power to 1
_CONST_SYM = _CONST_SYM / np.sqrt(np.mean(np.abs(_CONST_SYM)**2))
_CONST_BITS = np.array(_CONST_BITS, dtype=np.uint8)

def qam16_map(bits: np.ndarray):
    """Maps bits to 16-QAM symbols (Gray coded)."""
    assert bits.size % 4 == 0
    M = bits.size // 4
    syms = np.empty(M, dtype=np.complex64)
    # Fast approach using pre-calculated constants
    for i in range(M):
        # Convert 4 bits to an index in _CONST_BITS
        b = bits[4*i:4*i+4]
        b_tuple = tuple(b.tolist())
        try:
            idx = int(np.where((_CONST_BITS == b_tuple).all(axis=1))[0][0])
            syms[i] = _CONST_SYM[idx]
        except IndexError:
            # Fallback for when np.where is too slow or complex
            i_bits = (int(b[0]), int(b[1]))
            q_bits = (int(b[2]), int(b[3]))
            i_idx = _LEVEL_BITS.index(i_bits)
            q_idx = _LEVEL_BITS.index(q_bits)
            syms[i] = complex(_LEVELS[i_idx], _LEVELS[q_idx])
            
    # Re-normalize (already normalized in constants, but just in case)
    syms = syms / np.sqrt(np.mean(np.abs(syms)**2))
    return syms

def qam16_demod_fast(symbols):
    """Demaps symbols using Minimum Euclidean Distance (MED)."""
    bits = []
    for s in symbols:
        d = np.abs(_CONST_SYM - s)
        idx = int(np.argmin(d))
        bits.extend(_CONST_BITS[idx].tolist())
    return np.array(bits, dtype=np.uint8)

# ---------------- Channel simulation ----------------
def apply_rayleigh_and_awgn(symbols, snr_db):
    """
    Applies complex Rayleigh fading and Additive White Gaussian Noise (AWGN).
    """
    # Simple one-tap channel coefficient h
    h = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)
    
    faded = symbols * h
    sig_pow = np.mean(np.abs(faded)**2)
    snr_linear = 10**(snr_db/10.0)
    
    # Calculate noise power based on desired SNR
    noise_pow = sig_pow / snr_linear
    noise = (np.random.normal(scale=np.sqrt(noise_pow/2), size=faded.shape) +
             1j * np.random.normal(scale=np.sqrt(noise_pow/2), size=faded.shape))
    
    rx = faded + noise
    return rx.astype(np.complex64), h

# ---------------- Adaptive transmitter state ----------------
class AdaptiveState:
    """Manages the current FEC code rate based on received SNR feedback."""
    def __init__(self):
        self.lock = threading.Lock()
        self.code_rate = (1, 1) # Default: (num, den) -> (1, 1) is no coding
        self.last_reported_snr = None
        
    def get_code_rate(self):
        """Returns the current FEC code rate (num, den)."""
        with self.lock:
            return self.code_rate
    
    def update_snr(self, snr_db):
        """Updates the FEC code rate based on the reported SNR."""
        with self.lock:
            self.last_reported_snr = snr_db
            if snr_db is None:
                # If no SNR feedback, default to the strongest mode
                self.code_rate = (1, 2)
                return
            
            # ADAPTIVE LOGIC: Switch FEC rate based on SNR
            if snr_db < ADAPT_MODE_LOW:
                self.code_rate = (1, 2) # Stronger FEC (Rate 1/2)
            elif snr_db > ADAPT_MODE_HIGH:
                self.code_rate = (1, 1) # Weaker FEC (Rate 1/1 - No coding)
            else:
                self.code_rate = (1, 2) # Default/medium FEC mode

# ---------------- UDP Transmitter (16-QAM path) ----------------
def feedback_listener(feedback_port, adapt_state, stop_event):
    """Thread to listen for SNR feedback from the receiver."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", feedback_port))
    sock.settimeout(0.5)
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(1024)
            try:
                msg = json.loads(data.decode())
                est_snr = msg.get("est_snr_db", None)
                if est_snr is not None:
                    adapt_state.update_snr(float(est_snr))
            except Exception:
                pass
        except socket.timeout:
            continue
        except Exception:
            break
    sock.close()

def udp_tx_thread(host, port, phy_sim=False, snr_db=25.0, feedback_port=None):
    """Audio capture, compression, modulation, channel simulation, and UDP send loop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inp = None
    seq_num = 0
    adapt = AdaptiveState()
    stop_event = threading.Event()
    
    if feedback_port is None:
        feedback_port = port + 1
        
    fb_thread = threading.Thread(target=feedback_listener, 
                                 args=(feedback_port, adapt, stop_event), daemon=True)
    fb_thread.start()
    
    try:
        inp = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                             blocksize=FRAME_SAMPLES, dtype=np.float32)
        inp.start()
        print(f"🎤 [UDP TX] Sending to {host}:{port} | PHY_SIM={phy_sim} SNR={snr_db}dB | FB port={feedback_port}")
        
        while True:
            try:
                frame, _ = inp.read(FRAME_SAMPLES)
                i16_frame = (frame.flatten() * AUDIO_MAX_VAL).astype(AUDIO_DTYPE)
                payload = zlib.compress(i16_frame.tobytes())
                crc = compute_crc32(payload)
                
                if not phy_sim:
                    # RAW path (unmodified compressed audio)
                    header = json.dumps({
                        "seq": seq_num, "len": len(payload), "mode": "raw",
                        "src_host": host, "fb_port": feedback_port, "crc": crc
                    }).encode()
                    msg = struct.pack("!I", len(header)) + header + payload
                    sock.sendto(msg, (host, port))
                else:
                    # PHY path (16-QAM + Fading + Adaptive FEC)
                    
                    # 1. Prepare and Encode
                    bits = bytes_to_bits(payload)
                    bit_len = int(bits.size)
                    
                    rate_num, rate_den = adapt.get_code_rate()
                    encoded_bits = convolutional_encode(bits, rate_num, rate_den)
                    
                    # 2. Map
                    pad = (-encoded_bits.size) % 4
                    if pad:
                        encoded_bits = np.concatenate([encoded_bits, np.zeros(pad, dtype=np.uint8)])
                        
                    symbols = qam16_map(encoded_bits)
                    
                    # 3. Channel Simulation
                    rx_symbols, h = apply_rayleigh_and_awgn(symbols, snr_db)
                    
                    # 4. Serialize for UDP
                    inter = np.empty((rx_symbols.size * 2,), dtype=np.float32)
                    inter[0::2] = rx_symbols.real.astype(np.float32)
                    inter[1::2] = rx_symbols.imag.astype(np.float32)
                    payload_bytes = inter.tobytes()
                    
                    # Send original bits as hex for BER calculation at RX (for simulation)
                    orig_bits_hex = bits.tobytes().hex() if bit_len > 0 else ""
                    
                    header = json.dumps({
                        "seq": seq_num, "mode": "phy", "n_sym": symbols.size, 
                        "bit_len": bit_len, "h_real": h.real, "h_imag": h.imag,
                        "src_host": host, "fb_port": feedback_port, 
                        "rate_num": rate_num, "rate_den": rate_den, # Adaptive FEC Rate
                        "crc": crc, "orig_bits": orig_bits_hex
                    }).encode()
                    
                    msg = struct.pack("!I", len(header)) + header + payload_bytes
                    sock.sendto(msg, (host, port))
                    
                seq_num += 1
            except Exception as e:
                print(f"[TX ERROR] {e}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if inp: inp.stop()
        stop_event.set()
        fb_thread.join(timeout=1.0)
        sock.close()
        print("[TX] Stopped")

# ---------------- Packet Decoder ----------------
class PacketDecoder:
    """Handles decoding, FEC, decompression, and Packet Loss Concealment (PLC)."""
    def __init__(self):
        self.last_good_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        
    def decode_raw_payload(self, payload_bytes, expected_crc):
        """Decodes compressed audio from the 'raw' mode."""
        try:
            if compute_crc32(payload_bytes) != expected_crc:
                print(f"[DECODE RAW] CRC FAILED! Using PLC.")
                return self.last_good_frame.reshape(-1, 1)
                
            audio_bytes = zlib.decompress(payload_bytes)
            audio = np.frombuffer(audio_bytes, dtype=AUDIO_DTYPE).astype(np.float32) / AUDIO_MAX_VAL
            if len(audio) != FRAME_SAMPLES:
                raise ValueError("Decoded audio length mismatch")
                
            self.last_good_frame = audio
            return audio.reshape(-1, 1)
        except Exception as e:
            print(f"[DECODE RAW ERROR] {e}, using PLC")
            return self.last_good_frame.reshape(-1, 1)
            
    def decode_phy_payload(self, interleaved_bytes, n_sym, bit_len, h_real, h_imag,
                           rate_num=1, rate_den=1, expected_crc=None, orig_bits_hex=None):
        """
        Demodulates, equalizes, Viterbi decodes, and decompresses from the 'phy' mode.
        """
        try:
            # 1. Demodulation and Equalization
            arr = np.frombuffer(interleaved_bytes, dtype=np.float32)
            if arr.size != n_sym * 2:
                raise ValueError(f"PHY payload len mismatch")
                
            symbols = arr.reshape(-1, 2)
            rx_symbols = symbols[:,0] + 1j*symbols[:,1]
            h = complex(h_real, h_imag)
            eq = rx_symbols if h == 0 or np.abs(h) < 1e-6 else (rx_symbols / h)
            
            # 2. Demapping (Hard Decision)
            bits_hard_decision = qam16_demod_fast(eq)
            
            # 3. Viterbi Decoding
            decoded_bits = viterbi_decode(bits_hard_decision, rate_num, rate_den)
            
            # Trim decoded bits to the original (pre-FEC) bit length
            bits = decoded_bits[:bit_len]
            payload_bytes = bits_to_bytes(bits)
            
            # 4. CRC Check and BER calculation (for simulation)
            crc_ok = "OK"
            if expected_crc is not None and compute_crc32(payload_bytes) != expected_crc:
                crc_ok = "FAILED"
                
            error_rate = 0.0
            if orig_bits_hex:
                orig_bytes = bytes.fromhex(orig_bits_hex)
                orig_bits = np.unpackbits(np.frombuffer(orig_bytes, dtype=np.uint8))[:bit_len]
                errors = np.sum(orig_bits != bits[:len(orig_bits)])
                error_rate = errors / bit_len if bit_len > 0 else 0.0
                
            print(f"CRC check: {crc_ok} | Bit Length: {bit_len} | Rate: {rate_num}/{rate_den} | BER: {error_rate:.6f}")
            
            if crc_ok == "FAILED":
                return self.last_good_frame.reshape(-1, 1)
                
            # 5. Decompression and PLC
            audio_bytes = zlib.decompress(payload_bytes)
            audio = np.frombuffer(audio_bytes, dtype=AUDIO_DTYPE).astype(np.float32) / AUDIO_MAX_VAL
            if len(audio) != FRAME_SAMPLES:
                raise ValueError("Decoded audio length mismatch after PHY")
                
            self.last_good_frame = audio
            return audio.reshape(-1, 1)
            
        except Exception as e:
            print(f"[DECODE PHY ERROR] {e}, using PLC")
            return self.last_good_frame.reshape(-1, 1)

# ---------------- Receiver + Player Tasks ----------------
def estimate_snr_db(eq_symbols):
    """Estimates the post-equalization SNR using distance to nearest constellation point."""
    if eq_symbols.size == 0:
        return None
    
    # Calculate distance to nearest constellation point
    dists = np.abs(_CONST_SYM.reshape(1,-1) - eq_symbols.reshape(-1,1))
    idxs = np.argmin(dists, axis=1)
    nearest = _CONST_SYM[idxs]
    
    # Signal power is approximated by power of nearest constellation points
    signal_pow = np.mean(np.abs(nearest)**2)
    # Noise power is approximated by the mean squared error (MSE)
    noise_pow = np.mean(np.abs(eq_symbols - nearest)**2)
    
    if noise_pow == 0:
        return 100.0 # Effectively infinite SNR
    
    return 10 * math.log10(max(signal_pow / noise_pow, 1e-12)) # Convert to dB

async def player_task(queue, stream, decoder):
    """Plays audio frames from the jitter buffer queue."""
    loop = asyncio.get_event_loop()
    expected_seq = -1
    target_buffer_size = max(1, JITTER_BUFFER_MS // FRAME_MS)
    print(f"[PLAYER] Pre-buffering... (waiting for {target_buffer_size} packets)")
    
    while queue.qsize() < target_buffer_size:
        await asyncio.sleep(0.01)
        
    print("🚀 [PLAYER] Buffering complete, starting playback!")
    next_play_time = loop.time()
    
    while True:
        next_play_time += (FRAME_MS / 1000.0)
        frame_to_play = None
        
        try:
            # Wait for packet, but not longer than needed for smooth playback
            timeout = max(0, next_play_time - loop.time() - 0.005)
            seq, meta = await asyncio.wait_for(queue.get(), timeout=timeout)
            
            if expected_seq == -1:
                expected_seq = seq
                
            if seq > expected_seq:
                # Packet loss detected (or severe reordering)
                missing = seq - expected_seq
                print(f"[PLAYER] Packet loss: Missed {missing} packet(s). Using PLC.")
                for _ in range(missing):
                    stream.write(decoder.last_good_frame.reshape(-1,1))
                    
            elif seq < expected_seq:
                # Discard old packet (late arrival)
                print(f"[PLAYER] Discarding old packet {seq} (expected {expected_seq})")
                next_play_time -= (FRAME_MS / 1000.0)
                continue
            
            # Decode the current packet
            mode = meta['mode']
            if mode == 'raw':
                frame_to_play = decoder.decode_raw_payload(meta['payload'], meta['crc'])
            else:
                frame_to_play = decoder.decode_phy_payload(
                    meta['payload'], meta['n_sym'], meta['bit_len'],
                    meta['h_real'], meta['h_imag'], 
                    rate_num=meta.get('rate_num', 1), rate_den=meta.get('rate_den', 1),
                    expected_crc=meta.get('crc'), orig_bits_hex=meta.get('orig_bits')
                )
            expected_seq = seq + 1
            
        except asyncio.TimeoutError:
            print("[PLAYER] Buffer underrun! Playing PLC.")
            frame_to_play = decoder.last_good_frame.reshape(-1,1)
            if expected_seq != -1:
                expected_seq += 1
                
        stream.write(frame_to_play)
        # Sleep until the next frame's play time
        await asyncio.sleep(max(0, next_play_time - loop.time()))

async def receiver_task(port, queue):
    """Receives UDP packets and puts metadata/payload into the queue."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    print(f"📡 [RECEIVER] Listening on 0.0.0.0:{port}")
    buffer = b""
    loop = asyncio.get_event_loop()
    
    while True:
        try:
            # Read from socket asynchronously
            data, addr = await loop.run_in_executor(None, sock.recvfrom, 65536)
            buffer += data
            sender_host = addr[0]
        except BlockingIOError:
            await asyncio.sleep(0.001)
            continue
        except Exception as e:
            print(f"[RX NET ERROR] {e}")
            return
            
        # Parse packets from buffer
        while len(buffer) >= 4:
            try:
                hdr_len = struct.unpack("!I", buffer[:4])[0]
                if len(buffer) < 4 + hdr_len:
                    break # Not enough data for header
                
                header_bytes = buffer[4:4+hdr_len]
                header = json.loads(header_bytes.decode())
                seq = header.get("seq", 0)
                mode = header.get("mode", "raw")
                fb_port = header.get("fb_port", None)
                src_host = header.get("src_host", sender_host)
                crc = header.get("crc", 0)
                
                if mode == "raw":
                    payload_len = header["len"]
                    if len(buffer) < 4 + hdr_len + payload_len:
                        break
                    
                    payload_bytes = buffer[4+hdr_len:4+hdr_len+payload_len]
                    await queue.put((seq, {"mode":"raw", "payload": payload_bytes, "crc": crc}))
                    buffer = buffer[4+hdr_len+payload_len:]
                    
                elif mode == "phy":
                    n_sym = int(header["n_sym"])
                    rate_num = int(header.get("rate_num", 1))
                    rate_den = int(header.get("rate_den", 1))
                    bit_len = int(header.get("bit_len", 0))
                    expected_bytes = n_sym * 2 * 4 # complex symbols * 2 floats * 4 bytes/float
                    
                    if len(buffer) < 4 + hdr_len + expected_bytes:
                        break
                        
                    payload_bytes = buffer[4+hdr_len:4+hdr_len+expected_bytes]
                    
                    # ---- SNR feedback generation ----
                    arr = np.frombuffer(payload_bytes, dtype=np.float32)
                    if arr.size == n_sym * 2:
                        symbols = arr.reshape(-1,2)
                        rx_symbols = symbols[:,0] + 1j*symbols[:,1]
                        h = complex(header.get("h_real",0.0), header.get("h_imag",0.0))
                        eq = rx_symbols if h == 0 or np.abs(h) < 1e-6 else (rx_symbols / h)
                        
                        est_snr = estimate_snr_db(eq) # Estimate SNR on equalized symbols
                        
                        if fb_port is not None:
                            feedback = {"seq": seq, "est_snr_db": est_snr}
                            try:
                                # Send non-blocking feedback
                                fb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                fb_sock.sendto(json.dumps(feedback).encode(), (src_host, int(fb_port)))
                                fb_sock.close()
                            except Exception:
                                pass
                                
                    orig_bits_hex = header.get("orig_bits", "")
                    await queue.put((seq, {
                        "mode":"phy", "payload": payload_bytes, "n_sym": n_sym,
                        "bit_len": bit_len, "h_real": header.get("h_real",0.0),
                        "h_imag": header.get("h_imag",0.0), "crc": crc,
                        "rate_num": rate_num, "rate_den": rate_den, 
                        "orig_bits": orig_bits_hex
                    }))
                    buffer = buffer[4+hdr_len+expected_bytes:]
                    
                else:
                    print(f"[RX] Unknown mode {mode}, discarding packet.")
                    buffer = buffer[4+hdr_len:]
                    
            except Exception as e:
                print(f"[RX PARSE ERROR] {e}. Discarding buffer.")
                buffer = b""

async def udp_rx_main(port):
    """Initializes receiver and player tasks."""
    jitter_buffer = asyncio.PriorityQueue()
    decoder = PacketDecoder()
    out_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SAMPLES, dtype=np.float32)
    out_stream.start()
    
    try:
        await asyncio.gather(
            receiver_task(port, jitter_buffer),
            player_task(jitter_buffer, out_stream, decoder)
        )
    except KeyboardInterrupt:
        print("\n[RX] Stopping tasks...")
    finally:
        out_stream.stop()
        print("[RX] Stopped")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive UDP Voice with 16-QAM and FEC Simulation.")
    parser.add_argument("--role", choices=["tx", "rx"], required=True, help="Run as transmitter (tx) or receiver (rx).")
    parser.add_argument("--host", default="127.0.0.1", help="Destination host IP for TX.")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="UDP port for data transmission/reception.")
    parser.add_argument("--phy-sim", action="store_true", help="Enable PHY layer simulation (16QAM+Adaptive FEC+fading+awgn).")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB for simulated channel (TX only).")
    parser.add_argument("--fb-port", type=int, default=None, help="Feedback port for TX (default port+1).")
    args = parser.parse_args()

    if args.role == "rx":
        try:
            asyncio.run(udp_rx_main(args.port))
        except KeyboardInterrupt:
            print("\n[RX] Quit")
    else:
        tx_thread = threading.Thread(target=udp_tx_thread,
                                     args=(args.host, args.port, args.phy_sim, args.snr, args.fb_port),
                                     daemon=True)
        tx_thread.start()
        print(f"🎙️ [TX] Running... To use FEC/PHY sim: Run RX with '--phy-sim' first, then TX with '--phy-sim'.")
        print("Press Ctrl+C to stop.")
        try:
            while tx_thread.is_alive():
                tx_thread.join(1)
        except KeyboardInterrupt:
            print("\n[TX] Stopping...")