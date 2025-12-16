# Adaptive UDP Voice Streaming with 16‑QAM and Adaptive FEC

This project is a **real‑time UDP voice streaming system** with an optional **physical‑layer (PHY) simulation** that includes **16‑QAM modulation**, **Rayleigh fading**, **AWGN**, **CRC checking**, **packet loss concealment (PLC)**, and **adaptive convolutional FEC (Viterbi decoded)** based on **receiver SNR feedback**.

It is intended for **research, simulation, and academic demonstrations** of cross‑layer design (PHY ↔ application) rather than production VoIP use.

---

## ✨ Features

* 🎙️ Real‑time microphone capture and playback
* 📦 UDP packetization with sequence numbers
* 🔁 Jitter buffer with PLC
* 🧮 CRC‑32 error detection
* 📡 Optional PHY simulation:

  * 16‑QAM (Gray coded)
  * Rayleigh flat fading channel
  * AWGN
  * Hard‑decision demodulation
* 🛡️ Convolutional FEC (K=3, R=1/2) + Viterbi decoding
* 📊 BER and CRC monitoring
* 🔄 **Adaptive FEC** using receiver‑estimated SNR feedback

---

## 📁 Project Structure

```
.
├── adaptive_udp_voice.py   # Main TX/RX implementation
├── README.md               # This file
```

(You may rename `adaptive_udp_voice.py` as needed.)

---

## 🔧 Requirements

Python **3.9+** is recommended.

Install dependencies:

```bash
pip install numpy sounddevice
```

> ⚠️ `sounddevice` requires **PortAudio**:
>
> * **Windows**: usually works via pip
> * **Linux**: `sudo apt install portaudio19-dev`
> * **macOS**: `brew install portaudio`

---

## 🚀 Usage

### 1️⃣ Start the Receiver

```bash
python adaptive_udp_voice.py --role rx --port 8765
```

For PHY simulation:

```bash
python adaptive_udp_voice.py --role rx --port 8765 --phy-sim
```

---

### 2️⃣ Start the Transmitter

```bash
python adaptive_udp_voice.py --role tx --host 127.0.0.1 --port 8765
```

With PHY simulation and adaptive FEC:

```bash
python adaptive_udp_voice.py --role tx --host 127.0.0.1 --port 8765 --phy-sim --snr 20
```

---

## 🔁 Adaptive FEC Logic

The transmitter dynamically selects the FEC rate based on receiver feedback:

| Estimated SNR (dB) | FEC Rate                |
| ------------------ | ----------------------- |
| < 10 dB            | 1/2 (strong protection) |
| 10–16 dB           | 1/2                     |
| > 16 dB            | 1/1 (no coding)         |

SNR is estimated at the receiver using **minimum‑distance constellation error** after equalization.

---

## 📊 Runtime Diagnostics

The receiver prints:

* CRC status
* Bit length
* Selected FEC rate
* Bit Error Rate (BER)

Example:

```
CRC check: OK | Bit Length: 2560 | Rate: 1/2 | BER: 0.000312
```

---

## ⚠️ Notes & Limitations

* This is a **simulation‑oriented design**, not optimized for real networks
* PHY simulation is performed **at the transmitter** for research convenience
* No encryption or authentication
* Hard‑decision demodulation only (no soft Viterbi metrics)

---

## 📚 Educational Use

This project is suitable for:

* Digital communications labs
* Wireless PHY simulations
* Cross‑layer system demonstrations
* FEC and modulation experiments

---

## 📜 License

MIT License – free to use, modify, and distribute for academic and personal projects.

---

## 🙌 Acknowledgment

Inspired by classical digital communications theory:

* Proakis – *Digital Communications*
* Sklar – *Digital Communications: Fundamentals and Applications*

---
