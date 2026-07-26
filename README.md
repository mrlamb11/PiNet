Model type:        single-cell recurrent net, h ← W2 · relu(W1 · h + b1) + b2
Activation:        ReLU

State vector h:    63 registers   (int, unbounded precision)
Hidden layer a:    221 units      (post-ReLU activations)

W1  (hidden × state):   549 nonzero entries   [sparse]
b1  (hidden,):          159 nonzero entries   [sparse]
W2  (state × hidden):   221 nonzero entries   [sparse]
b2  (state,):           0 nonzero entries     [all zero]

Total nonzero parameters:  929

Weight magnitude range:    1  →  ~10^(bit-length of M)
  M (comparator threshold): sized per compile-time capacity
  compiled capacity:        10,000 digits
  M bit-length @ capacity:  ~23,780,000 bits (~2,973,137 bytes)

Numeric encoding:   arbitrary-precision int
  format:  sign (int8) + big-endian magnitude bytes,
           addressed via (offset, length) into one shared byte buffer
  dtype inventory: int64 (indices/offsets/lengths), int8 (signs), uint8 (byte buffer)

File format:        .safetensors
  tensor count:      20
  metadata:          none
  file size:          3,004,240 bytes  (~3.0 MB)

Tensors:
  buffer            uint8   [2,973,137]
  w1.row            int64   [549]
  w1.col            int64   [549]
  w1.val.sign       int8    [549]
  w1.val.offset     int64   [549]
  w1.val.length     int64   [549]
  b1.sign           int8    [221]
  b1.offset         int64   [221]
  b1.length         int64   [221]
  w2.row            int64   [221]
  w2.col            int64   [221]
  w2.val.sign       int8    [221]
  w2.val.offset     int64   [221]
  w2.val.length     int64   [221]
  b2                int64   [63]
  idx.init          int64   [1]
  idx.done          int64   [1]
  idx.kcnt          int64   [1]
  idx.Jreg          int64   [1]
  idx.PI            int64   [1]

Runtime inputs (not weights):  digit count → kcnt (term count), Jreg (working precision)
Output:                        register PI, read after halt (register `done` = 1)
Depth:                         O(n_terms · 128) recurrent steps (unrolled, sequential)
