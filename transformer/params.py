from data_set import WORDS


NUM_BLOCKS         = 3           # 簡単なタスクなので、Attention is all you needの半分
D_MODEL            = 256         # 簡単なタスクなので、Attention is all you needの半分
D_FF               = 1024        # 簡単なタスクなので、Attention is all you needの半分
NUM_HEADS          = 4           # 簡単なタスクなので、Attention is all you needの半分
DROPOUT_RATE       = 0.1         # ここはAttention is all you needのまま
X_VOCAB_SIZE       = len(WORDS)
Y_VOCAB_SIZE       = len(WORDS)  # 出力には演算記号はないのだけど、面倒なので含めておきます
X_MAXIMUM_POSITION = 100         # 余裕を持って多めに
Y_MAXIMUM_POSITION = 100         # 余裕を持って多めに
