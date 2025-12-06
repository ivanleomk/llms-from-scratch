from .bpe_in_place import train_bpe as bpe_in_place
from .bpe_in_place_parallel import train_bpe as bpe_in_place_parallel
from .bpe_inverted_index import train_bpe as bpe_inverted_index
from .bpe_naive import train_bpe as bpe_naive
from .bpe_parallel import train_bpe as bpe_parallel

IMPLEMENTATIONS = {
    "bpe_in_place": bpe_in_place,
    "bpe_in_place_parallel": bpe_in_place_parallel,
    "bpe_inverted_index": bpe_inverted_index,
    "bpe_naive": bpe_naive,
    "bpe_parallel": bpe_parallel,
}
