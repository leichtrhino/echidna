
import typing as tp
import torch
import librosa

from bisect import bisect

def merge_activation(base_list : tp.List[tp.Tuple[int, int, tp.List[str]]],
                     x : torch.Tensor,
                     tag : str,
                     top_db : float=60,
                     frame_length : int=2048,
                     hop_length : int=512) -> tp.List[tp.Tuple[int, int, str]]:
    # initial state of activation_list is [(0, length, [])]
    # calculate activation from silence
    activations = librosa.effects.split(x.numpy(),
                                        top_db=top_db,
                                        frame_length=frame_length,
                                        hop_length=hop_length)

    if base_list[-1][1] is None:
        base_list[-1] = (base_list[-1][0], x.shape[-1], base_list[-1][2])

    for a_f, a_t in activations:
        # find leftmost index
        i = bisect([b[0] for b in base_list], a_f) - 1

        # divide current section into two segments
        if i < len(base_list) and base_list[i][0] < a_f:
            b_s, b_t, b_tag = base_list[i]
            base_list[i] = (b_s, a_f, b_tag)
            base_list.insert(i+1, (a_f, b_t, b_tag[:]))
            i += 1

        # while adding tag to list, find rightmost index
        while i < len(base_list) and base_list[i][1] <= a_t:
            base_list[i][2].append(tag)
            i += 1

        # divide current section into two segments
        if i < len(base_list) and base_list[i][0] < a_t:
            b_s, b_t, b_tag = base_list[i]
            base_list[i] = (a_t, b_t, b_tag)
            base_list.insert(i, (b_s, a_t, b_tag+[tag]))

