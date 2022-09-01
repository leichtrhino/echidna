
import typing as tp
import torch

from bisect import bisect

def _split(x : torch.Tensor,
           top_db,
           frame_length,
           hop_length):

    powerspecgram = torch.stft(
        x,
        n_fft=frame_length,
        hop_length=hop_length,
        window=torch.ones(frame_length).to(x.device),
        return_complex=True,
    ).abs() ** 2
    rms = torch.sqrt(torch.mean(powerspecgram, dim=-2))
    db = 20 * torch.log10(rms)
    nosilent_frame = db > -top_db

    nosilent_frame = torch.cat((
        torch.zeros(1, dtype=bool, device=x.device),
        nosilent_frame,
        torch.zeros(1, dtype=bool, device=x.device)
    ), dim=-1)
    nosilent_samples = torch.nonzero(torch.diff(nosilent_frame, 1)) \
        * hop_length
    nosilent_samples = nosilent_samples.clamp(max=x.shape[-1])
    return nosilent_samples.reshape(-1, 2).tolist()


def merge_activation(base_list : tp.List[tp.Tuple[int, int, tp.List[str]]],
                     x : torch.Tensor,
                     tag : str,
                     top_db : float=60,
                     frame_length : int=2048,
                     hop_length : int=512) -> tp.List[tp.Tuple[int, int, str]]:
    # initial state of activation_list is [(0, length, [])]
    # calculate activation from silence
    activations = _split(x, top_db, frame_length, hop_length)

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

