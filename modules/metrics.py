import torch
import torch.nn.functional as F
import scipy.stats
import numpy as np
import concurrent.futures
import time


# Returns list of retrieved top k videos based on the sims matrix
def get_retrieved_videos(sims, k):
    argm = np.argsort(-sims, axis=1)
    topk = argm[:,:k].reshape(-1)
    retrieved_videos = np.unique(topk)
    return retrieved_videos

# Returns list of indices to normalize from sims based on videos
def get_index_to_normalize(sims, videos):
    argm = np.argsort(-sims, axis=1)[:,0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result

def qb_norm(train_test, test_test, k=1, beta=50):
    k = k
    beta = beta
    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test*beta)
    test_test = np.exp(test_test*beta)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    print(normalizing_sum.shape)
    print(len(index_for_normalizing))
    print(test_test_normalized.shape)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized


def beat_align_score(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return 0
    ba = 0
    for bb in gt:
        ba += np.exp(-np.min(((pred - bb))**2) / 2 / 9)
    return (ba / len(gt))

def comp(music_beat, motion_beats):
    similarity_matrix = np.apply_along_axis(comp1, 1, motion_beats, music_beat)
    return similarity_matrix

def comp1(motion_beat, music_beat):
    similarity = beat_align_score(music_beat, motion_beat)
    return similarity

def beat_similarity(music_beats, motion_beats):
    similarity_matrix = np.apply_along_axis(comp, 1, music_beats, motion_beats)
    return similarity_matrix


def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type='avg'):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type='avg'):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids)
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)
        
    return sims


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = {}

    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    # num_vids x max_text_per_vid x embed_dim
    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if pooling_type == 'avg':
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            # num_vids x max_text_per_vid x embed_dim
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        # num_vids x num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal

    stacked_sims = sims.permute(1,0,2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    # torch.save(stacked_sims, './vis/similarity.ckpt')
    # torch.save(sims_sort, './vis/sort.ckpt')
    # torch.save(valid_ranks, './vis/rank.ckpt')

    # path = './data/Music-Dance'
    # test_data_filename = open(path+'/test.txt').readlines()
    # test_data_filename = [x.rstrip('\n') for x in test_data_filename]
    # test_data_filename = np.array(test_data_filename)

    # k = 50
    # mismatch_id = sims_sort[0][:, :k]
    # query_gt_sum = 0
    # total_top1_score, total_top3_score, total_top5_score, total_top10_score, total_top20_score, total_top50_score = 0, 0, 0, 0, 0, 0
    # for index in range(test_data_filename.shape[0]):
    #     gt_filename = test_data_filename[index]
    #     queries = np.array(mismatch_id[index])
    #     query_filenames = np.array(test_data_filename[queries])
    #     gt_beat = torch.load(path+'/music_beat_openpose/'+gt_filename+'.pt')['music_beat']
    #     gt = np.array(torch.where(gt_beat > 0)[0])
    #     query_gt_beat = torch.load(path+'/music_beat_openpose/'+gt_filename+'.pt')['music_beat']
    #     query_gt = np.array(torch.where(query_gt_beat > 0)[0])
    #     query_gt_sum += beat_align_score(query_gt, gt)
    #     # print(query_gt)
    #     # print(gt)
    #     # import sys
    #     # sys.exit()
    #     topk_score_sum, cou = 0, 0
    #     for query_file in query_filenames:
    #         query_beat = torch.load(path+'/music_beat_openpose/'+str(query_file)+'.pt')['music_beat']
    #         query = np.array(torch.where(query_beat > 0)[0])
    #         score = beat_align_score(query, gt)
    #         # print(score)
    #         if score == 0:
    #             continue
    #         topk_score_sum += score
    #         cou += 1
    #         if cou == 1:
    #             top1_score = topk_score_sum / cou
    #         elif cou == 3:
    #             top3_score = topk_score_sum / cou
    #         elif cou == 5:
    #             top5_score = topk_score_sum /cou
    #         elif cou == 10:
    #             top10_score = topk_score_sum / cou
    #         elif cou == 20:
    #             top20_score = topk_score_sum / cou
    #         elif cou == 50:
    #             top50_score = topk_score_sum / cou
    #     total_top1_score += top1_score
    #     total_top3_score += top3_score
    #     total_top5_score += top5_score
    #     total_top10_score += top10_score
    #     total_top20_score += top20_score
    #     total_top50_score += top50_score

    # print('Mean top1 score', 100 * total_top1_score/test_data_filename.shape[0])
    # print('Mean top3 score', 100 * total_top3_score/test_data_filename.shape[0])
    # print('Mean top5 score', 100 * total_top5_score/test_data_filename.shape[0])
    # print('Mean top10 score', 100 * total_top10_score/test_data_filename.shape[0])
    # print('Mean top20 score', 100 * total_top20_score/test_data_filename.shape[0])
    # print('Mean top50 score', 100 * total_top50_score/test_data_filename.shape[0])
    # print('GT', 100 * query_gt_sum/test_data_filename.shape[0])
    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    # torch.save(sims_sort, './vis/v2m_sort.ckpt')
    # torch.save(ranks, './vis/v2m_rank.ckpt')

    # path = './data/Music-Dance'
    # test_data_filename = open(path+'/test.txt').readlines()
    # test_data_filename = [x.rstrip('\n') for x in test_data_filename]
    # test_data_filename = np.array(test_data_filename)

    # k = 50
    # mismatch_id = sims_sort[:, :k]
    # query_gt_sum = 0
    # total_top1_score, total_top3_score, total_top5_score, total_top10_score, total_top20_score, total_top50_score = 0, 0, 0, 0, 0, 0
    # for index in range(test_data_filename.shape[0]):
    #     gt_filename = test_data_filename[index]
    #     queries = np.array(mismatch_id[index])
    #     query_filenames = np.array(test_data_filename[queries])
    #     gt_beat = torch.load(path+'/video_beat_openpose/'+gt_filename+'.pt')['video_beat']
    #     gt = np.array(torch.where(gt_beat > 0)[0])
    #     query_gt_beat = torch.load(path+'/music_beat_openpose/'+gt_filename+'.pt')['music_beat']
    #     query_gt = np.array(torch.where(query_gt_beat > 0)[0])
    #     query_gt_sum += beat_align_score(query_gt, gt)
    #     topk_score_sum, cou = 0, 0
    #     for query_file in query_filenames:
    #         query_beat = torch.load(path+'/music_beat_openpose/'+str(query_file)+'.pt')['music_beat']
    #         query = np.array(torch.where(query_beat > 0)[0])
    #         score = beat_align_score(query, gt)
    #         topk_score_sum += score
    #         cou += 1
    #         if cou == 1:
    #             top1_score = topk_score_sum / cou
    #         elif cou == 3:
    #             top3_score = topk_score_sum / cou
    #         elif cou == 5:
    #             top5_score = topk_score_sum /cou
    #         elif cou == 10:
    #             top10_score = topk_score_sum / cou
    #         elif cou == 20:
    #             top20_score = topk_score_sum / cou
    #         elif cou == 50:
    #             top50_score = topk_score_sum / cou
    #     total_top1_score += top1_score
    #     total_top3_score += top3_score
    #     total_top5_score += top5_score
    #     total_top10_score += top10_score
    #     total_top20_score += top20_score
    #     total_top50_score += top50_score

    # print('Mean top1 score', 100 * total_top1_score/test_data_filename.shape[0])
    # print('Mean top3 score', 100 * total_top3_score/test_data_filename.shape[0])
    # print('Mean top5 score', 100 * total_top5_score/test_data_filename.shape[0])
    # print('Mean top10 score', 100 * total_top10_score/test_data_filename.shape[0])
    # print('Mean top20 score', 100 * total_top20_score/test_data_filename.shape[0])
    # print('Mean top50 score', 100 * total_top50_score/test_data_filename.shape[0])
    # print('GT', 100 * query_gt_sum/test_data_filename.shape[0])

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    metrics["AVG"] = (metrics["R1"] + metrics["R5"] + metrics["R10"] + metrics["R50"] + metrics["R100"] - metrics["MedR"] / 10 - metrics["MeanR"] / 10)
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input
