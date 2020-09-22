import numpy as np
import heapq

NUMPY_FLOAT64_MIN = np.finfo(np.float64).min

def avgsimtopk(art, arts, sim, k):
    n = len(arts)
    if n <= k:
        return sum(sim(art, a) for a in arts) / n
    else:
        h = []
        totsim = 0
        for a in arts:
            s = sim(art, a)
            totsim += s
            if len(h) < k:
                heapq.heappush(h, s)
            else:
                totsim -= heapq.heappushpop(h, s)
        return totsim / k

def avg_similarity(art, arts, sim):
    totsim = 0
    for a in arts:
        totsim += sim(art, a)
    return totsim / len(arts)

def max_similarity(art, arts, sim):
    maxsim = NUMPY_FLOAT64_MIN
    for a in arts:
        maxsim = max(maxsim, sim(art, a))
    return maxsim

def append_avgsimtopk(simfuncs, pairwise_simfunc, k):
    simfuncs.append(lambda art, arts: avgsimtopk(art, arts, pairwise_simfunc, k))

def append_simfunc_and_tags(profile_simfuncs, simfunc_tags, pairwise_simfunc, tag, ks):
    for k in ks:
        if k is None:
            profile_simfuncs.append(lambda art, arts: avg_similarity(art, arts, pairwise_simfunc))
            simfunc_tags.append('{}-avgsim'.format(tag))
        elif k == 1:
            profile_simfuncs.append(lambda art, arts: max_similarity(art, arts, pairwise_simfunc))
            simfunc_tags.append('{}-maxsim'.format(tag))
        else:
            append_avgsimtopk(profile_simfuncs, pairwise_simfunc, k)
            simfunc_tags.append('{}-avgsmtp{}'.format(tag, k))


def sanity_check_purchase_upload_events(events, artworks_dict):
    # test event types
    purchase_count = 0
    upload_count = 0
    for event in events:
        assert event.type == 'purchase' or event.type == 'upload'
        if event.type == 'purchase':
            purchase_count += 1
        else:
            upload_count += 1
    assert upload_count > 0 and purchase_count > 0
    print('CHECK: event types are correct')

    # test timestamp order
    last_e = None
    for event in events:
        if last_e is not None:
            assert last_e.timestamp <= event.timestamp
        last_e = event
    print('CHECK: events ordered by timestamp')

    # test: a product cannot be uploaded twice
    uploaded_ids = set()
    for event in events:
        if event.type == 'upload':
            assert event.artwork_id not in uploaded_ids
            uploaded_ids.add(event.artwork_id)
    assert len(uploaded_ids) > 0
    print('CHECK: products are only uploaded once')

    # test: products can only be purchased if present in inventory
    inventory = set()
    for event in events:
        if event.type == 'upload':
            inventory.add(event.artwork_id)
        else:
            for _id in event.artwork_ids:
                assert _id in inventory
                if artworks_dict[_id].original:
                    inventory.remove(_id)
    print('CHECK: products can only be purchased if present in inventory')
    
def recommendations_to_csv(recommendations, filename):
    import os
    import csv
    
    # make sure directories exist along the path
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # extract headers
    headers = recommendations[0].keys()

    # generate csv file
    with open(filename, 'w') as f:
        dict_writer = csv.DictWriter(f, headers, lineterminator='\n')
        dict_writer.writeheader()
        dict_writer.writerows(recommendations)
        print('** recommendations successfully saved to', filename)

def ground_truth_rank_indexes(ranked_inventory_ids, gt_ids_set):
    indexes = []
    for i, _id in enumerate(ranked_inventory_ids):
        if _id in gt_ids_set:
            indexes.append(i)
    return indexes
        
def run_personalized_recommendation_experiment(
        artworks_dict, customers_dict, time_events,
        create_profile_func, rec_size):

    assert isinstance(rec_size, int) and rec_size > 0
    
    print("---------- starting experiment ------------")    
    import time
    start_time = time.clock()
    inventory_artworks = set()
    n_tests = 0
    exp_recommendations = []

    # -- reset personalized recommendation variables ----
    
    # reset profiles
    for customer in customers_dict.values():
        new_profile = create_profile_func(customer.cid)
        customer.reset_profile(new_profile)

    # --- simulate events through time ----
    for event in time_events:

        # upload event
        if event.type == "upload":
            artwork = artworks_dict[event.artwork_id]
            inventory_artworks.add(artwork)

        # purchase event
        else:
            cid = event.customer_id
            customer = customers_dict[cid]

            # -- try to perform test before consuming purchase session ----
            if customer.can_test_next_session():
                
                # -- record recommendation --
                # rec_ids_list = customer.profile.get_recommendation(inventory_artworks, rec_size)
                ranked_inventory_ids = customer.profile.rank_inventory_ids(inventory_artworks)
                assert len(ranked_inventory_ids) == len(inventory_artworks)
                assert len(ranked_inventory_ids) >= rec_size
                rec_ids_list = ranked_inventory_ids[:rec_size]
                gt_ids_set = customer.get_all_future_session_ids()
                gt_indexes = ground_truth_rank_indexes(ranked_inventory_ids, gt_ids_set)
                assert 0 < len(gt_indexes) <= len(gt_ids_set)
                exp_recommendations.append(dict(
                    timestamp=event.timestamp,
                    customer_id=cid,
                    recommended_ids='|'.join(map(str, rec_ids_list)),
                    ground_truth_indexes='|'.join(map(str, gt_indexes)),
                    inventory_size=len(inventory_artworks),
                ))

                # -- print some feedback --
                n_tests += 1
                if n_tests % 500 == 0:
                    print("%d tests done! elapsed time: %.2f seconds" % (n_tests, time.clock()-start_time))

            # -- consume purchase session --------            
            customer.consume_next_purchase_session() # update customer's profile

            # -- remove original artworks from stock ---
            for _id in event.artwork_ids:
                artwork = artworks_dict[_id]
                if artwork.original:
                    inventory_artworks.remove(artwork)
    
    # -- last feedback ---
    print("%d tests done! elapsed time: %.2f seconds" % (n_tests, time.clock()-start_time))
    
    # -- return recommendations --
    return exp_recommendations

def list_filepaths(rootDir):
    import os
    fileList = list()
    for dir_, _, files in os.walk(rootDir):
        for fileName in files:
            relDir = os.path.relpath(dir_, rootDir)
            relFile = os.path.join(relDir, fileName)
            fileList.append(relFile)
    return fileList

def filter_regex(strings, pattern):
    import re
    r = re.compile(pattern)
    return [s for s in strings if r.match(s)]

def filter_best_key(keys, regex_pattern, score_getter):
    return max(filter_regex(keys, regex_pattern), key=score_getter)