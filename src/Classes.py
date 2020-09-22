from collections import deque

class Artwork:
    def __init__(self, id_, upload_date, **kwargs):
        self.id = id_
        self.upload_date = upload_date
        for key, val in kwargs.items():
            setattr(self, key, val)

class Customer:
    def __init__(self, cid=None):
        self.purchase_sessions = []
        self.reset_profile(None)
        self.cid = cid

    def append_purchase_session(self, pur_sess):
        self.purchase_sessions.append(pur_sess)
        
    def reset_profile(self, profile):
        self.last_sess_idx = -1
        self.num_items_consumed = 0
        self.num_purchase_sessions = 0
        self.profile = profile
        
    def consume_next_purchase_session(self):
        # update internal variables
        self.last_sess_idx += 1        
        pur_sess = self.purchase_sessions[self.last_sess_idx]
        self.num_items_consumed += len(pur_sess.artwork_ids)
        self.num_purchase_sessions += 1

        # update profile
        if self.profile:
            self.profile.update(pur_sess)
    
    def can_test_next_session(self):
        assert self.last_sess_idx + 1 < len(self.purchase_sessions)
        # profile must be ready (if not None)
        return self.profile is None or self.profile.ready()

    def get_next_session_ids(self):
        return set(self.purchase_sessions[self.last_sess_idx + 1].artwork_ids)
    
    def get_all_future_session_ids(self):
        ids = set()
        for i in range(self.last_sess_idx+1, len(self.purchase_sessions)):
            ids.update(self.purchase_sessions[i].artwork_ids)
        assert len(ids) > 0
        return ids
    
class UploadEvent:
    def __init__(self, timestamp, artwork_id, **kwargs):
        self.type = 'upload'
        self.timestamp = timestamp
        self.artwork_id = artwork_id
        for key, val in kwargs.items():
            setattr(self, key, val)
    def __str__(self):
        return "UploadEvent(timestamp=%s, artwork_id=%s)" % (self.timestamp, self.artwork_id)

class PurchaseSessionEvent:
    def __init__(self, timestamp, artwork_ids, customer_id, **kwargs):
        self.type = 'purchase'
        self.timestamp = timestamp
        self.artwork_ids = artwork_ids
        self.customer_id = customer_id
        for key, val in kwargs.items():
            setattr(self, key, val)
    def __str__(self):
        return "PurchaseSessionEvent(timestamp=%s, customer_id=%s, n_artworks=%d)" % (
            self.timestamp, self.customer_id, len(self.artwork_ids))
            
class ProfileBase:
    
    def __init__(self, maxprofsize, artworks_dict):
        self.consumed_artworks = set()
        self.id2count = dict()
        self.maxprofsize = maxprofsize
        self.artworks_dict = artworks_dict
        
        if maxprofsize is not None:
            if maxprofsize > 0:
                self.purch_sess_queue = deque()
            else:
                self.purch_sess_count = 0

    def handle_artwork_added(self, artwork):
        raise NotImplementedError("Please Implement this instance method")

    def handle_artwork_removed(self, artwork):
        raise NotImplementedError("Please Implement this instance method")
    
    @classmethod
    def global_purchase_session_event_handler(cls, purch_sess):
        raise NotImplementedError("Please Implement this classmethod")
            
    def update(self, purch_sess):

        # call the global handler, so that non-personalized updates can be performed too
        self.global_purchase_session_event_handler(purch_sess)

        add_to_profile = True

        # check if there is a maxprofsize
        if self.maxprofsize is not None:

            if self.maxprofsize > 0:

                # append purchase session to the queue
                q = self.purch_sess_queue
                q.append(purch_sess)
                
                # if there is overflow, remove last purchase session ids from memory
                if len(q) > self.maxprofsize:
                    popped_purch_sess = q.popleft()
                    for _id in popped_purch_sess.artwork_ids:
                        count = self.id2count[_id] - 1
                        assert count >= 0
                        if count == 0:
                            del self.id2count[_id]
                            artwork = self.artworks_dict[_id]
                            self.consumed_artworks.remove(artwork)
                            self.handle_artwork_removed(artwork)
            else:                
                self.purch_sess_count += 1
                if self.purch_sess_count > abs(self.maxprofsize):
                    add_to_profile = False
        
        # add latest purchase session ids into memory
        if add_to_profile:
            for _id in purch_sess.artwork_ids:
                self.id2count[_id] = self.id2count.get(_id, 0) + 1
                artwork = self.artworks_dict[_id]
                self.consumed_artworks.add(artwork)
                self.handle_artwork_added(artwork)

    def rank_inventory_ids(self, inventory_artworks):
        raise NotImplementedError("Please Implement this instance method")